# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math

import torch
import triton.language as tl

logger = logging.getLogger(__name__)


def _repad3d_bwd_gather(
    grad_out_ptr,
    grad_in_ptr,
    ACC: tl.constexpr,
    D_in: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    D_out: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    PF: tl.constexpr,
    PT: tl.constexpr,
    PL: tl.constexpr,
    PR: tl.constexpr,
    MAX_D: tl.constexpr,
    MAX_H: tl.constexpr,
    MAX_WP: tl.constexpr,
    MAX_WB: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # grid: (cdiv(W_in, BLOCK_W), H_in * B * D_in)
    # One program handles BLOCK_W input voxels of one input row (b, d, h, w).
    # Backward of replicate-pad forward: grad of input voxel (d,h,w) is the sum
    # of grad_output over the box of output positions that map onto it.
    #   d_out in [lo_d, lo_d+cnt_d), h_out in [lo_h, lo_h+cnt_h); along W the
    #   contribution is a contiguous run [w+pl .. w+pl+cnt_w) handled as a
    #   shifted load plus front/back corrections (pl and pr element windows).
    # Static d/h loops with scalar runtime `if` guards: interior rows execute
    # exactly one body iteration (no masked-iteration waste).
    # All shapes and pads are constexpr: boundary chains, row decode, and
    # stride arithmetic fold to constants per workload.
    pid_w = tl.program_id(0)
    row = tl.program_id(1)

    h = row % H_in
    plane = row // H_in
    d = plane % D_in
    b = plane // D_in

    w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    wmask = w < W_in
    vmask = wmask & (w + PL >= 0)
    kp = tl.arange(0, MAX_WP)
    kb = tl.arange(0, MAX_WB)

    lo_d = tl.where(
        D_in == 1,
        0,
        tl.where(d == 0, 0, tl.where(d == D_in - 1, D_in - 1 + PF, d + PF)),
    )
    cnt_d = tl.where(
        D_in == 1,
        D_out,
        tl.where(d == 0, PF + 1, tl.where(d == D_in - 1, D_out - (D_in - 1 + PF), 1)),
    )
    lo_h = tl.where(
        H_in == 1,
        0,
        tl.where(h == 0, 0, tl.where(h == H_in - 1, H_in - 1 + PT, h + PT)),
    )
    cnt_h = tl.where(
        H_in == 1,
        H_out,
        tl.where(h == 0, PT + 1, tl.where(h == H_in - 1, H_out - (H_in - 1 + PT), 1)),
    )

    in_off = (b * D_in + d) * H_in * W_in + h * W_in + w
    out_base = (b * D_out) * H_out * W_out

    acc = tl.zeros([BLOCK_W], dtype=ACC)
    for td in tl.static_range(MAX_D):
        if td < cnt_d:
            do = lo_d + td
            for th in tl.static_range(MAX_H):
                if th < cnt_h:
                    ho = lo_h + th
                    base = out_base + do * (H_out * W_out) + ho * W_out
                    acc += tl.load(
                        grad_out_ptr + base + w + PL, mask=vmask, other=0.0
                    ).to(ACC)
                    if PL >= 1:
                        f = tl.sum(
                            tl.load(
                                grad_out_ptr + base + kp, mask=kp < PL, other=0.0
                            ).to(ACC)
                        )
                        acc += tl.where(w == 0, f, 0.0)
                    if PR >= 1:
                        bsum = tl.sum(
                            tl.load(
                                grad_out_ptr + base + W_in + PL + kb,
                                mask=(kb < PR) & (W_in + PL + kb >= 0),
                                other=0.0,
                            ).to(ACC)
                        )
                        acc += tl.where(w == W_in - 1, bsum, 0.0)
    tl.store(grad_in_ptr + in_off, acc, mask=wmask)


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


def replication_pad3d_backward(grad_output, self, padding):
    # ATen replication_pad3d_backward: self (*, D, H, W), grad_output (*, D+pf+pbk, H+pt+pb, W+pl+pr)
    # padding: (pl, pr, pt, pb, pf, pbk)
    if tuple(grad_output.shape[:-3]) != tuple(self.shape[:-3]):
        raise ValueError("grad_output and self must have matching leading dimensions")
    spatial = self.shape[-3:]
    D, H, W = int(spatial[0]), int(spatial[1]), int(spatial[2])
    B = int(math.prod(self.shape[:-3])) if self.dim() > 3 else 1
    pl, pr, pt, pb, pf, pbk = (int(p) for p in padding)
    D_out = D + pf + pbk
    H_out = H + pt + pb
    W_out = W + pl + pr
    if tuple(grad_output.shape[-3:]) != (D_out, H_out, W_out):
        raise ValueError(
            f"grad_output spatial shape {tuple(grad_output.shape[-3:])} "
            f"does not match expected {(D_out, H_out, W_out)}."
        )

    out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    numel = out.numel()
    if numel == 0 or grad_output.numel() == 0:
        return out

    go = grad_output.contiguous()
    dtype = self.dtype

    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        acc = tl.float32
    elif dtype == torch.float64:
        acc = tl.float64
    else:
        raise TypeError(f"unsupported dtype {dtype}")

    maxd = D_out if D == 1 else max(pf + 1, pbk + 1, 1)
    maxh = H_out if H == 1 else max(pt + 1, pb + 1, 1)
    maxwp = max(1, 1 << (max(pl, 0) - 1).bit_length()) if max(pl, 0) > 0 else 1
    maxwb = max(1, 1 << (max(pr, 0) - 1).bit_length()) if max(pr, 0) > 0 else 1
    block_w = min(1024, max(16, _next_pow2(W)))
    num_warps = max(1, min(8, block_w // 64))
    grid = ((W + block_w - 1) // block_w, H * B * D)
    with torch.cuda.device(go.device):
        _repad3d_bwd_gather[grid](
            go,
            out,
            ACC=acc,
            D_in=D,
            H_in=H,
            W_in=W,
            D_out=D_out,
            H_out=H_out,
            W_out=W_out,
            PF=pf,
            PT=pt,
            PL=pl,
            PR=pr,
            MAX_D=maxd,
            MAX_H=maxh,
            MAX_WP=maxwp,
            MAX_WB=maxwb,
            BLOCK_W=block_w,
            num_warps=num_warps,
        )

    return out
