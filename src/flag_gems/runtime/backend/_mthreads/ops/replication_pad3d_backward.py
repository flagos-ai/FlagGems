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

"""replication_pad3d_backward as a single (plane, d)-tiled gather kernel.

Backward of the replication (clamp) pad on the trailing three axes:

    grad_input[n,c,d,h,w] =
        sum_{od in [lo_d,hi_d]} sum_{oh in [lo_h,hi_h]} sum_{ow in [lo_w,hi_w]}
            grad_output[n,c,od,oh,ow]

with, for one axis of input size X, output size Xo = X + L + R and input index i
(padding order for 3-D is [l, r, t, b, f, bk] over (W, H, D)):

    lo(i) = (i == 0)     ? 0        : i + L
    hi(i) = (i == X - 1) ? Xo - 1   : i + L
    lo <- max(lo, 0), hi <- min(hi, Xo - 1); an empty range contributes 0.

Grid: (plane, d, h-tile).  Each program owns one (n, c, d) slice and an
[HT, BW] tile of its (H, W) plane, sized so each thread still handles ~8
contiguous W elements (16-byte vector accesses) on the larger shapes.

Because the program's d is a scalar, the D-axis preimage loop has a runtime
*scalar* trip count (1 for an interior d, up to pf+1 / pbk+1 at a face), so the
predicated load/accumulate slot count drops from maxd*maxh*maxw (up to 36 for
the tested paddings) to cnt_d*maxh*maxw (~9) while keeping the coalesced,
vectorized access pattern of the flat kernel.  Only H and W keep per-lane masks.

Every grad_output element is read exactly once over the whole grid and every
grad_input element is written exactly once (an empty preimage stores an exact
0), so there is no zero-init pass and no atomic traffic.

Host wrapper: shape-contract validation, one torch.empty_like (preserves the
layout of `self`, which selects a flat or strided store) and one kernel launch.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.replication_pad3d_backward import (
    replication_pad3d_backward as default_replication_pad3d_backward,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _repl_pad3d_bwd_kernel(
    g_ptr,
    out_ptr,
    C,
    sN,
    sC,
    D,
    H,
    W,
    Do,
    Ho,
    Wo,
    pl,
    pt,
    pf,
    sD,
    sH,
    sW,
    oN,
    oC,
    oD,
    oH,
    oW,
    IS_5D: tl.constexpr,
    OUT_FLAT: tl.constexpr,
    ACC64: tl.constexpr,
    IDX64: tl.constexpr,
    HT: tl.constexpr,
    BW: tl.constexpr,
    MAXH: tl.constexpr,
    MAXW: tl.constexpr,
    UNROLL: tl.constexpr,
):
    p = tl.program_id(0)
    d = tl.program_id(1)
    ht = tl.program_id(2)

    hh = ht * HT + tl.arange(0, HT)
    ww = tl.arange(0, BW)
    m_h = hh < H
    m_w = ww < W

    if IS_5D:
        n = p // C
        c = p - n * C
        gbase = n * sN + c * sC
        obase = n * oN + c * oC
    else:
        gbase = p * sC
        obase = p * oC

    # D-axis preimage: scalar for this program
    lo_d = tl.where(d == 0, 0, d + pf)
    hi_d = tl.where(d == D - 1, Do - 1, d + pf)
    hi_d = tl.minimum(hi_d, Do - 1)
    lo_d = tl.maximum(lo_d, 0)
    cnt_d = tl.maximum(hi_d - lo_d + 1, 0)

    # H-axis preimage (per row of the tile)
    lo_h = tl.where(hh == 0, 0, hh + pt)
    hi_h = tl.where(hh == H - 1, Ho - 1, hh + pt)
    hi_h = tl.minimum(hi_h, Ho - 1)
    lo_h = tl.maximum(lo_h, 0)
    cnt_h = hi_h - lo_h + 1

    # W-axis preimage (per lane)
    lo_w = tl.where(ww == 0, 0, ww + pl)
    hi_w = tl.where(ww == W - 1, Wo - 1, ww + pl)
    hi_w = tl.minimum(hi_w, Wo - 1)
    lo_w = tl.maximum(lo_w, 0)
    cnt_w = hi_w - lo_w + 1

    if ACC64:
        acc = tl.zeros((HT, BW), dtype=tl.float64)
    else:
        acc = tl.zeros((HT, BW), dtype=tl.float32)

    for kd in range(cnt_d):
        od = lo_d + kd
        g_d = gbase + od * sD
        if UNROLL:
            # Constant H/W bounds are unrolled so the ~MAXH*MAXW predicated
            # loads of a tile are independent instructions instead of a
            # serialized dynamically-bounded loop.
            for kh in tl.static_range(MAXH):
                mh = (kh < cnt_h) & m_h
                g_h = g_d + (lo_h + kh) * sH
                for kw in tl.static_range(MAXW):
                    m = mh[:, None] & ((kw < cnt_w) & m_w)[None, :]
                    acc += tl.load(
                        g_ptr + g_h[:, None] + (lo_w + kw)[None, :] * sW,
                        mask=m,
                        other=0.0,
                    )
        else:
            for kh in range(MAXH):
                mh = (kh < cnt_h) & m_h
                g_h = g_d + (lo_h + kh) * sH
                for kw in range(MAXW):
                    m = mh[:, None] & ((kw < cnt_w) & m_w)[None, :]
                    acc += tl.load(
                        g_ptr + g_h[:, None] + (lo_w + kw)[None, :] * sW,
                        mask=m,
                        other=0.0,
                    )

    m_out = m_h[:, None] & m_w[None, :]
    if OUT_FLAT:
        if IDX64:
            base = (p.to(tl.int64) * D + d) * H * W
            off = base + hh[:, None] * W + ww[None, :]
        else:
            off = (p * D + d) * (H * W) + hh[:, None] * W + ww[None, :]
        tl.store(out_ptr + off, acc, mask=m_out)
    else:
        tl.store(
            out_ptr + obase + d * oD + hh[:, None] * oH + ww[None, :] * oW,
            acc,
            mask=m_out,
        )


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def replication_pad3d_backward(grad_output, self, padding):
    logger.debug("GEMS_MTHREADS REPLICATION_PAD3D_BACKWARD")

    if not _use_triton_kernel(self) or not _use_triton_kernel(grad_output):
        return default_replication_pad3d_backward(grad_output, self, padding)

    with torch_device_fn.device(self.device):
        return _launch_replication_pad3d_backward(grad_output, self, padding)


def _launch_replication_pad3d_backward(grad_output, self, padding):
    pad = [int(v) for v in padding]
    if len(pad) != 6:
        raise ValueError(
            "replication_pad3d_backward expects 6 padding values "
            "(l, r, t, b, front, back), got " + str(len(pad))
        )
    pl, pr, pt, pb, pf, pbk = pad

    dim = self.dim()
    if dim == 5:
        N, C, D, H, W = self.shape
        is5 = True
        planes = N * C
    elif dim == 4:
        C, D, H, W = self.shape
        is5 = False
        planes = C
    else:
        raise ValueError(
            "replication_pad3d_backward expects 4-D or 5-D input, got "
            + str(dim)
            + "-D"
        )

    Do = D + pf + pbk
    Ho = H + pt + pb
    Wo = W + pl + pr

    # Shape contract: grad_output must share the leading dims of self and be the
    # padded version of its trailing three dims.
    if grad_output.dim() != dim:
        raise ValueError(
            "replication_pad3d_backward: grad_output rank "
            + str(grad_output.dim())
            + " does not match self rank "
            + str(dim)
        )
    if tuple(grad_output.shape[:-3]) != tuple(self.shape[:-3]):
        raise ValueError(
            "replication_pad3d_backward: mismatch in leading dimensions, "
            "grad_output "
            + str(tuple(grad_output.shape[:-3]))
            + " vs self "
            + str(tuple(self.shape[:-3]))
        )
    if tuple(grad_output.shape[-3:]) != (Do, Ho, Wo):
        raise ValueError(
            "replication_pad3d_backward: grad_output spatial shape "
            + str(tuple(grad_output.shape[-3:]))
            + " does not match padded "
            "self shape " + str((Do, Ho, Wo))
        )

    out = torch.empty_like(self)
    total = out.numel()
    if total == 0:
        return out

    gs = grad_output.stride()
    if is5:
        sN, sC, sD, sH, sW = gs
    else:
        sN, sC, sD, sH, sW = 0, gs[0], gs[1], gs[2], gs[3]

    os_ = out.stride()
    if is5:
        oN, oC, oD, oH, oW = os_
    else:
        oN, oC, oD, oH, oW = 0, os_[0], os_[1], os_[2], os_[3]

    maxh = max(1, pt + 1, pb + 1, Ho if H == 1 else 1)
    maxw = max(1, pl + 1, pr + 1, Wo if W == 1 else 1)

    # Tile choice: grow the H tile while the grid keeps >= 480 blocks and the
    # per-program tile stays small, so each thread still gets ~8 contiguous W
    # elements (16-byte vector accesses) on the larger shapes and the small
    # shapes keep enough blocks to fill the device.
    bw = min(2048, max(16, triton.next_power_of_2(W)))
    wtiles = triton.cdiv(W, bw)
    npt = planes * D
    ht = 1
    while ht * 2 <= H and (ht * 2) * bw <= 2048:
        nxt = ht * 2
        if npt * triton.cdiv(H, nxt) * wtiles < 480:
            break
        ht = nxt
    e = ht * bw
    nw = min(256, max(32, e // 4)) // 32

    grid = (planes, D, triton.cdiv(H, ht) * wtiles)

    _repl_pad3d_bwd_kernel[grid](
        grad_output,
        out,
        C,
        sN,
        sC,
        D,
        H,
        W,
        Do,
        Ho,
        Wo,
        pl,
        pt,
        pf,
        sD,
        sH,
        sW,
        oN,
        oC,
        oD,
        oH,
        oW,
        IS_5D=is5,
        OUT_FLAT=out.is_contiguous(),
        ACC64=grad_output.dtype == torch.float64,
        IDX64=total >= 2147483647,
        HT=ht,
        BW=bw,
        MAXH=maxh,
        MAXW=maxw,
        UNROLL=(maxh * maxw <= 36),
        num_warps=nw,
    )
    return out
