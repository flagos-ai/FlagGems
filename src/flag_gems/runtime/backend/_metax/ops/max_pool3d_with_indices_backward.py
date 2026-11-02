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

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _triple(v):
    if isinstance(v, (tuple, list)):
        assert len(v) == 3
        return int(v[0]), int(v[1]), int(v[2])
    return int(v), int(v), int(v)


@triton.jit
def _maxpool3d_bwd_scatter(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    plane_size_out,
    plane_size_in,
    BLOCK: tl.constexpr,
):
    pid_block = tl.program_id(0)
    pid_plane = tl.program_id(1)
    offs = pid_block * BLOCK + tl.arange(0, BLOCK)
    base_out = pid_plane * plane_size_out
    mask = offs < plane_size_out
    g = tl.load(grad_output_ptr + base_out + offs, mask=mask, other=0.0)
    idx = tl.load(indices_ptr + base_out + offs, mask=mask, other=-1)
    valid = mask & (idx >= 0)
    dest = pid_plane * plane_size_in + idx
    tl.atomic_add(grad_input_ptr + dest, g, mask=valid, sem="relaxed")


@triton.jit
def _cast_f32_to_lowp(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    v = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, v.to(dst_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _maxpool3d_bwd_gather(
    grad_output_ptr,
    indices_ptr,
    out_ptr,
    plane_size_out,
    plane_size_in,
    pD,
    pH,
    pW,
    sD,
    sH,
    sW,
    dilD,
    dilH,
    dilW,
    kDm1_dilD,
    kHm1_dilH,
    kWm1_dilW,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    D_out: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    DIL_ANY: tl.constexpr,
    Q_D: tl.constexpr,
    Q_H: tl.constexpr,
    Q_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid0 = tl.program_id(0)
    d = tl.program_id(1)
    plane = tl.program_id(2)

    pos = pid0 * BLOCK + tl.arange(0, BLOCK)
    w = pos % W_in
    h = pos // W_in
    mask_pos = pos < H_in * W_in

    f = d * (H_in * W_in) + pos
    f64 = f.to(tl.int64)

    # candidate output-window ranges that cover this input element
    od_min = -((-(d + pD - kDm1_dilD)) // sD)
    od_max = (d + pD) // sD
    od_min = tl.maximum(od_min, 0)
    od_max = tl.minimum(od_max, D_out - 1)

    oh_min = -((-(h + pH - kHm1_dilH)) // sH)
    oh_max = (h + pH) // sH
    oh_min = tl.maximum(oh_min, 0)
    oh_max = tl.minimum(oh_max, H_out - 1)

    ow_min = -((-(w + pW - kWm1_dilW)) // sW)
    ow_max = (w + pW) // sW
    ow_min = tl.maximum(ow_min, 0)
    ow_max = tl.minimum(ow_max, W_out - 1)

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    base = plane * plane_size_out
    base_in = plane * plane_size_in
    HW_out = H_out * W_out
    clamp_hi = base + plane_size_out - 1

    addr0 = base + od_min * HW_out + oh_min * W_out + ow_min

    n_od = tl.maximum(od_max - od_min + 1, 0)
    for qd in range(0, n_od):
        od = od_min + qd
        dbase = addr0 + qd * HW_out
        for qh in tl.static_range(Q_H):
            oh = oh_min + qh
            row = dbase + qh * W_out
            for qw in tl.static_range(Q_W):
                ow = ow_min + qw
                valid = mask_pos & (oh <= oh_max) & (ow <= ow_max)
                if DIL_ANY:
                    valid &= (d + pD - od * sD) % dilD == 0
                    valid &= (h + pH - oh * sH) % dilH == 0
                    valid &= (w + pW - ow * sW) % dilW == 0
                addr = tl.minimum(tl.maximum(row + qw, base), clamp_hi)
                idx = tl.load(indices_ptr + addr)
                g = tl.load(grad_output_ptr + addr)
                acc += tl.where(valid & (idx == f64), g, 0.0)

    tl.store(out_ptr + (base_in + f).to(tl.int64), acc, mask=mask_pos)


def max_pool3d_with_indices_backward(
    grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices
):
    k = _triple(kernel_size)
    s = _triple(stride)
    p = _triple(padding)
    dil = _triple(dilation)

    N, C, D_in, H_in, W_in = self.shape
    D_out, H_out, W_out = grad_output.shape[2:]
    n_planes = N * C
    plane_size_in = D_in * H_in * W_in
    plane_size_out = D_out * H_out * W_out

    if plane_size_out == 0:
        return torch.zeros_like(self)

    # Large-plane class: the per-output-index scatter reads each grad/index
    # exactly once and is far cheaper than the candidate gather, which scans
    # every covering window for every input element.
    use_scatter = (plane_size_out >= 2048) and (plane_size_in >= 8192)

    if use_scatter:
        BLOCK = 512
        grid = (triton.cdiv(plane_size_out, BLOCK), n_planes)
        if grad_output.dtype == torch.float32:
            grad_input = torch.zeros_like(self)
            _maxpool3d_bwd_scatter[grid](
                grad_output,
                indices,
                grad_input,
                plane_size_out,
                plane_size_in,
                BLOCK=BLOCK,
                num_warps=4,
            )
            return grad_input
        else:
            acc = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
            _maxpool3d_bwd_scatter[grid](
                grad_output,
                indices,
                acc,
                plane_size_out,
                plane_size_in,
                BLOCK=BLOCK,
                num_warps=4,
            )
            grad_input = torch.empty_like(self)
            _cast_f32_to_lowp[(triton.cdiv(self.numel(), BLOCK),)](
                acc,
                grad_input,
                self.numel(),
                BLOCK=BLOCK,
                num_warps=4,
            )
            return grad_input

    # Gather path (round-6 winner).
    out = torch.empty_like(self)

    Q_D = (dil[0] * (k[0] - 1) + s[0] - 1) // s[0] + 1
    Q_H = (dil[1] * (k[1] - 1) + s[1] - 1) // s[1] + 1
    Q_W = (dil[2] * (k[2] - 1) + s[2] - 1) // s[2] + 1

    BLOCK = 64
    grid = (triton.cdiv(H_in * W_in, BLOCK), D_in, n_planes)
    _maxpool3d_bwd_gather[grid](
        grad_output,
        indices,
        out,
        plane_size_out,
        plane_size_in,
        p[0],
        p[1],
        p[2],
        s[0],
        s[1],
        s[2],
        dil[0],
        dil[1],
        dil[2],
        dil[0] * (k[0] - 1),
        dil[1] * (k[1] - 1),
        dil[2] * (k[2] - 1),
        H_in=H_in,
        W_in=W_in,
        D_out=D_out,
        H_out=H_out,
        W_out=W_out,
        DIL_ANY=(dil[0] > 1 or dil[1] > 1 or dil[2] > 1),
        Q_D=Q_D,
        Q_H=Q_H,
        Q_W=Q_W,
        BLOCK=BLOCK,
        num_warps=1,
    )
    return out
