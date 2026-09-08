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


@triton.jit
def upsample_linear1d_backward_kernel(
    grad_out_ptr,
    grad_in_ptr,
    n,
    c,
    in_w,
    out_w,
    go_stride_n,
    go_stride_c,
    go_stride_w,
    gi_stride_n,
    gi_stride_c,
    gi_stride_w,
    align_corners: tl.constexpr,
    WL: tl.constexpr,
    WR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # 2D grid: axis0 = x-block within a (n, c) row, axis1 = flattened (n, c) row.
    # Compared with the generic implementation, the per-element integer div/mod
    # on the flattened index is hoisted to scalar ops on the program id, and
    # the window is asymmetric & exact (WL/WR) instead of a fixed WINDOW, which
    # removes a large number of redundant gather loads.
    pid_x = tl.program_id(0)
    pid_row = tl.program_id(1)
    x_in = pid_x * BLOCK + tl.arange(0, BLOCK)
    mask = x_in < in_w
    c_idx = pid_row % c
    n_idx = pid_row // c

    x_in_f = x_in.to(tl.float32)
    in_w_f = tl.cast(in_w, tl.float32)
    out_w_f = tl.cast(out_w, tl.float32)

    if align_corners:
        if in_w > 1:
            s = (out_w_f - 1.0) / (in_w_f - 1.0)
            center = x_in_f * s
        else:
            center = tl.zeros([BLOCK], dtype=tl.float32)
        if out_w > 1:
            s_inv = (in_w_f - 1.0) / (out_w_f - 1.0)
        else:
            s_inv = 0.0
    else:
        s = out_w_f / in_w_f
        center = (x_in_f + 0.5) * s - 0.5
        s_inv = in_w_f / out_w_f

    base = tl.floor(center).to(tl.int32)
    base_f = base.to(tl.float32)
    if align_corners:
        xr_base = base_f * s_inv
    else:
        xr_base = (base_f + 0.5) * s_inv - 0.5

    go_base = grad_out_ptr + n_idx * go_stride_n + c_idx * go_stride_c
    gi_base = grad_in_ptr + n_idx * gi_stride_n + c_idx * gi_stride_c

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(-WL, WR + 1):
        x_out = base + i
        valid = (x_out >= 0) & (x_out < out_w)
        # x_real is affine in i; the per-iteration division is hoisted to xr_base.
        x_real = xr_base + i * s_inv
        x0_f = tl.floor(x_real)
        w1 = x_real - x0_f
        w0 = 1.0 - w1
        x0_i = tl.maximum(x0_f, 0.0).to(tl.int32)
        x1_i = tl.minimum(x0_f + 1.0, in_w_f - 1.0).to(tl.int32)
        g = tl.load(
            go_base + x_out * go_stride_w,
            mask=mask & valid,
            other=0.0,
        ).to(tl.float32)
        same = x0_i == x1_i
        is_x0 = x_in == x0_i
        is_x1 = x_in == x1_i
        acc += tl.where(same & is_x0, g * (w0 + w1), 0.0)
        acc += tl.where((~same) & is_x0, g * w0, 0.0)
        acc += tl.where((~same) & is_x1, g * w1, 0.0)
    tl.store(gi_base + x_in * gi_stride_w, acc, mask=mask)


def _compute_window(in_w, out_w, align_corners):
    if align_corners and in_w > 1 and out_w > 1:
        num, den = out_w - 1, in_w - 1
    else:
        num, den = out_w, in_w
    if num >= den:
        w = triton.cdiv(num, den)
        return max(1, w - 1), w
    return 1, 1


def upsample_linear1d_backward(
    grad_output: torch.Tensor,
    output_size,
    input_size,
    align_corners: bool,
    scale_factors=None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE_LINEAR1D_BACKWARD (mthreads)")

    if len(input_size) == 3:
        n, c, in_w = input_size
    elif len(input_size) == 2:
        n, c, in_w = input_size[0], 1, input_size[1]
    elif len(input_size) == 1:
        n, c, in_w = 1, 1, input_size[0]
    else:
        raise ValueError

    if output_size is not None:
        out_w = output_size[0]
    else:
        assert scale_factors is not None
        out_w = int(in_w * scale_factors[0])

    assert grad_output.shape[-1] == out_w

    grad_out_3d = grad_output.contiguous().view(n, c, out_w)

    grad_in = torch.zeros(
        (n, c, in_w),
        device=grad_output.device,
        dtype=grad_output.dtype,
    )

    go_stride_n, go_stride_c, go_stride_w = grad_out_3d.stride()
    gi_stride_n, gi_stride_c, gi_stride_w = grad_in.stride()

    WL, WR = _compute_window(in_w, out_w, align_corners)
    BLOCK = 256
    num_warps = 4
    grid = (triton.cdiv(in_w, BLOCK), n * c)

    upsample_linear1d_backward_kernel[grid](
        grad_out_3d,
        grad_in,
        n,
        c,
        in_w,
        out_w,
        go_stride_n,
        go_stride_c,
        go_stride_w,
        gi_stride_n,
        gi_stride_c,
        gi_stride_w,
        align_corners,
        WL=WL,
        WR=WR,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )

    return grad_in
