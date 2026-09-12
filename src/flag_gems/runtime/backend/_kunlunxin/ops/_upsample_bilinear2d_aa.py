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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext

# XPU perf repair 2026-08-16: @triton.autotune over 12 configs recompiled per
# (OH, OW) key on the XPU backend and picked suboptimal tiles; fixed bounded
# dispatch (BLOCK_X=256 for OW<=512 else 512, BLOCK_Y=2, num_warps=4) measured
# strictly better than every autotuned baseline case on the official benchmark
# matrix (see harness/solution/performance/upsample_bilinear2d_aa_xpu2_20260816.md).


@triton.jit
def _upsample_bilinear2d_aa_kernel(
    output_ptr,
    input_ptr,
    N,
    C,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    align_corners,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    pid_x = ext.program_id(0)
    pid_y = ext.program_id(1)
    pid_nc = ext.program_id(2)

    ow = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    oh = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    mask = (ow[None, :] < OW) & (oh[:, None] < OH)

    n = pid_nc // C
    c = pid_nc % C
    output_base = (n * C + c) * OH * OW
    input_base = (n * C + c) * IH * IW

    center_y = (oh + 0.5) * reciprocal_scale_h
    center_x = (ow + 0.5) * reciprocal_scale_w
    y0 = tl.maximum(center_y - 0.5, 0).to(tl.int32)
    x0 = tl.maximum(center_x - 0.5, 0).to(tl.int32)
    y1 = tl.minimum(y0 + 1, IH - 1)
    x1 = tl.minimum(x0 + 1, IW - 1)

    wy0 = tl.maximum(1.0 - tl.abs(y0 - center_y + 0.5), 0.0)
    wy1 = tl.maximum(1.0 - tl.abs(y1 - center_y + 0.5), 0.0)
    wx0 = tl.maximum(1.0 - tl.abs(x0 - center_x + 0.5), 0.0)
    wx1 = tl.maximum(1.0 - tl.abs(x1 - center_x + 0.5), 0.0)
    wy_total = wy0 + wy1
    wx_total = wx0 + wx1
    wy0 /= wy_total
    wy1 /= wy_total
    wx0 /= wx_total
    wx1 /= wx_total

    row0 = tl.load(input_ptr + input_base + y0[:, None] * IW + x0[None, :])
    row1 = tl.load(input_ptr + input_base + y0[:, None] * IW + x1[None, :])
    row2 = tl.load(input_ptr + input_base + y1[:, None] * IW + x0[None, :])
    row3 = tl.load(input_ptr + input_base + y1[:, None] * IW + x1[None, :])
    value = (
        row0 * wy0[:, None] * wx0[None, :]
        + row1 * wy0[:, None] * wx1[None, :]
        + row2 * wy1[:, None] * wx0[None, :]
        + row3 * wy1[:, None] * wx1[None, :]
    )
    tl.store(
        output_ptr + output_base + oh[:, None] * OW + ow[None, :], value, mask=mask
    )


def _upsample_bilinear2d_aa(
    input, output_size, align_corners, scales_h=None, scales_w=None
):
    assert input.ndim == 4, "The ndim of input must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"

    n, c, ih, iw = input.shape
    oh, ow = output_size
    output = torch.empty((n, c, oh, ow), device=input.device, dtype=input.dtype)
    if align_corners:
        reciprocal_scale_h = (ih - 1) / (oh - 1) if oh > 1 else 0.0
        reciprocal_scale_w = (iw - 1) / (ow - 1) if ow > 1 else 0.0
    else:
        reciprocal_scale_h = ih / oh if scales_h is None else 1.0 / scales_h
        reciprocal_scale_w = iw / ow if scales_w is None else 1.0 / scales_w
    # Fixed bounded dispatch on XPU: measured per-width best of
    # (BLOCK_X, BLOCK_Y, warps) on the official benchmark matrix
    # (BLOCK_X=256/512, BLOCK_Y=2, num_warps=4). No @triton.autotune.
    block_x = 256 if ow <= 512 else 512
    grid = (triton.cdiv(ow, block_x), triton.cdiv(oh, 2), n * c)
    with torch_device_fn.device(input.device):
        _upsample_bilinear2d_aa_kernel[grid](
            output,
            input,
            n,
            c,
            oh,
            ow,
            ih,
            iw,
            reciprocal_scale_h,
            reciprocal_scale_w,
            align_corners,
            BLOCK_X=block_x,
            BLOCK_Y=2,
            num_warps=4,
        )
    return output
