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


def _configs():
    return [
        triton.Config({"BLOCK_X": block_x, "BLOCK_Y": block_y}, num_warps=warps)
        for block_x in (64, 128, 256)
        for block_y in (1, 2)
        for warps in (4, 8)
    ]


@triton.autotune(configs=_configs(), key=["OH", "OW"])
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

    source_y = tl.where(
        align_corners,
        oh * reciprocal_scale_h,
        (oh + 0.5) * reciprocal_scale_h - 0.5,
    )
    source_x = tl.where(
        align_corners,
        ow * reciprocal_scale_w,
        (ow + 0.5) * reciprocal_scale_w - 0.5,
    )
    y0 = tl.maximum(source_y, 0).to(tl.int32)
    x0 = tl.maximum(source_x, 0).to(tl.int32)
    y1 = tl.minimum(y0 + 1, IH - 1)
    x1 = tl.minimum(x0 + 1, IW - 1)
    wy = source_y - y0
    wx = source_x - x0
    wy = tl.maximum(tl.minimum(wy, 1.0), 0.0)
    wx = tl.maximum(tl.minimum(wx, 1.0), 0.0)

    row0 = tl.load(input_ptr + input_base + y0[:, None] * IW + x0[None, :])
    row1 = tl.load(input_ptr + input_base + y0[:, None] * IW + x1[None, :])
    row2 = tl.load(input_ptr + input_base + y1[:, None] * IW + x0[None, :])
    row3 = tl.load(input_ptr + input_base + y1[:, None] * IW + x1[None, :])
    top = row0 * (1.0 - wx[None, :]) + row1 * wx[None, :]
    bottom = row2 * (1.0 - wx[None, :]) + row3 * wx[None, :]
    value = top * (1.0 - wy[:, None]) + bottom * wy[:, None]
    tl.store(output_ptr + output_base + oh[:, None] * OW + ow[None, :], value, mask=mask)


def _upsample_bilinear2d_aa(input, output_size, align_corners, scales_h=None, scales_w=None):
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
    grid = lambda meta: (
        triton.cdiv(ow, meta["BLOCK_X"]),
        triton.cdiv(oh, meta["BLOCK_Y"]),
        n * c,
    )
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
        )
    return output
