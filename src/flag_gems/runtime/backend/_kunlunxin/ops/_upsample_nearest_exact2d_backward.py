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
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def upsample_nearest_exact2d_backward_kernel(
    grad_output,
    grad_input,
    total,
    in_h,
    in_w,
    out_h,
    out_w,
    scale_h: tl.constexpr,
    scale_w: tl.constexpr,
    max_h: tl.constexpr,
    max_w: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    output_mask = offsets < total

    w_idx = offsets % in_w
    h_idx = (offsets // in_w) % in_h
    nc_idx = offsets // (in_h * in_w)

    h_start = tl.ceil(h_idx.to(tl.float32) * scale_h - 0.5).to(tl.int32)
    h_end = tl.ceil((h_idx.to(tl.float32) + 1.0) * scale_h - 0.5).to(tl.int32)
    w_start = tl.ceil(w_idx.to(tl.float32) * scale_w - 0.5).to(tl.int32)
    w_end = tl.ceil((w_idx.to(tl.float32) + 1.0) * scale_w - 0.5).to(tl.int32)

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for dh in tl.static_range(max_h):
        for dw in tl.static_range(max_w):
            oh = h_start + dh
            ow = w_start + dw
            valid = (
                output_mask
                & (oh >= 0)
                & (oh < h_end)
                & (oh < out_h)
                & (ow >= 0)
                & (ow < w_end)
                & (ow < out_w)
            )
            oh_safe = tl.where(valid, oh, 0)
            ow_safe = tl.where(valid, ow, 0)
            grad_offset = nc_idx * (out_h * out_w) + oh_safe * out_w + ow_safe
            value = tl.load(grad_output + grad_offset, mask=valid, other=0.0)
            acc += tl.where(valid, value.to(tl.float32), 0.0)

    tl.store(grad_input + offsets, acc, mask=output_mask)


def _upsample_nearest_exact2d_backward(
    grad_output,
    output_size,
    input_size,
    scales_h=None,
    scales_w=None,
):
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE NEAREST EXACT2D BACKWARD")

    in_h, in_w = input_size[-2:]
    out_h, out_w = output_size
    scale_h = float(scales_h) if scales_h is not None else out_h / in_h
    scale_w = float(scales_w) if scales_w is not None else out_w / in_w

    grad_input = torch.empty(input_size, device=grad_output.device, dtype=grad_output.dtype)
    total = grad_input.numel()
    block = 1024
    grid = (triton.cdiv(total, block),)
    upsample_nearest_exact2d_backward_kernel[grid](
        grad_output,
        grad_input,
        total,
        in_h,
        in_w,
        out_h,
        out_w,
        scale_h=scale_h,
        scale_w=scale_w,
        max_h=math.ceil(scale_h) + 1,
        max_w=math.ceil(scale_w) + 1,
        BLOCK=block,
        num_warps=1,
        buffer_size_limit=2048,
        isCloseVectorization=True,
    )
    return grad_input
