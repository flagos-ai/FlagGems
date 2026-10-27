# Copyright 2026, The FlagOS Contributors.
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

from flag_gems.utils import libentry

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def max_unpool2d_kernel(
    pooled_ptr,
    indices_ptr,
    output_ptr,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    c,
    pooled_h,
    pooled_w,
    out_h,
    out_w,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    spatial = pooled_h * pooled_w
    out_spatial = out_h * out_w

    while start < total:
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total
        pooled_vals = tl.load(pooled_ptr + offsets, mask=mask, other=0.0)
        indices_flat = tl.load(indices_ptr + offsets, mask=mask, other=0)

        nc = offsets // spatial
        c_idx = nc % c
        n_idx = nc // c
        h_orig = indices_flat // out_w
        w_orig = indices_flat - h_orig * out_w
        out_offsets = (
            n_idx * out_stride_n
            + c_idx * out_stride_c
            + h_orig * out_stride_h
            + w_orig * out_stride_w
        )
        out_mask = mask & (indices_flat >= 0) & (indices_flat < out_spatial)
        tl.store(output_ptr + out_offsets, pooled_vals, mask=out_mask)
        start += step


def max_unpool2d(pooled: torch.Tensor, indices: torch.Tensor, output_size: list):
    logger.debug("GEMS_CAMBRICON MAX_UNPOOL2D")

    pooled = pooled.contiguous()
    indices = indices.contiguous()

    n, c, pooled_h, pooled_w = pooled.shape
    out_h, out_w = output_size[0], output_size[1]

    output = torch.zeros((n, c, out_h, out_w), device=pooled.device, dtype=pooled.dtype)
    total = pooled.numel()
    if output.numel() == 0 or total == 0:
        return output

    block_size = 256
    grid = (min(triton.cdiv(total, block_size), TOTAL_CORE_NUM),)

    max_unpool2d_kernel[grid](
        pooled,
        indices,
        output,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        c,
        pooled_h,
        pooled_w,
        out_h,
        out_w,
        total,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )

    return output
