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
from flag_gems.utils import libentry, pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _masked_scale_pointwise(input, mask, scale):
    return tl.where(mask != 0, input * scale, 0.0)


@libentry()
@triton.jit
def _masked_scale_contiguous_kernel(
    input_ptr,
    mask_ptr,
    out_ptr,
    scale,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    STREAMING: tl.constexpr,
):
    load_cache: tl.constexpr = ".cg" if STREAMING else ""
    store_cache: tl.constexpr = ".cs" if STREAMING else ""
    pid = tl.program_id(0)
    if N > 2147483647:
        offsets = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    else:
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if N % BLOCK_SIZE == 0:
        x = tl.load(input_ptr + offsets, cache_modifier=load_cache)
        mask = tl.load(mask_ptr + offsets, cache_modifier=load_cache)
    else:
        valid = offsets < N
        x = tl.load(input_ptr + offsets, valid, other=0, cache_modifier=load_cache)
        mask = tl.load(mask_ptr + offsets, valid, other=0, cache_modifier=load_cache)
    result = tl.where(mask != 0, x * scale, 0.0)
    if N % BLOCK_SIZE == 0:
        tl.store(out_ptr + offsets, result, cache_modifier=store_cache)
    else:
        tl.store(out_ptr + offsets, result, offsets < N, cache_modifier=store_cache)


def run(input, mask, scale):
    if not input.is_floating_point():
        raise ValueError(f"Only floating-point dtype is supported, got {input.dtype}")
    if (
        input.shape != mask.shape
        or not input.is_contiguous()
        or not mask.is_contiguous()
    ):
        return _masked_scale_pointwise(input, mask, scale)
    out = torch.empty_like(input)
    count = input.numel()
    if count:
        # Use the measured streaming configuration for full-memory passes.
        streaming = count >= 100_000_000
        block, warps = 1024, 8 if streaming else 4
        with torch_device_fn.device(input.device):
            _masked_scale_contiguous_kernel[(triton.cdiv(count, block),)](
                input,
                mask,
                out,
                float(scale),
                count,
                BLOCK_SIZE=block,
                STREAMING=streaming,
                num_warps=warps,
            )
    return out


masked_scale = run
