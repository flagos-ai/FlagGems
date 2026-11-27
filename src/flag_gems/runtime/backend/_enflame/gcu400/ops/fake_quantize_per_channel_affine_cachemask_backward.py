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

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


# This operator is just a conditional elementwise copy; the measured ceiling is a pure copy kernel
# (about 26.8us for 1M fp16), so there is only one tuning goal: keep the grid in the bandwidth-optimal range
# (8192 elements per program, 2 warps, about 8~128 programs), and avoid extra work beyond the implicit
# bool→float conversion.
_BLOCK_MAX = 8192


def _pick_block(n_elements):
    block = 1024
    while block * 2 <= n_elements // 16 and block * 2 <= _BLOCK_MAX:
        block *= 2
    return min(block, _BLOCK_MAX)


@triton.jit
def fake_quantize_per_channel_affine_cachemask_backward_kernel(
    grad_ptr,
    mask_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    bound = offsets < n_elements
    grad = tl.load(grad_ptr + offsets, mask=bound)
    quant_mask = tl.load(mask_ptr + offsets, mask=bound)
    out = grad * quant_mask
    tl.store(output_ptr + offsets, out, mask=bound)


def fake_quantize_per_channel_affine_cachemask_backward(grad, mask):
    grad = grad.contiguous()
    mask = mask.contiguous()
    output = torch.empty_like(grad)
    n_elements = grad.numel()
    if n_elements == 0:
        return output
    block = _pick_block(n_elements)
    grid = (triton.cdiv(n_elements, block),)
    with torch_device_fn.device(grad.device):
        fake_quantize_per_channel_affine_cachemask_backward_kernel[grid](
            grad, mask, output, n_elements, BLOCK_SIZE=block, num_warps=2
        )
    return output
