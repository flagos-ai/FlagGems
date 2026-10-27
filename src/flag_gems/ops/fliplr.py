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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def fliplr_contiguous_kernel(
    input,
    output,
    n_elements,
    width,
    inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    offsets = offsets.to(tl.int64)
    column = (offsets // inner_size) % width
    input_offsets = offsets + (width - 1 - 2 * column) * inner_size
    values = tl.load(input + input_offsets, mask=mask)
    tl.store(output + offsets, values, mask=mask)


@libentry()
@triton.jit
def fliplr_strided_kernel(
    input,
    output,
    n_elements,
    SHAPE: tl.constexpr,
    INPUT_STRIDES: tl.constexpr,
    OUTPUT_STRIDES: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    linear = offsets.to(tl.int64)
    input_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    output_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

    for dim in tl.static_range(NDIM - 1, -1, -1):
        index = linear % SHAPE[dim]
        linear //= SHAPE[dim]
        input_index = SHAPE[dim] - 1 - index if dim == 1 else index
        input_offsets += input_index * INPUT_STRIDES[dim]
        output_offsets += index * OUTPUT_STRIDES[dim]

    values = tl.load(input + input_offsets, mask=mask)
    tl.store(output + output_offsets, values, mask=mask)


def fliplr(self: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS FLIPLR")

    if self.ndim < 2:
        raise RuntimeError("Input must be >= 2-d.")

    output = torch.empty_like(self)
    n_elements = self.numel()
    if n_elements == 0:
        return output

    grid = (triton.cdiv(n_elements, 1024),)
    with torch_device_fn.device(self.device):
        if self.is_contiguous() and output.is_contiguous():
            inner_size = math.prod(self.shape[2:])
            fliplr_contiguous_kernel[grid](
                self,
                output,
                n_elements,
                self.shape[1],
                inner_size,
                BLOCK_SIZE=1024,
            )
        else:
            fliplr_strided_kernel[grid](
                self,
                output,
                n_elements,
                SHAPE=tuple(self.shape),
                INPUT_STRIDES=tuple(self.stride()),
                OUTPUT_STRIDES=tuple(output.stride()),
                NDIM=self.ndim,
                BLOCK_SIZE=1024,
            )

    return output
