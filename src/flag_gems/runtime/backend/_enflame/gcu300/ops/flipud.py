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

"""GCU300 specialization of ``flipud``.

Both generic kernels widen the flat index and every offset derived from it to
int64.  This target miscompiles lane-varying int64 arithmetic once a block
covers 256 or more lanes, so half of the copied elements come from the wrong
address.  The kernels here keep the flat index and the computed offsets in
int32; the contiguous path folds the row shift into the pointer expression
with scalar arithmetic instead of widening the vector.
"""

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
def flipud_contiguous_kernel(
    input,
    output,
    n_elements,
    height,
    row_size,
    BLOCK_SIZE: tl.constexpr,
):
    # int32 flat index; the row shift is a scalar expression on the same word
    # width, so nothing here is lane-varying int64.
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    row = offsets // row_size
    input_offsets = offsets + (height - 1 - 2 * row) * row_size
    values = tl.load(input + input_offsets, mask=mask)
    tl.store(output + offsets, values, mask=mask)


@libentry()
@triton.jit
def flipud_strided_kernel(
    input,
    output,
    n_elements,
    SHAPE: tl.constexpr,
    INPUT_STRIDES: tl.constexpr,
    OUTPUT_STRIDES: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = linear < n_elements
    input_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    output_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    for dim in tl.static_range(NDIM - 1, -1, -1):
        index = linear % SHAPE[dim]
        linear = linear // SHAPE[dim]
        input_index = SHAPE[dim] - 1 - index if dim == 0 else index
        input_offsets += input_index * INPUT_STRIDES[dim]
        output_offsets += index * OUTPUT_STRIDES[dim]

    values = tl.load(input + input_offsets, mask=mask)
    tl.store(output + output_offsets, values, mask=mask)


def _flipud_impl(self: torch.Tensor) -> torch.Tensor:
    if self.ndim < 1:
        raise RuntimeError("Input must be >= 1-d.")

    if self.is_complex():
        return torch.view_as_complex(_flipud_impl(torch.view_as_real(self)))

    output = torch.empty_like(self)
    n_elements = self.numel()
    if n_elements == 0:
        return output

    grid = (triton.cdiv(n_elements, 1024),)
    with torch_device_fn.device(self.device):
        if self.is_contiguous() and output.is_contiguous():
            row_size = math.prod(self.shape[1:])
            flipud_contiguous_kernel[grid](
                self,
                output,
                n_elements,
                self.shape[0],
                row_size,
                BLOCK_SIZE=1024,
            )
        else:
            flipud_strided_kernel[grid](
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


class Flipud(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return _flipud_impl(input)

    @staticmethod
    def backward(ctx, grad_output):
        return _flipud_impl(grad_output)


def flipud(self: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME FLIPUD")

    if torch.is_grad_enabled() and self.requires_grad:
        return Flipud.apply(self)
    return _flipud_impl(self)
