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

logger = logging.getLogger(__name__)


@triton.jit
def _select_backward_kernel(
    grad,
    output,
    output_numel,
    inner_size: tl.constexpr,
    dim_size: tl.constexpr,
    index: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_numel
    inner_offsets = offsets % inner_size
    axis_rows = offsets // inner_size
    axis_offsets = axis_rows % dim_size
    outer_offsets = axis_rows // dim_size
    grad_offsets = outer_offsets * inner_size + inner_offsets
    values = tl.load(grad + grad_offsets, mask=mask, other=0.0)
    result = tl.where(axis_offsets == index, values, 0.0)
    tl.store(output + offsets, result, mask=mask)


def select_backward(grad, input_sizes, dim, index, out=None):
    logger.debug("GEMS_KUNLUNXIN SELECT_BACKWARD")
    sizes = list(input_sizes)
    ndim = len(sizes)
    dim = int(dim)
    index = int(index)

    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError("invalid dim")

    dim_size = sizes[dim]
    if index < 0:
        index += dim_size
    if index < 0 or index >= dim_size:
        raise ValueError("index out of range")

    if out is None:
        out = torch.empty(sizes, dtype=grad.dtype, device=grad.device)
    else:
        if tuple(out.shape) != tuple(sizes):
            raise ValueError("out shape mismatch")
        if out.dtype != grad.dtype:
            raise ValueError("dtype mismatch")
        if out.device != grad.device:
            raise ValueError("device mismatch")

    output_numel = math.prod(sizes)
    if output_numel == 0:
        return out

    inner_size = math.prod(sizes[dim + 1 :]) if dim < ndim - 1 else 1
    block_size = 1024
    _select_backward_kernel[(triton.cdiv(output_numel, block_size),)](
        grad.contiguous(),
        out,
        output_numel,
        inner_size,
        dim_size,
        index,
        BLOCK_SIZE=block_size,
    )
    return out
