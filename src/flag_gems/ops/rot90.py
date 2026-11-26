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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.autotune(configs=runtime.get_tuned_config("rot90"), key=["n_elements"])
@triton.jit
def rot90_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    BASE_OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    remaining = offsets
    source = tl.full((BLOCK_SIZE,), BASE_OFFSET, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        source += (remaining % SHAPE[axis]) * STRIDES[axis]
        remaining //= SHAPE[axis]
    mask = offsets < n_elements
    values = tl.load(in_ptr + source, mask=mask)
    tl.store(out_ptr + offsets, values, mask=mask)


def rot90(input, k=1, dims=(0, 1)):
    logger.debug("GEMS ROT90")
    rank = input.ndim
    if len(dims) != 2:
        raise RuntimeError("expected total rotation dims == 2")
    if rank < 2:
        raise RuntimeError("expected total dims >= 2")
    if any(axis < -rank or axis >= rank for axis in dims):
        raise RuntimeError("rotation dim out of range")
    dim0, dim1 = (axis % rank for axis in dims)
    if dim0 == dim1:
        raise RuntimeError("expected rotation dims to be different")

    x = input
    if x.is_complex():
        x = torch.view_as_real(x.resolve_conj())
    shape = list(x.shape)
    strides = list(x.stride())
    k %= 4
    base_offset = 0
    # Reverse the input axes before exchanging them for odd rotations.
    if k in (2, 3):
        base_offset += (shape[dim0] - 1) * strides[dim0]
        strides[dim0] = -strides[dim0]
    if k in (1, 2):
        base_offset += (shape[dim1] - 1) * strides[dim1]
        strides[dim1] = -strides[dim1]
    if k % 2:
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        strides[dim0], strides[dim1] = strides[dim1], strides[dim0]

    out = torch.empty(shape, dtype=x.dtype, device=x.device)
    n_elements = out.numel()
    if n_elements:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        with torch_device_fn.device(x.device):
            rot90_kernel[grid](
                x, out, n_elements, tuple(shape), tuple(strides), base_offset
            )
    return torch.view_as_complex(out) if input.is_complex() else out
