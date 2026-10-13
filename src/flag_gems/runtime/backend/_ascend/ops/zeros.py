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

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.shape_utils import volume

device_ = device
logger = logging.getLogger(__name__)


@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    pid = ext.program_id(axis=0)

    for block_start_idx in range(
        pid * BLOCK_SIZE, n_elements, tl.num_programs(0) * BLOCK_SIZE
    ):
        for sub_block_start_idx in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
            sub_offset = (
                block_start_idx + sub_block_start_idx + tl.arange(0, BLOCK_SIZE_SUB)
            )
            mask = sub_offset < n_elements
            tl.store(output_ptr + sub_offset, 0.0, mask=mask)


@triton.jit
def zero_strided_kernel(
    output_ptr,
    n_elements,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    DIVISORS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    for block_start in range(
        pid * BLOCK_SIZE, n_elements, tl.num_programs(0) * BLOCK_SIZE
    ):
        index = block_start + tl.arange(0, BLOCK_SIZE)
        offset = tl.full((BLOCK_SIZE,), 0, tl.int64)
        for axis in tl.static_range(len(SHAPE)):
            offset += (index // DIVISORS[axis] % SHAPE[axis]) * STRIDES[axis]
        tl.store(output_ptr + offset, 0, mask=index < n_elements)


def zeros(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_ASCEND ZEROS")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    grid_fn = lambda meta: (min(max(triton.cdiv(N, meta["BLOCK_SIZE"]), 1), 65535),)
    with torch_device_fn.device(device):
        zeros_kernel[grid_fn](out, N, BLOCK_SIZE=20480, BLOCK_SIZE_SUB=1024)
    return out


def zero_(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    if not x.is_contiguous():
        shape = tuple(x.shape)
        strides = tuple(x.stride())
        divisors = [1] * x.ndim
        for axis in range(x.ndim - 2, -1, -1):
            divisors[axis] = divisors[axis + 1] * shape[axis + 1]
        N = x.numel()
        grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK_SIZE"]), 65535),)
        with torch_device_fn.device(x.device):
            zero_strided_kernel[grid_fn](
                x,
                N,
                SHAPE=shape,
                STRIDES=strides,
                DIVISORS=tuple(divisors),
                BLOCK_SIZE=1024,
            )
        return x

    N = x.numel()
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK_SIZE"]), 65535),)
    with torch_device_fn.device(x.device):
        zeros_kernel[grid_fn](x, N, BLOCK_SIZE=20480, BLOCK_SIZE_SUB=1024)
    return x
