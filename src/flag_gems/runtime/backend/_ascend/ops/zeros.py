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
_MAX_CORE_DIM = 65535


def _zeros_grid(n_elements, block_size):
    return (max(1, min(triton.cdiv(n_elements, block_size), _MAX_CORE_DIM)),)


@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
    SHAPE: tl.constexpr = (),
    STRIDES: tl.constexpr = (),
):
    # Keep the launch within coreDim's limit and the live tile within the UB.
    # Cast before multiplying so tensors larger than INT32_MAX remain valid.
    for block_id in range(
        ext.program_id(0),
        tl.cdiv(tl.cast(n_elements, tl.int64), BLOCK_SIZE),
        ext.num_programs(0),
    ):
        for sub_block_start_idx in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
            index = (
                block_id.to(tl.int64) * BLOCK_SIZE
                + sub_block_start_idx
                + tl.arange(0, BLOCK_SIZE_SUB)
            )
            offset = index
            if len(SHAPE) > 0:
                # The tensor pointer already includes its storage offset.
                offset = tl.full((BLOCK_SIZE_SUB,), 0, tl.int64)
                remaining = index
                for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
                    offset += (remaining % SHAPE[dim]) * STRIDES[dim]
                    remaining = remaining // SHAPE[dim]
            tl.store(output_ptr + offset, 0.0, mask=index < n_elements)


def zeros(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_ASCEND ZEROS")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    grid_fn = lambda meta: _zeros_grid(N, meta["BLOCK_SIZE"])
    with torch_device_fn.device(device):
        zeros_kernel[grid_fn](out, N, BLOCK_SIZE=20480, BLOCK_SIZE_SUB=1024)
    return out


def zero_(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_ASCEND ZERO_")
    N = x.numel()
    dense = x.is_contiguous() or torch.ops.aten.is_non_overlapping_and_dense(x)
    shape = () if N == 0 or dense else tuple(x.shape)
    strides = () if N == 0 or dense else tuple(x.stride())
    grid_fn = lambda meta: _zeros_grid(N, meta["BLOCK_SIZE"])
    with torch_device_fn.device(x.device):
        zeros_kernel[grid_fn](
            x, N, BLOCK_SIZE=20480, BLOCK_SIZE_SUB=1024, SHAPE=shape, STRIDES=strides
        )
    return x
