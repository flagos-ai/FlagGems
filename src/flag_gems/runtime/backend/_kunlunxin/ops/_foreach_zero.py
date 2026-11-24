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
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_reduction import _foreach_zero as _generic_foreach_zero
from flag_gems.ops._foreach_reduction import _foreach_zero_ as _generic_foreach_zero_
from flag_gems.utils.foreach import check_tensor_list, tl_dtype

logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def _metadata_for(device_key, _dtype, _n_elements, pointers):
    return torch.tensor(pointers, dtype=torch.int64).to(
        torch.device(device_key), non_blocking=True
    )


@triton.jit
def _foreach_zero_unmasked_kernel(
    meta_ptr,
    DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tensor_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    dst = tl.load(meta_ptr + tensor_idx).to(tl.pointer_type(DTYPE))
    offsets = chunk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(dst + offsets, 0.0)


def _pick_block(n_elements):
    if n_elements >= 262_144:
        return 131072, 32
    if n_elements >= 65_536:
        return 32768, 16
    if n_elements >= 16_384:
        return 16384, 16
    if n_elements >= 4_096:
        return 4096, 8
    if n_elements >= 1_024:
        return 1024, 4
    return 128, 4


def _try_fast(tensors):
    if not tensors:
        return False
    first = tensors[0]
    dtype = first.dtype
    device = first.device
    n_elements = first.numel()
    if n_elements == 0 or n_elements >= 2**31:
        return False

    for tensor in tensors:
        if tensor.dtype != dtype or tensor.device != device:
            return False
        if tensor.numel() != n_elements or not tensor.is_contiguous():
            return False

    block_size, num_warps = _pick_block(n_elements)
    if n_elements % block_size != 0:
        return False

    pointers = tuple(tensor.data_ptr() for tensor in tensors)
    metadata = _metadata_for(str(device), dtype, n_elements, pointers)
    grid = (len(tensors), n_elements // block_size)
    _foreach_zero_unmasked_kernel[grid](
        metadata,
        tl_dtype(dtype),
        block_size,
        num_warps=num_warps,
        unroll_num=16,
        buffer_size_limit=8192,
        isCloseMemoryAsync=False,
    )
    return True


def _foreach_zero(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_ZERO")
    tensors = check_tensor_list(self)
    if not _try_fast(tensors):
        return _generic_foreach_zero(tensors)
    return tensors


def _foreach_zero_(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_ZERO_")
    tensors = check_tensor_list(self)
    if not _try_fast(tensors):
        return _generic_foreach_zero_(tensors)
    return None


__all__ = ["_foreach_zero", "_foreach_zero_"]
