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

from flag_gems.ops._functional_sym_constrain_range_for_size import (
    _functional_sym_constrain_range_for_size as default__functional_sym_constrain_range_for_size,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    first = pid * BLOCK_SIZE
    step = num_progs * BLOCK_SIZE
    for off in range(first, n_elements, step):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        val = tl.load(src_ptr + offsets, mask=mask, eviction_policy="evict_first")
        tl.store(dst_ptr + offsets, val, mask=mask, eviction_policy="evict_first")


# Element-size-specialized tile configs (probed on MTT S5000):
# 2-byte types (fp16/bf16) peak with 4096-elem/4-warp tiles; 4-byte types
# (fp32) peak with 2048-elem/4-warp tiles. Large copies use a persistent
# grid-stride launch capped at _MAX_BLOCKS to amortize block scheduling.
_BLOCK_2B = 4096
_BLOCK_4B = 2048
_NUM_WARPS = 4
_TINY = 1024  # single-block-style launch for tiny tensors
_MAX_BLOCKS = 960


def _functional_sym_constrain_range_for_size(*args, **kwargs):
    logger.debug("GEMS_MTHREADS _FUNCTIONAL_SYM_CONSTRAIN_RANGE_FOR_SIZE")
    # Find the dep_token tensor argument; if absent or not a musa tensor of supported dtype, fall back.
    dep_token = next(
        (arg for arg in args if isinstance(arg, torch.Tensor)),
        next(
            (value for value in kwargs.values() if isinstance(value, torch.Tensor)),
            None,
        ),
    )
    if dep_token is None:
        return default__functional_sym_constrain_range_for_size(*args, **kwargs)
    if (
        not isinstance(dep_token, torch.Tensor)
        or dep_token.device.type != "musa"
        or dep_token.dtype not in _SUPPORTED_DTYPES
        or not dep_token.is_contiguous()
        or dep_token.numel() == 0
    ):
        return default__functional_sym_constrain_range_for_size(*args, **kwargs)
    output = torch.empty_like(dep_token)
    n_elements = dep_token.numel()
    if dep_token.element_size() <= 2:
        block = _BLOCK_2B
    else:
        block = _BLOCK_4B
    if n_elements < _TINY:
        block = _TINY
    blocks = triton.cdiv(n_elements, block)
    if blocks > _MAX_BLOCKS:
        grid = (_MAX_BLOCKS,)
    else:
        grid = (blocks,)
    with torch_device_fn.device(dep_token.device):
        _copy_kernel[grid](
            dep_token, output, n_elements, BLOCK_SIZE=block, num_warps=_NUM_WARPS
        )
    return output


__all__ = ["_functional_sym_constrain_range_for_size"]
