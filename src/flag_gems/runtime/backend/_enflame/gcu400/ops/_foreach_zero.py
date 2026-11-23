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
from typing import List, Sequence

import torch
import triton
import triton.language as tl

from flag_gems.utils.foreach import check_tensor_list

logger = logging.getLogger(__name__)

# The generic _foreach_reduction._zero_list_kernel uses a 2-D grid
# (NT, max_chunks) plus an int64 device metadata table holding tensor pointers
# and element counts.  Neither works on GCU400: grid.y is capped at 255 while
# max_chunks reaches 65536, and int64 is silently replaced with int32 by the
# torch_gcu transfer layer, truncating 64-bit pointers.  This backend instead
# launches one flat 1-D Triton kernel per tensor, with no device-side pointer
# table at all.
_MAX_GRID_X = 65535
_BASE_BLOCK = 131072


@triton.jit
def _foreach_zero_kernel(out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    tl.store(
        out_ptr + offsets,
        tl.zeros((BLOCK,), dtype=out_ptr.dtype.element_ty),
        mask=mask,
    )


def _zero_one(t: torch.Tensor, block: int, num_warps: int) -> None:
    n = t.numel()
    if n == 0:
        return
    if not t.is_contiguous():
        staged = torch.empty_like(t, memory_format=torch.contiguous_format)
        _zero_one(staged, block, num_warps)
        t.copy_(staged)
        return
    # Grow BLOCK so the flat grid stays within the GCU400 grid.x limit.
    while triton.cdiv(n, block) > _MAX_GRID_X:
        block *= 2
    _foreach_zero_kernel[(triton.cdiv(n, block),)](t, n, block, num_warps=num_warps)


def _zero_list_fast(tensors: List[torch.Tensor]) -> bool:
    block = _BASE_BLOCK
    num_warps = 4
    for tensor in tensors:
        _zero_one(tensor, block, num_warps)
    return True


def _foreach_zero(self: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    logger.debug("GEMS_GCU400 _FOREACH_ZERO")
    tensors = check_tensor_list(self)
    _zero_list_fast(tensors)
    return tensors


def _foreach_zero_(self: Sequence[torch.Tensor]) -> None:
    logger.debug("GEMS_GCU400 _FOREACH_ZERO_")
    tensors = check_tensor_list(self)
    _zero_list_fast(tensors)
    return None


__all__ = ["_foreach_zero", "_foreach_zero_"]
