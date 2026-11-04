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

"""Hygon fast path for contiguous FP32 functional masked_fill."""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.masked_fill import masked_fill as _masked_fill
from flag_gems.runtime import device, torch_device_fn

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 1024
_MAX_NUMEL = 1 << 20
_FP32_MAX = 3.4028234663852886e38


@triton.jit
def _masked_fill_contiguous(inp, mask, out, value, BLOCK: tl.constexpr):
    # The gate guarantees a positive multiple of BLOCK and dense, zero-offset
    # storage for both inputs, so every lane has a valid address.
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(inp + offsets)
    m = tl.load(mask + offsets)
    tl.store(out + offsets, tl.where(m, value.to(tl.float32), x))


def _can_use_contiguous(inp, mask, value):
    # Keep tensor values and scalar conversion/error handling in the generic op.
    if type(value) is not float:
        return False
    if math.isfinite(value) and abs(value) > _FP32_MAX:
        return False
    if type(inp) is not torch.Tensor or type(mask) is not torch.Tensor:
        return False
    if inp.dtype != torch.float32 or mask.dtype != torch.bool:
        return False
    if inp.layout != torch.strided or mask.layout != torch.strided:
        return False
    if inp.device.type != device.name or mask.device != inp.device:
        return False
    if inp.requires_grad or mask.requires_grad:
        return False
    if inp.shape != mask.shape or not 1 <= inp.ndim <= 8:
        return False
    n = inp.numel()
    if n == 0 or n > _MAX_NUMEL or n % _BLOCK_SIZE != 0:
        return False
    if not inp.is_contiguous() or not mask.is_contiguous():
        return False
    if inp.storage_offset() != 0 or mask.storage_offset() != 0:
        return False
    if inp.is_conj() or inp.is_neg() or mask.is_conj() or mask.is_neg():
        return False
    return True


def masked_fill(inp, mask, value):
    logger.debug("GEMS_HYGON MASKED_FILL")
    if not _can_use_contiguous(inp, mask, value):
        return _masked_fill(inp, mask, value)
    out = torch.empty_like(inp)
    with torch_device_fn.device(inp.device):
        _masked_fill_contiguous[(inp.numel() // _BLOCK_SIZE,)](
            inp, mask, out, value, BLOCK=_BLOCK_SIZE, num_warps=4
        )
    return out
