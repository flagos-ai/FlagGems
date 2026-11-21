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

from flag_gems.ops.masked_select import masked_select as _generic_masked_select

logger = logging.getLogger(__name__)


def _empty_broadcast_shape(left, right):
    # Check EVERY axis, including axes after a zero dimension. In particular,
    # (0, 3) and (2, 3) cannot broadcast, but (0, 3) and (1, 3) produce (0, 3).
    rank = max(len(left), len(right))
    left = (1,) * (rank - len(left)) + tuple(left)
    right = (1,) * (rank - len(right)) + tuple(right)
    result = []
    for dim, (a, b) in enumerate(zip(left, right)):
        if a == b:
            result.append(a)
        elif a == 1:
            result.append(b)
        elif b == 1:
            result.append(a)
        else:
            raise RuntimeError(
                f"The size of tensor a ({a}) must match the size of tensor b "
                f"({b}) at non-singleton dimension {dim}"
            )
    return tuple(result)


def masked_select(inp, mask):
    logger.debug("GEMS_HYGON MASKED_SELECT")
    # A nonempty all-false mask is NOT a metadata-known empty result: retain
    # the generic counting kernels and their exact-cardinality synchronization.
    if inp.numel() != 0 and mask.numel() != 0:
        return _generic_masked_select(inp, mask)
    if inp.layout != torch.strided or mask.layout != torch.strided:
        return _generic_masked_select(inp, mask)
    if mask.dtype != torch.bool:
        raise RuntimeError("masked_select: expected BoolTensor for mask")
    if inp.device != mask.device:
        raise RuntimeError("masked_select: input and mask must be on the same device")
    # Match the historical F03 domain; leave experimental/quantized dtypes to
    # the original provider rather than introducing support through an early exit.
    supported = (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
    if inp.dtype not in supported:
        return _generic_masked_select(inp, mask)
    common = _empty_broadcast_shape(tuple(inp.shape), tuple(mask.shape))
    if 0 not in common:
        return _generic_masked_select(inp, mask)
    # Cardinality is exactly zero from validated metadata. No payload reads,
    # mask reduction, scalar transfer, contiguous materialization or kernel.
    return torch.empty((0,), dtype=inp.dtype, device=inp.device)
