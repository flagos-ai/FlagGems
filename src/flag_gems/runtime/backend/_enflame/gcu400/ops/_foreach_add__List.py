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

"""GCU400-specific ``_foreach_add_.List`` (in-place).

The common ``foreach_binary_list`` executor places the tensor address table on the device and uses a 2D grid
for chunk metadata; on GCU400 that int64 pointer table is truncated to int32 by torch_gcu, and grid.y is
limited to 255 by hardware. This uses a per-tensor 1D fixed grid with in-kernel striding instead, with no
device-side pointer table at all; non-float / complex / non-contiguous inputs fall back to the common implementation.
"""

import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import _apply_list
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---


# Generated per round by the optimization driver script; compile-time / launch-time config, semantics unchanged.
FIXED_GRID = 48
BLOCK_SIZE = 8192
NUM_WARPS = 4
NUM_STAGES = 1
USE_DEVICE_CONTEXT = 1
SKIP_EMPTY_TENSORS = 1
REVERSE_LAUNCH_ORDER = 0
FP32_ROUNDTRIP = 0
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END ---

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@libentry()
@triton.jit(do_not_specialize=["N", "alpha"])
def _foreach_add_list_flat_kernel(
    self_ptr,
    other_ptr,
    N,
    alpha,
    BLOCK: tl.constexpr,
    FP32_ROUNDTRIP: tl.constexpr,
):
    """1D fixed grid with striding, computing self += alpha * other."""
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(N, BLOCK), num_programs):
        offs = block_id * BLOCK + arange
        mask = offs < N
        lhs = tl.load(self_ptr + offs, mask=mask)
        rhs = tl.load(other_ptr + offs, mask=mask)
        if FP32_ROUNDTRIP:
            # Same as the common implementation: promote low precision to fp32 before adding, then implicitly cast back
            # on store.
            result = lhs.to(tl.float32) + alpha * rhs.to(tl.float32)
        else:
            result = lhs + alpha * rhs
        tl.store(self_ptr + offs, result, mask=mask)


def _can_use_flat_kernel(self, other):
    """Accept only contiguous float tensor lists of equal length and equal dtype."""
    if len(self) == 0:
        return False
    if len(self) != len(other):
        return False
    for lhs, rhs in zip(self, other):
        if lhs.dtype not in _FLOAT_DTYPES or lhs.dtype.is_complex:
            return False
        if rhs.dtype != lhs.dtype or rhs.shape != lhs.shape:
            return False
        if not lhs.is_contiguous() or not rhs.is_contiguous():
            return False
    return True


def _foreach_add__List(self, other, *, alpha=1):
    logger.debug("GEMS_GCU400 FOREACH_ADD__LIST")
    if FALLBACK_TO_PUBLIC or not _can_use_flat_kernel(self, other):
        return _apply_list("add", self, other, True, alpha)

    pairs = list(zip(self, other))
    if SKIP_EMPTY_TENSORS:
        pairs = [(lhs, rhs) for lhs, rhs in pairs if lhs.numel() > 0]
    if REVERSE_LAUNCH_ORDER:
        pairs = list(reversed(pairs))
    if not pairs:
        return None

    device_context = (
        torch_device_fn.device(self[0].device) if USE_DEVICE_CONTEXT else nullcontext()
    )
    with device_context:
        for lhs, rhs in pairs:
            num_elems = lhs.numel()
            grid = (min(triton.cdiv(num_elems, BLOCK_SIZE), FIXED_GRID),)
            _foreach_add_list_flat_kernel[grid](
                lhs,
                rhs,
                num_elems,
                alpha,
                BLOCK_SIZE,
                FP32_ROUNDTRIP,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
            )
    return None
