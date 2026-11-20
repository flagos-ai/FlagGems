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

"""GCU400-specific ``_foreach_lerp__List``: per-tensor 1D flat kernel with three inputs.

The common ``foreach_ternary`` / ``foreach_binary_list`` rely on a device-side chunk table and
a 2D grid, which on GCU400 needs int64 pointer tables and easily exceeds the grid.y limit of 255.
This uses a per-tensor 1D fixed grid with in-kernel striding and fp32 computation throughout.
"""

import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import _apply_lerp
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---


# Generated per round by the optimization driver script; compile-time / launch-time config, semantics unchanged.
FIXED_GRID = 48
BLOCK_SIZE = 65536
NUM_WARPS = 4
NUM_STAGES = 1
USE_DEVICE_CONTEXT = 1
REVERSE_LAUNCH_ORDER = 0
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END ---

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@libentry()
@triton.jit(do_not_specialize=["N", "SCALAR"])
def _foreach_lerp_list_flat_kernel(
    p0, p1, p2, out_ptr, N, SCALAR, BLOCK: tl.constexpr, FIXED_GRID: tl.constexpr
):
    """1D fixed grid with striding for _foreach_lerp__List."""
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(N, BLOCK), num_programs):
        offs = block_id * BLOCK + arange
        mask = offs < N
        a = tl.load(p0 + offs, mask=mask).to(tl.float32)
        b = tl.load(p1 + offs, mask=mask).to(tl.float32)
        c = tl.load(p2 + offs, mask=mask).to(tl.float32)
        result = a + c * (b - a)
        tl.store(out_ptr + offs, result, mask=mask)


def _can_use_flat(tensors):
    """All tensors must be contiguous float tensors of the same dtype."""
    if not tensors or len(tensors) == 0:
        return False
    first = tensors[0]
    if first.dtype not in _FLOAT_DTYPES or first.dtype.is_complex:
        return False
    for tensor in tensors:
        if tensor is None or tensor.dtype != first.dtype:
            return False
        if not tensor.is_contiguous() or tensor.shape != first.shape:
            return False
    return True


def _flat_ready(self, others, thirds):
    if FALLBACK_TO_PUBLIC:
        return False
    if len(self) == 0 or len(self) != len(others):
        return False
    if thirds is not None and len(thirds) != len(self):
        return False
    groups = [self, others] + ([thirds] if thirds is not None else [])
    for group in groups:
        if not _can_use_flat(group):
            return False
    for index in range(len(self)):
        if others[index].shape != self[index].shape:
            return False
        if thirds is not None and thirds[index].shape != self[index].shape:
            return False
    return True


def _run_flat(self, others, thirds, scalar, inplace):
    indices = list(range(len(self)))
    if REVERSE_LAUNCH_ORDER:
        indices.reverse()
    if not indices:
        return [] if not inplace else None
    outputs = [None] * len(self) if not inplace else None
    device_context = (
        torch_device_fn.device(self[0].device) if USE_DEVICE_CONTEXT else nullcontext()
    )
    with device_context:
        for index in indices:
            first, second = self[index], others[index]
            third = thirds[index] if thirds is not None else second
            num_elems = first.numel()
            out_tensor = first if inplace else torch.empty_like(first)
            # Empty tensors have no elements, so skip the kernel launch; but the empty result must still be kept
            # in the output list to stay one-to-one with the input (foreach returns as many tensors as it receives).
            if num_elems == 0:
                if not inplace:
                    outputs[index] = out_tensor
                continue
            grid = (min(triton.cdiv(num_elems, BLOCK_SIZE), FIXED_GRID),)
            _foreach_lerp_list_flat_kernel[grid](
                first,
                second,
                third,
                out_tensor,
                num_elems,
                scalar,
                BLOCK_SIZE,
                FIXED_GRID,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
            )
            if not inplace:
                outputs[index] = out_tensor
    return None if inplace else outputs


def _foreach_lerp__List(self, tensors1, weight):
    logger.debug("GEMS_GCU400 FOREACH_LERP__LIST")
    if not _flat_ready(
        self, tensors1, weight if isinstance(weight, (list, tuple)) else None
    ):
        return _apply_lerp(self, tensors1, weight, True)
    return _run_flat(self, tensors1, weight, 0.0, inplace=True)
