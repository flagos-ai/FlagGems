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

"""GCU400-specific ``_foreach_cos_``: per-tensor 1D flat kernel.

The common ``foreach_unary`` executor relies on a device-side chunk table and a 2D grid, which on GCU400
needs int64 pointer tables and easily exceeds the grid.y limit of 255. This uses a per-tensor
1D fixed grid with in-kernel striding and fp32 computation for accuracy consistent with the common implementation.
"""

import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import apply_unary
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---


# Generated per round by the optimization driver script; compile-time / launch-time config, semantics unchanged.
FIXED_GRID = 48
BLOCK_SIZE = 65536
NUM_WARPS = 2
NUM_STAGES = 1
USE_DEVICE_CONTEXT = 1
REVERSE_LAUNCH_ORDER = 0
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END ---

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _foreach_cos_flat_kernel(
    in_ptr, out_ptr, N, BLOCK: tl.constexpr, FIXED_GRID: tl.constexpr
):
    """1D fixed grid with striding for elementwise cos.

    in_ptr and out_ptr differ only in the out-of-place variant; avoiding a host-side clone prevents
    hitting the grid.x limit of the torch_gcu copy kernel on very large tensors.
    """
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(N, BLOCK), num_programs):
        offs = block_id * BLOCK + arange
        mask = offs < N
        value = tl.load(in_ptr + offs, mask=mask).to(tl.float32)
        result = tl.cos(value)
        tl.store(out_ptr + offs, result, mask=mask)


def _can_use_flat_kernel(self):
    """Accept only non-empty lists of contiguous float tensors."""
    if len(self) == 0:
        return False
    try:
        for tensor in self:
            if tensor.dtype not in _FLOAT_DTYPES or tensor.dtype.is_complex:
                return False
            if not tensor.is_contiguous():
                return False
    except AttributeError:
        # Non-Tensor elements: return False so the caller falls back to the common apply_unary for matching exceptions.
        return False
    return True


def _run_flat_kernel(self, inplace):
    if FALLBACK_TO_PUBLIC or not _can_use_flat_kernel(self):
        return apply_unary("cos", self, inplace)
    tensors = list(self)
    if not tensors:
        return [] if not inplace else None

    # Reversing only changes launch order; out-of-place outputs must keep the same order as the original list.
    launch_indices = list(range(len(tensors)))
    if REVERSE_LAUNCH_ORDER:
        launch_indices.reverse()
    outputs = [None] * len(tensors) if not inplace else None
    device_context = (
        torch_device_fn.device(tensors[0].device)
        if USE_DEVICE_CONTEXT
        else nullcontext()
    )
    with device_context:
        for index in launch_indices:
            tensor = tensors[index]
            num_elems = tensor.numel()
            out_tensor = tensor if inplace else torch.empty_like(tensor)
            # Empty tensors have no elements, so skip the kernel launch; but the empty result must still be kept
            # in the output list to stay one-to-one with the input (foreach returns as many tensors as it receives).
            if num_elems == 0:
                if not inplace:
                    outputs[index] = out_tensor
                continue
            grid = (min(triton.cdiv(num_elems, BLOCK_SIZE), FIXED_GRID),)
            _foreach_cos_flat_kernel[grid](
                tensor,
                out_tensor,
                num_elems,
                BLOCK_SIZE,
                FIXED_GRID,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
            )
            if not inplace:
                outputs[index] = out_tensor
    return None if inplace else outputs


def _foreach_cos_(self):
    logger.debug("GEMS_GCU400 _FOREACH_COS_")
    return _run_flat_kernel(self, inplace=True)
