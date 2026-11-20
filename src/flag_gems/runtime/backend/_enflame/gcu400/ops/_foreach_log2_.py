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

"""GCU400-specific ``_foreach_log2_`` (in-place).

The common ``foreach_unary`` executor carries the chunk table in a 2D grid; GCU400's
grid.y hardware limit is 255, so large shapes raise ``out of resource: grid.y``.
Float tensor lists use a 1D fixed grid + in-kernel striding log2 kernel instead,
with grid.x fixed at 48, removing the hardware limit and the per-tensor metadata cost;
non-float / complex / integral inputs still fall back to common ``apply_unary`` to keep dtype policy.
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
# Generated per round by the optimization driver script; compile-time config, semantics unchanged.
FIXED_GRID = 48
BLOCK_SIZE = 8192
NUM_WARPS = 4
USE_DEVICE_CONTEXT = 0
SKIP_EMPTY_TENSORS = 0
REVERSE_LAUNCH_ORDER = 1
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END ---

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _foreach_log2_flat_kernel(
    data_ptr,
    N,
    BLOCK: tl.constexpr,
    FIXED_GRID: tl.constexpr,
):
    """1D fixed grid with striding writing log2 back, supporting tensors of any length."""
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(N, BLOCK), num_programs):
        offs = block_id * BLOCK + arange
        mask = offs < N
        value = tl.load(data_ptr + offs, mask=mask).to(tl.float32)
        tl.store(data_ptr + offs, tl.log2(value), mask=mask)


def _foreach_log2_(self):
    logger.debug("GEMS_GCU400 FOREACH_LOG2_")
    if FALLBACK_TO_PUBLIC or not all(
        tensor.dtype in _FLOAT_DTYPES and not tensor.dtype.is_complex for tensor in self
    ):
        return apply_unary("log2", self, inplace=True)

    tensors = self
    if SKIP_EMPTY_TENSORS:
        tensors = [tensor for tensor in self if tensor.numel() > 0]
    if REVERSE_LAUNCH_ORDER:
        tensors = list(reversed(tensors))

    device_context = (
        torch_device_fn.device(self[0].device) if USE_DEVICE_CONTEXT else nullcontext()
    )
    with device_context:
        for tensor in tensors:
            num_elems = tensor.numel()
            grid = (min(triton.cdiv(num_elems, BLOCK_SIZE), FIXED_GRID),)
            _foreach_log2_flat_kernel[grid](
                tensor,
                num_elems,
                BLOCK_SIZE,
                FIXED_GRID,
                num_warps=NUM_WARPS,
            )
    return None
