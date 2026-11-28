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

"""GCU400-specific ``_foreach_clamp_min_.List`` (in-place).

The common ``foreach_binary_list`` carries the tensor list in a 2D grid, and GCU400's
``grid.y`` limit of 255 causes ``out of resource`` on large shapes. Contiguous float
lists use a 1D fixed grid + in-kernel striding maximum kernel instead; non-float / non-contiguous /
broadcast length mismatches still delegate to the common ``_apply_list`` to keep dtype policy.
"""

import logging

from flag_gems.ops._foreach_binary import _apply_list
from flag_gems.utils.foreach import _device_meta

from ._foreach_flat import (
    _TL_DTYPES,
    FALLBACK,
    MODE_LIST,
    OP_MAX,
    _fused_grid,
    can_use_flat,
    fused_binary_list_kernel,
    run_binary,
)

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# Generated per round by the optimization-iteration driver script; all are compile-time/launch-time settings, not
# changing operator semantics.
# Geometry sweep on this machine's new stack (Triton GCU 4.1.6) (27 configs × 3 rounds interleaved in one process): 24
# programs × 16K tile wins across all shapes/dtypes for 16-tensor lists (1024² fp32
# 64.1us = 2.9TB/s streaming peak, vs 48×8K at 85.4us; grids 16/32/48 are all worse
# by 35-75%), consistent with the ``_foreach_binary`` family setting of 24/16384.
FIXED_GRID = 24
BLOCK_SIZE = 16384
NUM_WARPS = 4
NUM_STAGES = 1
USE_DEVICE_CONTEXT = 0
FP32_PROMOTE = 0
FUSED_LAUNCH = 1
FALLBACK_TO_PUBLIC = 0
# Read-only metadata cache: the key contains only device and addresses/numels, so steady-state calls incur zero H2D.
USE_META_CACHE = 1
# --- TUNING PARAMS END ---


def _run_clamp_min_cached(self, other):
    """Single-launch path for paired same-shape float lists (metadata via read-only cache).

    A broadcast ``.List`` (different second-operand element count) does not share the flat
    index with the input, and shapes outside ``can_use_flat`` also return ``FALLBACK`` for the
    caller to route. The four-part metadata layout (input/output/second-operand addresses plus
    element counts) matches ``_foreach_flat``'s fused kernel; in place the output address is
    the input address and per-element ``maximum`` allows same-buffer write back.
    """
    if not can_use_flat(self, other, MODE_LIST):
        return FALLBACK
    if not all(
        other_tensor.numel() == tensor.numel()
        for tensor, other_tensor in zip(self, other)
    ):
        return FALLBACK
    num_tensors = len(self)
    in_ptrs = [tensor.data_ptr() for tensor in self]
    out_ptrs = in_ptrs
    b_ptrs = [tensor.data_ptr() for tensor in other]
    numels = [tensor.numel() for tensor in self]
    meta = _device_meta(in_ptrs + out_ptrs + b_ptrs + numels, self[0].device)
    grid = (_fused_grid(self, FIXED_GRID, BLOCK_SIZE),)
    fused_binary_list_kernel[grid](
        meta,
        num_tensors,
        OP_MAX,
        _TL_DTYPES[self[0].dtype],
        FP32_PROMOTE,
        BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return None


def _foreach_clamp_min__List(self, other):
    logger.debug("GEMS_GCU400 FOREACH_CLAMP_MIN__LIST")

    def fallback():
        return _apply_list("clamp_min", self, other, inplace=True)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    if USE_META_CACHE:
        result = _run_clamp_min_cached(self, other)
        if result is not FALLBACK:
            return result
    result = run_binary(
        self,
        other,
        OP_MAX,
        MODE_LIST,
        True,
        fixed_grid=FIXED_GRID,
        block_size=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        use_device_context=USE_DEVICE_CONTEXT,
        fp32_promote=FP32_PROMOTE,
        fused=FUSED_LAUNCH,
    )
    if result is FALLBACK:
        return fallback()
    return result
