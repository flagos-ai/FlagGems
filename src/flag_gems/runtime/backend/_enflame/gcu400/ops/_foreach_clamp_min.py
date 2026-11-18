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

from ._foreach_flat import FALLBACK, MODE_LIST, OP_MAX, run_binary

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# Generated per round by the optimization driver script; all compile-time / launch-time config, semantics unchanged.
FIXED_GRID = 48
BLOCK_SIZE = 8192
NUM_WARPS = 4
NUM_STAGES = 1
USE_DEVICE_CONTEXT = 0
FP32_PROMOTE = 0
FUSED_LAUNCH = 1
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END ---


def _foreach_clamp_min__List(self, other):
    logger.debug("GEMS_GCU400 FOREACH_CLAMP_MIN__LIST")

    def fallback():
        return _apply_list("clamp_min", self, other, inplace=True)

    if FALLBACK_TO_PUBLIC:
        return fallback()
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
