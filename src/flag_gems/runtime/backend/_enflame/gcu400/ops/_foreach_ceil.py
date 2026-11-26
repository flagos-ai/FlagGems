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

"""GCU400-specific _foreach_ceil / _foreach_ceil_ (mostly in-place).

The common foreach executor carries the tensor list in a 2D grid, and GCU400's ``grid.y``
limit of 255 causes ``out of resource`` on large shapes. Contiguous float lists use the
1D fixed grid / metadata fused kernel from ``_foreach_flat``; non-float,
non-contiguous or broadcast shapes still delegate to the common executor to keep dtype policy and exception semantics.
"""

import logging

from flag_gems.ops._foreach_unary import apply_unary

from ._foreach_flat import FALLBACK, UNARY_CEIL, run_unary

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# Generated per round by the optimization driver script; all compile-time / launch-time config, semantics unchanged.
FIXED_GRID = 24
BLOCK_SIZE = 8192
NUM_WARPS = 4
NUM_STAGES = 1
USE_DEVICE_CONTEXT = 0
FP32_PROMOTE = 1
FUSED_LAUNCH = 1
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END ---


def _foreach_ceil_(self):
    logger.debug("GEMS_GCU400 _FOREACH_CEIL_")

    def fallback():
        return apply_unary("ceil", self, inplace=True)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_unary(
        self,
        UNARY_CEIL,
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


def _foreach_ceil(self):
    logger.debug("GEMS_GCU400 _FOREACH_CEIL")

    def fallback():
        return apply_unary("ceil", self, inplace=False)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_unary(
        self,
        UNARY_CEIL,
        False,
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
