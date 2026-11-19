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

"""GCU400-specific scalar overloads of _foreach_minimum / _foreach_minimum_.

The common foreach executor carries the tensor list in a 2D grid, and GCU400's ``grid.y``
limit of 255 causes ``out of resource`` on large shapes. Contiguous float lists use the
1D fixed grid / metadata fused kernel from ``_foreach_flat``; non-float,
non-contiguous or broadcast shapes still delegate to the common executor to keep dtype policy and exception semantics.
"""

import logging

from flag_gems.ops._foreach_binary import _apply_scalar

from ._foreach_flat import FALLBACK, MODE_SCALAR, MODE_SCALAR_LIST, OP_MIN, run_binary

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


def _foreach_minimum_Scalar(self, scalar):
    logger.debug("GEMS_GCU400 _FOREACH_MINIMUM_SCALAR")

    def fallback():
        return _apply_scalar("minimum", self, scalar, inplace=False)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        scalar,
        OP_MIN,
        MODE_SCALAR,
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


def _foreach_minimum_ScalarList(self, scalars):
    logger.debug("GEMS_GCU400 _FOREACH_MINIMUM_SCALARLIST")

    def fallback():
        return _apply_scalar("minimum", self, scalars, inplace=False)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        scalars,
        OP_MIN,
        MODE_SCALAR_LIST,
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


def _foreach_minimum__Scalar(self, scalar):
    logger.debug("GEMS_GCU400 _FOREACH_MINIMUM__SCALAR")

    def fallback():
        return _apply_scalar("minimum", self, scalar, inplace=True)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        scalar,
        OP_MIN,
        MODE_SCALAR,
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


def _foreach_minimum__ScalarList(self, scalars):
    logger.debug("GEMS_GCU400 _FOREACH_MINIMUM__SCALARLIST")

    def fallback():
        return _apply_scalar("minimum", self, scalars, inplace=True)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        scalars,
        OP_MIN,
        MODE_SCALAR_LIST,
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
