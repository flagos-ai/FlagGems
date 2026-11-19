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

"""GCU400-specific _foreach_mul / _foreach_mul_ (all 4 overloads).

The common foreach executor carries the tensor list in a 2D grid, and GCU400's ``grid.y``
limit of 255 causes ``out of resource`` on large shapes. Contiguous float lists use the
1D fixed grid / metadata fused kernel from ``_foreach_flat``; non-float,
non-contiguous or broadcast shapes still delegate to the common executor to keep dtype policy and exception semantics.
"""

import logging
from contextlib import nullcontext

import torch

from flag_gems.ops._foreach_binary import _apply_list, _apply_scalar, _apply_tensor
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.foreach import _device_meta

from ._foreach_flat import (
    _TL_DTYPES,
    FALLBACK,
    MODE_LIST,
    MODE_SCALAR,
    MODE_SCALAR_LIST,
    MODE_TENSOR,
    OP_MUL,
    _fused_grid,
    can_use_flat,
    fused_binary_list_kernel,
    run_binary,
)

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
# .List specific: reuse the common _device_meta cache instead of rebuilding int64 metadata each time.
USE_CACHED_META = 1
# Small shapes use the common binary executor (its 2D grid stays in bounds and its cache is more complete).
PUBLIC_SMALL_SHAPE = 0
SMALL_SHAPE_MAX_NUMEL = 261120
# --- TUNING PARAMS END ---


def _run_mul_list_cached(self, other, outputs):
    """Build int64 metadata using the common _device_meta cache, avoiding a synchronous H2D per call.

    The four-segment layout (input/output/second-operand addresses + element counts) must exactly match
    _foreach_flat._fused_meta, otherwise the fused kernel reads wrong pointers.
    """
    num_tensors = len(self)
    pointers = (
        [tensor.data_ptr() for tensor in self]
        + [tensor.data_ptr() for tensor in outputs]
        + [other_tensor.data_ptr() for other_tensor in other]
        + [tensor.numel() for tensor in self]
    )
    meta = _device_meta(pointers, self[0].device)
    grid = (_fused_grid(self, FIXED_GRID, BLOCK_SIZE),)
    device_context = (
        torch_device_fn.device(self[0].device) if USE_DEVICE_CONTEXT else nullcontext()
    )
    with device_context:
        fused_binary_list_kernel[grid](
            meta,
            num_tensors,
            OP_MUL,
            _TL_DTYPES[self[0].dtype],
            FP32_PROMOTE,
            BLOCK_SIZE,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    return outputs


def _foreach_mul_List(self, other):
    logger.debug("GEMS_GCU400 _FOREACH_MUL_LIST")

    def fallback():
        return _apply_list("mul", self, other, inplace=False)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    if (
        PUBLIC_SMALL_SHAPE
        and self
        and max(t.numel() for t in self) <= SMALL_SHAPE_MAX_NUMEL
    ):
        return fallback()
    if USE_CACHED_META and can_use_flat(self, other, MODE_LIST):
        fused_ready = all(
            other_tensor.numel() == tensor.numel()
            for tensor, other_tensor in zip(self, other)
        )
        if fused_ready:
            outputs = [torch.empty_like(tensor) for tensor in self]
            return _run_mul_list_cached(self, other, outputs)
    result = run_binary(
        self,
        other,
        OP_MUL,
        MODE_LIST,
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


def _foreach_mul_Scalar(self, scalar):
    logger.debug("GEMS_GCU400 _FOREACH_MUL_SCALAR")

    def fallback():
        return _apply_scalar("mul", self, scalar, inplace=False)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        scalar,
        OP_MUL,
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


def _foreach_mul_ScalarList(self, scalars):
    logger.debug("GEMS_GCU400 _FOREACH_MUL_SCALARLIST")

    def fallback():
        return _apply_scalar("mul", self, scalars, inplace=False)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        scalars,
        OP_MUL,
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


def _foreach_mul_Tensor(self, other):
    logger.debug("GEMS_GCU400 _FOREACH_MUL_TENSOR")

    def fallback():
        return _apply_tensor("mul", self, other, inplace=False)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        other,
        OP_MUL,
        MODE_TENSOR,
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


def _foreach_mul__List(self, other):
    logger.debug("GEMS_GCU400 _FOREACH_MUL__LIST")

    def fallback():
        return _apply_list("mul", self, other, inplace=True)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        other,
        OP_MUL,
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


def _foreach_mul__Scalar(self, scalar):
    logger.debug("GEMS_GCU400 _FOREACH_MUL__SCALAR")

    def fallback():
        return _apply_scalar("mul", self, scalar, inplace=True)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        scalar,
        OP_MUL,
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


def _foreach_mul__ScalarList(self, scalars):
    logger.debug("GEMS_GCU400 _FOREACH_MUL__SCALARLIST")

    def fallback():
        return _apply_scalar("mul", self, scalars, inplace=True)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        scalars,
        OP_MUL,
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


def _foreach_mul__Tensor(self, other):
    logger.debug("GEMS_GCU400 _FOREACH_MUL__TENSOR")

    def fallback():
        return _apply_tensor("mul", self, other, inplace=True)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        other,
        OP_MUL,
        MODE_TENSOR,
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
