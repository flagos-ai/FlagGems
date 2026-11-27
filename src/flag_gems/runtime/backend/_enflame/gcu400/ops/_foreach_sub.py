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

"""GCU400-specific .Scalar and .ScalarList overloads of _foreach_sub / _foreach_sub_.

The common foreach executor carries the tensor list in a 2D grid, and GCU400's ``grid.y``
limit of 255 causes ``out of resource`` on large shapes. Contiguous float lists use the
1D fixed grid / metadata fused kernel from ``_foreach_flat``; non-float,
non-contiguous or broadcast shapes still delegate to the common executor to keep dtype policy and exception semantics.
"""

import logging

import torch

from flag_gems.ops._foreach_binary import _apply_scalar
from flag_gems.utils.foreach import _device_meta

from ._foreach_flat import (
    _TL_DTYPES,
    FALLBACK,
    MODE_SCALAR,
    MODE_SCALAR_LIST,
    OP_SUB,
    _fused_grid,
    can_use_flat,
    fused_binary_scalar_kernel,
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
# All overloads reuse the read-only _device_meta cache, eliminating the synchronous H2D on steady-state calls.
USE_META_CACHE = 1
# The fp32 scalar table for .ScalarList is cached separately by (device, value).
USE_SCALAR_CACHE = 1
SCALAR_CACHE_LIMIT = 64
# --- TUNING PARAMS END ---

# Read-only metadata cache: the key contains only device and address/element count, not tensor data; the kernel still
# fully reads inputs and recomputes each time, so address reuse or in-place content changes do not affect correctness.
_SCALAR_CACHE = {}


def _cached_scalar_tensor(scalars, device):
    """fp32 scalar table for ScalarList (read-only), cached by (device, value tuple)."""
    key = (str(device), tuple(scalars))
    cached = _SCALAR_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_SCALAR_CACHE) >= SCALAR_CACHE_LIMIT:
        _SCALAR_CACHE.clear()
    cached = torch.tensor(scalars, dtype=torch.float32, device=device)
    _SCALAR_CACHE[key] = cached
    return cached


def _run_sub_cached(self, operand, mode, inplace):
    """Lean single-launch path for Scalar / ScalarList (metadata via read-only cache).

    Shapes that do not apply return ``FALLBACK``: this file only serves the two scalar forms
    and the caller covers everything else. The four-part metadata layout (input/output/
    second-operand addresses plus element counts) matches ``_foreach_flat``'s fused kernel.
    """
    if mode not in (MODE_SCALAR, MODE_SCALAR_LIST) or not can_use_flat(
        self, operand, mode
    ):
        return FALLBACK
    num_tensors = len(self)
    in_ptrs = [tensor.data_ptr() for tensor in self]
    outputs = self if inplace else [torch.empty_like(tensor) for tensor in self]
    out_ptrs = in_ptrs if inplace else [tensor.data_ptr() for tensor in outputs]
    numels = [tensor.numel() for tensor in self]
    triton_dtype = _TL_DTYPES[self[0].dtype]
    device = self[0].device
    grid = (_fused_grid(self, FIXED_GRID, BLOCK_SIZE),)
    meta = _device_meta(in_ptrs + out_ptrs + [0] * num_tensors + numels, device)

    if mode == MODE_SCALAR_LIST:
        if USE_SCALAR_CACHE:
            scalar_tensor = _cached_scalar_tensor(operand, device)
        else:
            scalar_tensor = torch.tensor(operand, dtype=torch.float32, device=device)
        fused_binary_scalar_kernel[grid](
            meta,
            scalar_tensor,
            num_tensors,
            OP_SUB,
            triton_dtype,
            0,
            triton_dtype,
            0.0,
            1,
            FP32_PROMOTE,
            BLOCK_SIZE,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    else:
        fused_binary_scalar_kernel[grid](
            meta,
            self[0],
            num_tensors,
            OP_SUB,
            triton_dtype,
            0,
            triton_dtype,
            float(operand),
            0,
            FP32_PROMOTE,
            BLOCK_SIZE,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    return None if inplace else outputs


def _dispatch(self, operand, mode, inplace, fallback):
    """The lean single-launch path wins; odd shapes fall back to the shared semantics."""
    if FALLBACK_TO_PUBLIC:
        return fallback()
    if USE_META_CACHE:
        result = _run_sub_cached(self, operand, mode, inplace)
        if result is not FALLBACK:
            return result
        # can_use_flat failed: leave dtype policy and exception semantics to the common executor.
        return fallback()
    result = run_binary(
        self,
        operand,
        OP_SUB,
        mode,
        inplace,
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


def _foreach_sub_Scalar(self, scalar):
    logger.debug("GEMS_GCU400 _FOREACH_SUB_SCALARLIST")
    return _dispatch(
        self,
        scalar,
        MODE_SCALAR,
        False,
        lambda: _apply_scalar("sub", self, scalar, False),
    )


def _foreach_sub_ScalarList(self, scalars):
    return _dispatch(
        self,
        scalars,
        MODE_SCALAR_LIST,
        False,
        lambda: _apply_scalar("sub", self, scalars, False),
    )


def _foreach_sub__Scalar(self, scalar):
    logger.debug("GEMS_GCU400 _FOREACH_SUB__SCALAR")
    return _dispatch(
        self,
        scalar,
        MODE_SCALAR,
        True,
        lambda: _apply_scalar("sub", self, scalar, True),
    )


def _foreach_sub__ScalarList(self, scalars):
    logger.debug("GEMS_GCU400 _FOREACH_SUB__SCALARLIST")
    return _dispatch(
        self,
        scalars,
        MODE_SCALAR_LIST,
        True,
        lambda: _apply_scalar("sub", self, scalars, True),
    )
