# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400-specific _foreach_mul / _foreach_mul_ (all 4 overloads).

The shared foreach executor carries the tensor list on a 2-D grid, and GCU400's ``grid.y`` cap of
255 makes large shapes ``out of resource``. Contiguous float lists go through
``_foreach_flat``'s 1-D fixed-grid / metadata-fused kernel; non-float, non-contiguous and
broadcast forms still delegate to the shared executor to keep dtype policy and error semantics.

Uploading the fused kernel's metadata (``torch.tensor(..., device=)``) is a synchronous H2D of
~90us per call, pinning every uncached overload at a shape-independent ~100-200us. The four
metadata segments hold only addresses and element counts, while the kernel still reads the actual
tensor data on every call, so caching exactly by ``(device, address, element count)`` gives zero
H2D on steady-state calls (the allocator reuses the same addresses); the ScalarList fp32 scalar
table is likewise read-only and cached independently by value.
"""

import logging

import torch
import triton

from flag_gems.ops._foreach_binary import _apply_list, _apply_scalar, _apply_tensor
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
    fused_binary_scalar_kernel,
    run_binary,
)

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# Generated per round by the optimization driver script; all compile-time / launch-time config, semantics unchanged.
FIXED_GRID = 96
BLOCK_SIZE = 16384
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
# Small shapes use the common binary executor (its 2D grid stays in range and its cache is more complete).
PUBLIC_SMALL_SHAPE = 0
SMALL_SHAPE_MAX_NUMEL = 261120
# --- TUNING PARAMS END ---

# Read-only metadata cache: the key contains only device and address/element count, not tensor data; the kernel still
# fully reads inputs and recomputes each time, so address reuse or in-place content changes do not affect correctness.
_SCALAR_CACHE = {}
# (num_tensors, max_numel, block) -> grid tuple, see _run_mul_cached.
_GRID_CACHE = {}


def _cached_scalar_tensor(scalars, device):
    """The ScalarList fp32 scalar table (read-only), cached by (device, value tuple)."""
    key = (str(device), tuple(scalars))
    cached = _SCALAR_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_SCALAR_CACHE) >= SCALAR_CACHE_LIMIT:
        _SCALAR_CACHE.clear()
    cached = torch.tensor(scalars, dtype=torch.float32, device=device)
    _SCALAR_CACHE[key] = cached
    return cached


def _run_mul_cached(self, operand, mode, inplace):
    """Lean single-launch path for the four operand forms (metadata goes through the read-only cache).

    Inapplicable forms return ``FALLBACK`` and the caller picks the fallback target: broadcast
    ``.List`` needs ``run_binary``'s per-tensor kernel; the rest fall back to the shared executor.
    The four metadata segments (input/output/second-operand addresses plus element counts) are
    identical in layout to ``_foreach_flat``'s fused kernel.
    """
    if not can_use_flat(self, operand, mode):
        return FALLBACK
    num_tensors = len(self)
    in_ptrs = [tensor.data_ptr() for tensor in self]
    outputs = self if inplace else [torch.empty_like(tensor) for tensor in self]
    out_ptrs = in_ptrs if inplace else [tensor.data_ptr() for tensor in outputs]
    numels = [tensor.numel() for tensor in self]
    first = self[0]
    triton_dtype = _TL_DTYPES[first.dtype]
    device = first.device
    # The fused grid only depends on (list length, largest numel, block width): cache it so a steady
    # stream of same-shaped calls does not re-scan the list for its maximum.
    grid_key = (num_tensors, max(numels), BLOCK_SIZE)
    grid_edges = _GRID_CACHE.get(grid_key)
    if grid_edges is None:
        grid_edges = (
            max(1, min(FIXED_GRID, num_tensors * triton.cdiv(grid_key[1], BLOCK_SIZE))),
        )
        _GRID_CACHE[grid_key] = grid_edges
    grid = grid_edges

    if mode == MODE_LIST:
        # Same-shape lists share flat indices; reuse the numel vector already built above instead of
        # walking both lists pairwise again (32 x numel() per call is measurable at this launch scale).
        if numels != [other.numel() for other in operand]:
            # Broadcast .List does not share flat indices with the inputs, so hand it to the per-tensor kernel.
            return FALLBACK
        b_ptrs = [other.data_ptr() for other in operand]
        meta = _device_meta(in_ptrs + out_ptrs + b_ptrs + numels, device)
        fused_binary_list_kernel[grid](
            meta,
            num_tensors,
            OP_MUL,
            triton_dtype,
            FP32_PROMOTE,
            BLOCK_SIZE,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    elif mode == MODE_SCALAR_LIST:
        if USE_SCALAR_CACHE:
            scalar_tensor = _cached_scalar_tensor(operand, device)
        else:
            scalar_tensor = torch.tensor(operand, dtype=torch.float32, device=device)
        meta = _device_meta(in_ptrs + out_ptrs + [0] * num_tensors + numels, device)
        fused_binary_scalar_kernel[grid](
            meta,
            scalar_tensor,
            num_tensors,
            OP_MUL,
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
    elif mode == MODE_TENSOR:
        # A 0-d tensor is often fp32; it must be read using its own dtype, not the list dtype.
        b_dtype = _TL_DTYPES.get(operand.dtype, triton_dtype)
        meta = _device_meta(
            in_ptrs + out_ptrs + [operand.data_ptr()] * num_tensors + numels, device
        )
        fused_binary_scalar_kernel[grid](
            meta,
            operand,
            num_tensors,
            OP_MUL,
            triton_dtype,
            1,
            b_dtype,
            0.0,
            0,
            FP32_PROMOTE,
            BLOCK_SIZE,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    else:
        meta = _device_meta(in_ptrs + out_ptrs + [0] * num_tensors + numels, device)
        fused_binary_scalar_kernel[grid](
            meta,
            self[0],
            num_tensors,
            OP_MUL,
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
    """Lean single-launch path first; broadcast/anomalous forms fall back to the shared semantics."""
    if FALLBACK_TO_PUBLIC:
        return fallback()
    if USE_META_CACHE:
        result = _run_mul_cached(self, operand, mode, inplace)
        if result is not FALLBACK:
            return result
        if mode != MODE_LIST:
            # can_use_flat failed: leave dtype policy and exception semantics to the common executor.
            return fallback()
    result = run_binary(
        self,
        operand,
        OP_MUL,
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


def _foreach_mul_List(self, other):
    logger.debug("GEMS_GCU400 _FOREACH_MUL_SCALAR")
    return _dispatch(
        self, other, MODE_LIST, False, lambda: _apply_list("mul", self, other, False)
    )


def _foreach_mul_Scalar(self, scalar):
    return _dispatch(
        self,
        scalar,
        MODE_SCALAR,
        False,
        lambda: _apply_scalar("mul", self, scalar, False),
    )


def _foreach_mul_ScalarList(self, scalars):
    logger.debug("GEMS_GCU400 _FOREACH_MUL_SCALARLIST")
    return _dispatch(
        self,
        scalars,
        MODE_SCALAR_LIST,
        False,
        lambda: _apply_scalar("mul", self, scalars, False),
    )


def _foreach_mul_Tensor(self, other):
    logger.debug("GEMS_GCU400 _FOREACH_MUL_TENSOR")
    return _dispatch(
        self,
        other,
        MODE_TENSOR,
        False,
        lambda: _apply_tensor("mul", self, other, False),
    )


def _foreach_mul__List(self, other):
    logger.debug("GEMS_GCU400 _FOREACH_MUL__LIST")
    return _dispatch(
        self, other, MODE_LIST, True, lambda: _apply_list("mul", self, other, True)
    )


def _foreach_mul__Scalar(self, scalar):
    logger.debug("GEMS_GCU400 _FOREACH_MUL__SCALAR")
    return _dispatch(
        self,
        scalar,
        MODE_SCALAR,
        True,
        lambda: _apply_scalar("mul", self, scalar, True),
    )


def _foreach_mul__ScalarList(self, scalars):
    logger.debug("GEMS_GCU400 _FOREACH_MUL__SCALARLIST")
    return _dispatch(
        self,
        scalars,
        MODE_SCALAR_LIST,
        True,
        lambda: _apply_scalar("mul", self, scalars, True),
    )


def _foreach_mul__Tensor(self, other):
    logger.debug("GEMS_GCU400 _FOREACH_MUL__TENSOR")
    return _dispatch(
        self, other, MODE_TENSOR, True, lambda: _apply_tensor("mul", self, other, True)
    )
