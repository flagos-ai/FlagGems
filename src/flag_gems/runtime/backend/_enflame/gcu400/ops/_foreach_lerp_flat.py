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
"""GCU400-specific ``_foreach_lerp(.Scalar/.ScalarList)`` 1-D fused implementation.

The shared ``foreach_ternary`` executor launches on a 2-D ``grid=(block, tensor)`` and
GCU400's ``grid.y`` hardware cap is 255; the 1-D form of ``_flat_ternary``,
``grid=min(24, nt*parts)``, additionally drops tail tensor tiles when the grid does not divide
the tensor count (nt=16 gives 24%16!=0, tensors 8-15 lose half their tiles). This file rewrites
both scalar-weight paths with the always-divisible construction of ``_foreach_lerp__List``,
finishing the whole list in one launch:

1. ``grid = NT * PARTS`` is always a multiple of the tensor count, each tensor gets exactly
   ``PARTS`` programs splitting tiles with a ``PARTS`` stride, and coverage holes do not exist;
2. the ``.Scalar`` broadcast value enters the kernel as an fp32 by-value argument (zero H2D, no
   scalar table) and no longer detours through the 24-grid family hole of ``_flat_ternary``;
3. per-tensor ``.ScalarList`` weights are packed as fp32 bit patterns into int64 metadata and go
   through the shared ``_device_meta`` read-only cache with addresses and counts (zero H2D on steady state);
4. fp32 inputs enable the full-block mask-free split loop (the tail tile is finished by a single
   program with a mask), while 16-bit re-measured 3-5% worse and keeps the whole-block mask loop.

Non-float, non-contiguous, mixed dtype, cross-device and 0-d scalar forms are still delegated by
the caller to the shared executor, with dtype policy and error semantics unchanged.
"""

import logging
import struct
from typing import Dict, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.foreach import _device_meta

from ._foreach_flat import _TL_DTYPES, FLOAT_DTYPES

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START (foreach_lerp_scalar_list_) ---
# Geometry sweep (new stack Triton GCU 4.1.6, 26 configurations interleaved over P{1..6}xB{4096..32768})
# Confirmed P3B8192w4s1 as the best overall (31.4µs, runner-up P3B16384 33.2µs), matching
# the ``_foreach_lerp__List`` / addcdiv family convention; all are compile-time/launch-time settings,
# without changing operator semantics.
FAST_PARTS_CAP = 3
FAST_BLOCK = 8192
FAST_WARPS = 4
FAST_STAGES = 1
FP32_PROMOTE = 1
FP32_MASK_FREE = 1
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END (foreach_lerp_scalar_list_) ---

# GCU400 grid.x hard limit; the fast-path grid = NT * PARTS must stay entirely within the limit.
_MAX_GRID_X = 65535

# Return this sentinel when the fast path is unsuitable and let the caller delegate to the common executor.
FALLBACK = object()

# Read-only cache of fp32 bit patterns for per-tensor weights: benchmark steady-state weights are unchanged, so a hit
# avoids struct packing on every call; out-of-range scalars (fp32 overflow) return None and are handed back to
# the common executor. The key is a pure Python numeric value and contains no tensor data.
_WEIGHT_BITS_CACHE: Dict[Tuple, list] = {}
_WEIGHT_BITS_CACHE_MAX = 64


def _weight_bits(weight):
    key = tuple(float(value) for value in weight)
    cached = _WEIGHT_BITS_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        bits = [struct.unpack("<i", struct.pack("<f", value))[0] for value in key]
    except OverflowError:
        return None
    if len(_WEIGHT_BITS_CACHE) >= _WEIGHT_BITS_CACHE_MAX:
        _WEIGHT_BITS_CACHE.clear()
    _WEIGHT_BITS_CACHE[key] = bits
    return bits


@libentry()
@triton.jit
def fused_lerp_scalar_list_kernel(
    meta_ptr,
    weight,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    WEIGHT_MODE: tl.constexpr,
    MASK_FREE: tl.constexpr,
    BLOCK: tl.constexpr,
    PARTS: tl.constexpr,
):
    """One launch performs ``self + weight * (tensors1 - self)`` over the whole list.

    ``meta_ptr`` holds the input, output, second-operand addresses and element counts (NT items
    each, int64); with ``WEIGHT_MODE=1`` ``weight`` is a broadcast fp32 scalar, and with 0 the
    fifth segment holds per-tensor fp32 bit patterns. The grid is linearized by ``(tensor, part)``
    and is always ``NT * PARTS``: each tensor gets ``PARTS`` programs splitting tiles with a
    ``PARTS`` stride, so coverage holes do not exist by construction.
    """
    pid = tl.program_id(0)
    tensor_id = pid % NT
    part_id = pid // NT

    a_ptr = tl.load(meta_ptr + tensor_id).to(tl.pointer_type(A_DT))
    out_ptr = tl.load(meta_ptr + NT + tensor_id).to(tl.pointer_type(A_DT))
    b_ptr = tl.load(meta_ptr + 2 * NT + tensor_id).to(tl.pointer_type(A_DT))
    n_elements = tl.load(meta_ptr + 3 * NT + tensor_id)
    if WEIGHT_MODE:
        w = weight
    else:
        raw = tl.load(meta_ptr + 4 * NT + tensor_id)
        w = tl.cast(raw.to(tl.int32), tl.float32, bitcast=True)

    arange = tl.arange(0, BLOCK)
    # Keep the tile count as i32, consistent with the address-arithmetic width of the common executor.
    num_tiles = tl.cdiv(n_elements, BLOCK).to(tl.int32)
    if MASK_FREE:
        # Mask-free full blocks: the main loop generates no per-element comparisons/masks, and the tail tile is
        # finalized
        # by the single program that owns it (the tail tile always has a part assignment, because every
        # tensor has programs with PARTS part_id values from 0..PARTS-1).
        num_full = (n_elements // BLOCK).to(tl.int32)
        for block_id in tl.range(part_id, num_full, PARTS):
            idx = block_id * BLOCK + arange
            a = tl.load(a_ptr + idx)
            b = tl.load(b_ptr + idx)
            if FP32_PROMOTE:
                a = a.to(tl.float32)
                b = b.to(tl.float32)
            tl.store(out_ptr + idx, a + w * (b - a))
        if num_full < num_tiles and part_id == num_full % PARTS:
            idx = num_full * BLOCK + arange
            mask = idx < n_elements
            a = tl.load(a_ptr + idx, mask=mask)
            b = tl.load(b_ptr + idx, mask=mask)
            if FP32_PROMOTE:
                a = a.to(tl.float32)
                b = b.to(tl.float32)
            tl.store(out_ptr + idx, a + w * (b - a), mask=mask)
    else:
        for block_id in tl.range(part_id, num_tiles, PARTS):
            idx = block_id * BLOCK + arange
            mask = idx < n_elements
            a = tl.load(a_ptr + idx, mask=mask)
            b = tl.load(b_ptr + idx, mask=mask)
            if FP32_PROMOTE:
                a = a.to(tl.float32)
                b = b.to(tl.float32)
            tl.store(out_ptr + idx, a + w * (b - a), mask=mask)


def _can_use_flat(self, tensors1, weight):
    """Returns the weight form (1=broadcast scalar / 0=per-tensor table), None when preconditions fail.

    Only same dtype, same device, contiguous, equally long pairs, non-empty, with a pure numeric
    weight scalar (or scalar list) pass; tensor weights (including 0-d), mixed dtype,
    cross-device, non-contiguous and empty tensors all fall back to the shared executor with unchanged errors.
    """
    if not isinstance(self, (list, tuple)) or not self:
        return None
    if not isinstance(tensors1, (list, tuple)) or len(tensors1) != len(self):
        return None
    if isinstance(weight, (list, tuple)):
        if len(weight) != len(self):
            return None
        if not all(isinstance(value, (int, float)) for value in weight):
            return None
        weight_mode = 0
    elif isinstance(weight, (int, float)):
        weight_mode = 1
    else:
        return None
    if len(self) * FAST_PARTS_CAP > _MAX_GRID_X:
        return None
    first = self[0]
    if not isinstance(first, torch.Tensor) or first.dtype not in FLOAT_DTYPES:
        return None
    dtype = first.dtype
    device = first.device
    for tensor, other in zip(self, tensors1):
        if not isinstance(tensor, torch.Tensor) or not isinstance(other, torch.Tensor):
            return None
        if tensor.dtype != dtype or other.dtype != dtype:
            return None
        if tensor.device != device or other.device != device:
            return None
        if not tensor.is_contiguous() or not other.is_contiguous():
            return None
        if tensor.numel() == 0 or other.numel() != tensor.numel():
            return None
    return weight_mode


def run_lerp_scalar_list(self, tensors1, weight, inplace: bool):
    """Executes lerp ``.Scalar`` / ``.ScalarList``; returns ``FALLBACK`` when unsuitable.

    Out-of-place outputs are allocated with ``torch.empty_like``; metadata holds only addresses,
    element counts and weight bit patterns, goes through the shared ``_device_meta`` read-only
    cache (zero synchronous H2D on steady state), and the kernel still reads and computes the
    tensor data on every call.
    """
    if FALLBACK_TO_PUBLIC:
        return FALLBACK
    weight_mode = _can_use_flat(self, tensors1, weight)
    if weight_mode is None:
        return FALLBACK

    num_tensors = len(self)
    if inplace:
        outputs = None
        out_tensors = self
    else:
        outputs = [torch.empty_like(tensor) for tensor in self]
        out_tensors = outputs

    a_ptrs = [tensor.data_ptr() for tensor in self]
    out_ptrs = [tensor.data_ptr() for tensor in out_tensors]
    b_ptrs = [tensor.data_ptr() for tensor in tensors1]
    numels = [tensor.numel() for tensor in self]
    if weight_mode:
        # Broadcast scalar: the fp32 by-value argument is passed with the launch, and metadata has no scalar segment.
        values = numels
        scalar = float(weight)
    else:
        # Per-tensor weights: fp32 bit patterns are cached along with the int64 metadata.
        bits = _weight_bits(weight)
        if bits is None:
            return FALLBACK
        values = numels + bits
        scalar = 0.0
    meta = _device_meta(a_ptrs + out_ptrs + b_ptrs + values, self[0].device)

    dtype = self[0].dtype
    parts = max(1, min(FAST_PARTS_CAP, triton.cdiv(max(numels), FAST_BLOCK)))
    fused_lerp_scalar_list_kernel[(num_tensors * parts,)](
        meta,
        scalar,
        num_tensors,
        _TL_DTYPES[dtype],
        FP32_PROMOTE,
        weight_mode,
        int(FP32_MASK_FREE and dtype == torch.float32),
        FAST_BLOCK,
        parts,
        num_warps=FAST_WARPS,
        num_stages=FAST_STAGES,
    )
    return None if inplace else outputs
