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

"""GCU400-specific foreach ScalarList maximum-family operators.

The in-place overload metadata is fully read-only and holds no tensor data, so it can be cached
safely by ``(device, address, numel, scalar bit pattern)``, removing repeated synchronous H2D;
out-of-place uses ``PACK_OUTPUTS``: allocate the whole output buffer contiguously once and put
each tensor's element offset into static metadata (cached with the input), while the output base
address travels as a kernel scalar argument at launch (zero H2D); the 16 output views are built
with a single ``torch.as_strided`` (the narrow+view double-dispatch host chain is visible inside
the do_bench timing window). The launch goes through the ``_launch_fused_max`` replay cache
(mirroring ``_foreach_binary``'s ``_launch_ternary_flat``): on a steady-state hit it skips
Triton's Python dispatch, and a launch hook (topsprof etc.) or any exception falls back
automatically. This file and the clamp_min version keep separate tuning parameters and metadata
caches so iterations on the two operators do not interfere; non-float, non-contiguous and
scalar-count-mismatch shapes fall back to the shared executor, dtype policy and errors unchanged.
"""

import logging
import struct
from typing import Any, Dict, List

import torch
import triton
import triton.language as tl
from triton import knobs
from triton.runtime import driver as _triton_driver

from flag_gems.ops._foreach_binary import _apply_scalar
from flag_gems.utils import libentry

from ._foreach_flat import (
    _TL_DTYPES,
    FALLBACK,
    MODE_SCALAR_LIST,
    OP_MAX,
    can_use_flat,
    run_binary,
)

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# Compile-time / launch-time config, adjusted per optimization round; semantics unchanged.
FIXED_GRID = 48
BLOCK_SIZE = 8192
NUM_WARPS = 4
NUM_STAGES = 1
FP32_PROMOTE = 0
FUSED_LAUNCH = 1
FALLBACK_TO_PUBLIC = 0
USE_SINGLE_H2D = 1
USE_META_CACHE = 1
META_CACHE_LIMIT = 32
# Out-of-place output allocation: 0 = per-tensor empty_like + address table H2D;
# 1 = one contiguous allocation + output base address as a kernel scalar argument (zero H2D);
# 2 = one contiguous allocation + a 1-entry int64 address table H2D.
PACK_OUTPUTS = 1
# --- TUNING PARAMS END ---


@libentry()
@triton.jit
def fused_max_scalarlist_kernel(
    meta_ptr,
    out_ptr_meta,
    out_base,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_OUT: tl.constexpr,
    PACK_OUT: tl.constexpr,
    ELEM_BYTES: tl.constexpr,
):
    """Compute ``max(tensor, scalar_i)`` for the whole list in one launch.

    The three segments of ``meta_ptr`` are input addresses, element counts, and each fp32 scalar bit pattern;
    with ``HAS_OUT=1`` output addresses are read from ``out_ptr_meta``, otherwise input is written in place.
    grid.x is linearized over ``(tensor, part)`` to avoid the ``grid.y <= 255`` limit.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tensor_id = pid % NT
    part_id = pid // NT
    num_parts = tl.cdiv(num_programs, NT)

    in_raw = tl.load(meta_ptr + tensor_id)
    num_elems = tl.load(meta_ptr + NT + tensor_id)
    scalar_raw = tl.load(meta_ptr + 2 * NT + tensor_id)
    if HAS_OUT:
        if PACK_OUT == 1:
            # The output base address is a kernel scalar argument; static metadata stores only each tensor's offset.
            out_offset = tl.load(meta_ptr + 3 * NT + tensor_id)
            out_raw = out_base + out_offset * ELEM_BYTES
        elif PACK_OUT == 2:
            out_raw = tl.load(out_ptr_meta)
        else:
            out_raw = tl.load(out_ptr_meta + tensor_id)
    else:
        out_raw = in_raw
    in_ptr = in_raw.to(tl.pointer_type(A_DT))
    out_ptr = out_raw.to(tl.pointer_type(A_DT))
    scalar = tl.cast(scalar_raw.to(tl.int32), tl.float32, bitcast=True)

    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(part_id, tl.cdiv(num_elems, BLOCK), num_parts):
        offs = block_id * BLOCK + arange
        mask = offs < num_elems
        value = tl.load(in_ptr + offs, mask=mask)
        if FP32_PROMOTE:
            result = tl.maximum(value.to(tl.float32), scalar)
        else:
            result = tl.maximum(value, scalar.to(value.dtype))
        tl.store(out_ptr + offs, result, mask=mask)


# Read-only metadata cache: the key contains only device, addresses, element counts and scalar bit patterns, no tensor
# data;
# the kernel still reads input fully and recomputes each time, so address reuse or in-place content changes stay
# correct.
_META_CACHE = {}


def _get_cached_meta(values, device):
    cache_key = (str(device),) + tuple(values)
    cached = _META_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if len(_META_CACHE) >= META_CACHE_LIMIT:
        _META_CACHE.clear()
    cached = torch.tensor(values, dtype=torch.int64, device=device)
    _META_CACHE[cache_key] = cached
    return cached


# Steady-state launch replay, mirroring ``_foreach_binary._launch_ternary_flat``: all arguments except the two
# metadata tables and the output base address are constexpr; ``(device, NT, dtype,
# promote, block, has_out, pack_out, elem_bytes, grid, alignment bits)`` uniquely determines the
# compiled variant, and replay only swaps in runtime buffers and scalars. The precondition for a hit is that there is no
# launch hook
# (tools such as topsprof install hooks and automatically fall back to normal dispatch); any exception from the memo or
# dict
# evicts the cache and falls back, and no device-visible state is cached.
_LAUNCH_CACHE: Dict[Any, Any] = {}
_LAUNCH_LAST: List[Any] = [None] * 8


def _launch_fused_max(
    grid,
    meta,
    out_ptr_meta,
    out_base,
    nt,
    dt,
    promote,
    block,
    has_out,
    pack_out,
    elem_bytes,
):
    """Cached launcher for ``fused_max_scalarlist_kernel[grid](...)``."""
    runtime = knobs.runtime
    last = _LAUNCH_LAST
    spec = (grid, nt, dt, promote, block, has_out, pack_out, elem_bytes)
    if (
        last[0] is meta
        and last[1] is out_ptr_meta
        and last[2] is dt
        and last[4] == spec
        and not (out_base & 15)
        and not runtime.launch_enter_hook.calls
        and not runtime.launch_exit_hook.calls
    ):
        try:
            last[5](
                grid[0],
                1,
                1,
                _triton_driver.active.get_current_stream(meta.device.index),
                last[6],
                last[7],
                None,
                None,
                None,
                meta,
                out_ptr_meta,
                out_base,
                nt,
                dt,
                promote,
                block,
                has_out,
                pack_out,
                elem_bytes,
            )
            return
        except Exception:
            last[0] = None
    key = (
        fused_max_scalarlist_kernel,
        meta.device,
        nt,
        dt,
        promote,
        block,
        has_out,
        pack_out,
        elem_bytes,
        grid,
        not (meta.data_ptr() & 15),
        not (out_base & 15),
    )
    entry = _LAUNCH_CACHE.get(key)
    if (
        entry is not None
        and not runtime.launch_enter_hook.calls
        and not runtime.launch_exit_hook.calls
    ):
        try:
            entry[0](
                grid[0],
                1,
                1,
                _triton_driver.active.get_current_stream(meta.device.index),
                entry[1],
                entry[2],
                None,
                None,
                None,
                meta,
                out_ptr_meta,
                out_base,
                nt,
                dt,
                promote,
                block,
                has_out,
                pack_out,
                elem_bytes,
            )
            last[0] = meta
            last[1] = out_ptr_meta
            last[2] = dt
            last[4] = spec
            last[5], last[6], last[7] = entry
            return
        except Exception:
            _LAUNCH_CACHE.pop(key, None)
            last[0] = None
    compiled = fused_max_scalarlist_kernel[grid](
        meta,
        out_ptr_meta,
        out_base,
        nt,
        dt,
        promote,
        block,
        has_out,
        pack_out,
        elem_bytes,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    if compiled is not None:
        kernel = compiled[0] if isinstance(compiled, tuple) else compiled
        _LAUNCH_CACHE[key] = (
            kernel.run,
            kernel.function,
            kernel.packed_metadata,
        )


def _run_max_scalarlist(self, scalars, inplace):
    """Single-H2D fast path; unsupported shapes return FALLBACK to the common executor."""
    if not can_use_flat(self, scalars, MODE_SCALAR_LIST):
        return FALLBACK
    num_tensors = len(self)
    scalar_bits = [
        struct.unpack("<i", struct.pack("<f", float(value)))[0] for value in scalars
    ]
    numels = [tensor.numel() for tensor in self]
    out_offsets = []
    running_offset = 0
    for numel in numels:
        out_offsets.append(running_offset)
        running_offset += numel
    # The fourth segment holds each tensor's element offset in the contiguous output buffer (constant across calls).
    static_values = (
        [tensor.data_ptr() for tensor in self] + numels + scalar_bits + out_offsets
    )
    if USE_META_CACHE:
        meta = _get_cached_meta(static_values, self[0].device)
    else:
        meta = torch.tensor(static_values, dtype=torch.int64, device=self[0].device)

    if inplace:
        outputs = self
        out_ptr_meta = meta
        out_base = 0
        has_out = 0
        pack_out = 0
    elif PACK_OUTPUTS:
        # One contiguous allocation replaces 16 empty_like calls; outputs are independent views on the same buffer,
        # with no overlap between segments, so in-place modification of one output by the user does not affect the
        # others. The views are
        # constructed once with as_strided (narrow+view requires two dispatches; for 16 tensors the
        # host chain is visible within the do_bench timing window, and the sibling _foreach_scalarlist_opt measured
        # on the order of 66µs -> 19µs).
        flat_output = torch.empty(
            running_offset, dtype=self[0].dtype, device=self[0].device
        )
        outputs = []
        element_offset = 0
        for tensor in self:
            outputs.append(
                torch.as_strided(
                    flat_output, tensor.shape, tensor.stride(), element_offset
                )
            )
            element_offset += tensor.numel()
        if PACK_OUTPUTS == 1:
            out_ptr_meta = meta
            out_base = flat_output.data_ptr()
        else:
            out_ptr_meta = torch.tensor(
                [flat_output.data_ptr()],
                dtype=torch.int64,
                device=self[0].device,
            )
            out_base = 0
        has_out = 1
        pack_out = PACK_OUTPUTS
    else:
        outputs = [torch.empty_like(tensor) for tensor in self]
        out_ptr_meta = torch.tensor(
            [tensor.data_ptr() for tensor in outputs],
            dtype=torch.int64,
            device=self[0].device,
        )
        out_base = 0
        has_out = 1
        pack_out = 0

    # The grid is linearly expanded as (tensor, part) and must always be an integer multiple of NT: the kernel uses
    # ``pid % NT / pid // NT`` to decompose it and strides by ``cdiv(num_programs, NT)`` to fetch tiles,
    # while a non-divisible ``min(cap, NT*tiles)`` causes the trailing ``grid % NT`` tensors to miss odd
    # tiles (a red-line precedent for the family; see _foreach_binary._flat_grid).
    blocks_per_tensor = triton.cdiv(max(numels), BLOCK_SIZE)
    parts = min(-(-FIXED_GRID // num_tensors), blocks_per_tensor)
    grid = (num_tensors * parts,)
    _launch_fused_max(
        grid,
        meta,
        out_ptr_meta,
        out_base,
        num_tensors,
        _TL_DTYPES[self[0].dtype],
        FP32_PROMOTE,
        BLOCK_SIZE,
        has_out,
        pack_out,
        self[0].element_size(),
    )
    return None if inplace else outputs


def _dispatch(self, scalars, inplace, name):
    """Dispatch between the common executor, generic fusion and the single-H2D fast path by tuning switches."""
    if FALLBACK_TO_PUBLIC:
        return _apply_scalar(name, self, scalars, inplace=inplace)
    if USE_SINGLE_H2D:
        result = _run_max_scalarlist(self, scalars, inplace)
        if result is not FALLBACK:
            return result
        return _apply_scalar(name, self, scalars, inplace=inplace)
    result = run_binary(
        self,
        scalars,
        OP_MAX,
        MODE_SCALAR_LIST,
        inplace,
        fixed_grid=FIXED_GRID,
        block_size=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        use_device_context=0,
        fp32_promote=FP32_PROMOTE,
        fused=FUSED_LAUNCH,
    )
    if result is FALLBACK:
        return _apply_scalar(name, self, scalars, inplace=inplace)
    return result


def _foreach_maximum_ScalarList(self, scalars):
    logger.debug("GEMS_GCU400 _FOREACH_MAXIMUM_SCALARLIST")
    return _dispatch(self, scalars, False, "maximum")


def _foreach_maximum__ScalarList(self, scalars):
    logger.debug("GEMS_GCU400 _FOREACH_MAXIMUM__SCALARLIST")
    return _dispatch(self, scalars, True, "maximum")
