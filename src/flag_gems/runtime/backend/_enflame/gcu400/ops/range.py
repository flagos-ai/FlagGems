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

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# Flat grid-stride rewrite of the seed per-block kernel (same structure as the
# proven relu.py flat path on this board). The seed kernel launched
# cdiv(size, 2048) tiny programs; its measured cost was ~52 ns per 2048-element
# program kernel-only (16M+1 fp16: 423 us) while a flat grid-stride kernel caps
# the grid at 48 looping CTAs and runs the same shape in 14.4 us (2.3 TB/s; do
# _bench kernel median, interleaved A/B probes 2026-11-17). The full wrapper
# path (alloc + launch) is 14.5 us for 16M+1 fp16 and 665 us for 1G+1 fp16
# (3.2 TB/s write roofline). Design points validated by probes:
#   * grid capped at 48 CTAs, each CTA loops over BLOCK-element tiles;
#     BLOCK ladder: size <= 8192 -> single-CTA next_pow2 (>=256);
#     size <= 2^23 -> 16384; else 65536. num_warps=2 for every flat config
#     (LibEntry's dispatch key omits num_warps, so BLOCK->nw must stay 1:1).
#   * EVEN constexpr arm elides the store mask only when size % BLOCK == 0;
#     otherwise the loop body stays uniformly masked. A "full tiles unmasked +
#     tail masked" variant with a per-tile scalar branch measured 29-41%
#     SLOWER (fp16 1G: 938 vs 665 us) - the branch breaks loop pipelining
#     while predicated stores are cheap.
#   * step == 1 keeps the multiply-free kernels (the fp32 multiply in the
#     general-step twin costs 2.9 us at 16M+1 fp16, +20%); torch.range calls
#     with a step other than 1 route to the range_step_* twins.
#   * `size % BLOCK == 0` is rare for the public API (torch.range is
#     inclusive: 2**n end -> 2**n + 1 elements), so the masked arm is the one
#     the official benchmark shapes always take.
#   * offsets stay in the int32 safe domain: the flat path is only used when
#     size <= _FLAT_MAX_NUMEL; larger sizes keep the seed per-block kernel
#     (which has the same int32 limits as before).
# GCU400 8-byte dtype containers ("Long/Double use Int/Float replace"):
#   * int64 readback consumes the full 8 bytes, so the int64 fast path writes
#     the value into the low 4 bytes and its sign extension into the high 4
#     bytes through result.view(torch.int32) - single kernel, no staging
#     buffer and no aten copy_ on the host path (values are computed in
#     int32, same i32 value domain as the previous staging implementation).
#   * fp64 has no arithmetic on this backend: keep the fp32 staging buffer +
#     value-level copy_ upcast (same precision behavior as before).
_FLAT_GRID_CAP = 48
_FLAT_MAX_NUMEL = 2**31 - 1 - 65536
# int64 direct-write path addresses the int32 view at 2*i (+1), so the view
# index must stay inside int32: 2*size - 1 <= 2**31 - 1.
_I64_LANES_MAX_NUMEL = 2**30

_INT_RANGE_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)
_FLOAT_RANGE_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


@libentry()
@triton.jit(do_not_specialize=["size"])
def range_func_flat(y_ptr, start, size, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    num_tiles = tl.cdiv(size, BLOCK)
    # tl.range (not the builtin): this module defines an op wrapper named
    # `range`, which shadows the builtin inside jit codegen and makes the
    # GCU frontend reject the loop iterator.
    for tile in tl.range(pid, num_tiles, num_ctas):
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            tl.store(y_ptr + offs, offs + start)
        else:
            mask = offs < size
            tl.store(y_ptr + offs, offs + start, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["size"])
def range_step_flat(y_ptr, start, step, size, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    # General-step twin of range_func_flat. The multiply costs 2.9 us at
    # 16M+1 fp16 (+20%), so step == 1 stays on the multiply-free kernel.
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    num_tiles = tl.cdiv(size, BLOCK)
    for tile in tl.range(pid, num_tiles, num_ctas):
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        vals = offs * step + start
        if EVEN:
            tl.store(y_ptr + offs, vals)
        else:
            mask = offs < size
            tl.store(y_ptr + offs, vals, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["size"])
def range_func_i64_lanes(y32_ptr, start, size, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    num_tiles = tl.cdiv(size, BLOCK)
    # int32 addressing through the int32 view: max index 2*size-1 stays inside
    # int32 because the host only routes size <= 2**30 here. This keeps the
    # whole kernel free of int64 arithmetic (independent of
    # TORCH_GCU_ENABLE_INT64_AND_UINT64).
    for tile in tl.range(pid, num_tiles, num_ctas):
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        vals = offs + start
        hi = vals >> 31
        if EVEN:
            tl.store(y32_ptr + offs * 2, vals)
            tl.store(y32_ptr + offs * 2 + 1, hi)
        else:
            mask = offs < size
            tl.store(y32_ptr + offs * 2, vals, mask=mask)
            tl.store(y32_ptr + offs * 2 + 1, hi, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["size"])
def range_step_i64_lanes(
    y32_ptr, start, step, size, BLOCK: tl.constexpr, EVEN: tl.constexpr
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    num_tiles = tl.cdiv(size, BLOCK)
    for tile in tl.range(pid, num_tiles, num_ctas):
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        vals = offs * step + start
        hi = vals >> 31
        if EVEN:
            tl.store(y32_ptr + offs * 2, vals)
            tl.store(y32_ptr + offs * 2 + 1, hi)
        else:
            mask = offs < size
            tl.store(y32_ptr + offs * 2, vals, mask=mask)
            tl.store(y32_ptr + offs * 2 + 1, hi, mask=mask)


@libentry()
@triton.jit
def range_func(y_ptr, start, step, size, BLOCK_SIZE: tl.constexpr):
    # Seed per-block kernel, kept for sizes outside the int32-safe flat domain
    # (BLOCK_SIZE 2048, one tile per program). The seed signature carried a
    # dead `end` argument; it is replaced by `step` for range_step parity
    # (step=1 folds back to the seed value expression).
    pid = tle.program_id(0)
    y_ptr += pid * BLOCK_SIZE
    offset = pid * BLOCK_SIZE

    cols = tl.arange(0, BLOCK_SIZE)
    range_val = (cols + offset) * step + start
    mask = cols + pid * BLOCK_SIZE
    tl.store(y_ptr + cols, range_val, mask=mask < size)


def _flat_grid(size, block):
    grid = -(-size // block)
    if grid > _FLAT_GRID_CAP:
        grid = _FLAT_GRID_CAP
    return grid


def _flat_block(size):
    if size <= 8192:
        block = 1 << (size - 1).bit_length() if size > 1 else 1
        return 256 if block < 256 else block
    if size <= (1 << 23):
        return 16384
    return 65536


def _launch_range(result, start, step, size, device, dtype):
    """Fill the pre-allocated ``result`` with ``start + i * step`` values."""
    unit_step = step == 1
    if size <= _FLAT_MAX_NUMEL:
        block = _flat_block(size)
        grid = _flat_grid(size, block)
        even = size % block == 0
        if dtype == torch.int64:
            if size <= _I64_LANES_MAX_NUMEL:
                # Keep the int32 value domain of the previous staging path:
                # wrap start/step into int32 so kernel arithmetic stays int32.
                start = ((start + 2**31) % 2**32) - 2**31
                if unit_step:
                    range_func_i64_lanes[(grid,)](
                        result.view(torch.int32),
                        start,
                        size,
                        BLOCK=block,
                        EVEN=even,
                        num_warps=2,
                    )
                else:
                    step = ((step + 2**31) % 2**32) - 2**31
                    range_step_i64_lanes[(grid,)](
                        result.view(torch.int32),
                        start,
                        step,
                        size,
                        BLOCK=block,
                        EVEN=even,
                        num_warps=2,
                    )
                return result
        elif dtype == torch.float64:
            staging = torch.empty((size,), device=device, dtype=torch.float32)
            if unit_step:
                range_func_flat[(grid,)](
                    staging,
                    float(start),
                    size,
                    BLOCK=block,
                    EVEN=even,
                    num_warps=2,
                )
            else:
                range_step_flat[(grid,)](
                    staging,
                    float(start),
                    float(step),
                    size,
                    BLOCK=block,
                    EVEN=even,
                    num_warps=2,
                )
            result.copy_(staging)
            return result
        else:
            if unit_step:
                range_func_flat[(grid,)](
                    result, start, size, BLOCK=block, EVEN=even, num_warps=2
                )
            else:
                range_step_flat[(grid,)](
                    result, start, step, size, BLOCK=block, EVEN=even, num_warps=2
                )
            return result

    # Out of the int32-safe flat domain: keep the seed per-block kernel (and
    # the staging route for the 8-byte replace containers).
    if dtype in (torch.int64, torch.float64):
        staging_dtype = torch.int32 if dtype == torch.int64 else torch.float32
        staging = torch.empty((size,), device=device, dtype=staging_dtype)
        range_func[(triton.cdiv(size, 2048),)](staging, start, step, size, 2048)
        result.copy_(staging)
    else:
        range_func[(triton.cdiv(size, 2048),)](result, start, step, size, 2048)
    return result


def range(start, end, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_ENFLAME RANGE")
    if dtype == torch.bfloat16:
        raise RuntimeError("torch.range does not support bfloat16 on CUDA")
    start = float(start)
    end = float(end)

    if dtype is None:
        # If any of start/end are float, use float64, otherwise int64
        if isinstance(start, float) or isinstance(end, float):
            dtype = torch.float64
        else:
            dtype = torch.int64

    if dtype in (torch.int32, torch.int64):
        start = int(start)
        end = int(end)
        size = end - start + 1
        step = 1
    else:
        # Float types
        size = math.ceil(end - start) + 1
        step = 1.0

    size = int(size)
    if size < 0:
        size = 0

    if pin_memory is None:
        pin_memory = False

    if device is None:
        device = runtime.device.name

    result = torch.empty((size,), device=device, dtype=dtype, pin_memory=pin_memory)
    if size == 0:
        return result
    return _launch_range(result, start, step, size, device, dtype)


def range_step(
    start, end, step=1, *, dtype=None, layout=None, device=None, pin_memory=None
):
    """GCU implementation of ``aten::range.step``.

    ``torch.range(start, end, step=1, ...)`` binds to the ``aten::range.step``
    schema; ``aten::range`` / ``range.default`` is a *separate* overload the
    public API never reaches, and ``_FULL_CONFIG`` only carries
    ``("range", range)`` - so without the registration at the bottom of this
    module ``torch.range(device=gcu)`` bypassed Gems entirely and landed on
    the torch_gcu native kernel (57.4 us @ 16M+1 fp16 vs 14.5 us here; the
    native int64 path SIGABRTs the process). Semantics mirror the ATen
    reference probed on CPU and on the native GCU kernel (2026-11-17):

    * ``dtype=None`` uses ``torch.get_default_dtype()`` (fp32 by default,
      regardless of the argument types - unlike the ``range.default``
      wrapper above);
    * length is ``floor((end - start) / step) + 1`` - ``end`` is included
      when it lands on the grid and never exceeded otherwise;
    * integer outputs truncate ``start`` / ``end`` / ``step`` towards zero
      and then use floor division; a fractional step that truncates to zero
      raises instead of hanging the device (native CPU loops forever there);
    * ``step`` equal to ``0`` or ``NaN`` -> ``RuntimeError("step must be
      nonzero")``;
    * non-finite ``start`` / ``end`` -> ``RuntimeError("unsupported range:
      <start> -> <end>")``;
    * ``(end - start)`` and ``step`` of opposite signs (with ``end !=
      start``) -> ``RuntimeError("upper bound and lower bound inconsistent
      with step sign")`` (CPU wording; the native GCU message says "larger
      bound").
    * bfloat16 is supported (CPU parity; the native GCU kernel raises
      NotImplementedError for it).
    """
    if layout not in (None, torch.strided):
        raise NotImplementedError("torch.range only supports strided layout")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if dtype not in _INT_RANGE_DTYPES + _FLOAT_RANGE_DTYPES:
        raise NotImplementedError(f'"range" not implemented for {dtype}')

    start_value = float(start)
    end_value = float(end)
    step_value = float(step)
    # Error precedence follows ATen (probed): step, then bound finiteness,
    # then the sign consistency check.
    if step_value == 0 or math.isnan(step_value):
        raise RuntimeError("step must be nonzero")
    if not (math.isfinite(start_value) and math.isfinite(end_value)):
        raise RuntimeError(f"unsupported range: {start} -> {end}")
    if end_value != start_value and (end_value - start_value) * step_value < 0:
        raise RuntimeError("upper bound and lower bound inconsistent with step sign")

    if dtype in _INT_RANGE_DTYPES:
        start_value = int(start_value)
        end_value = int(end_value)
        step_value = int(step_value)
        if step_value == 0:
            raise RuntimeError("step must be nonzero")
        size = (end_value - start_value) // step_value + 1
    else:
        size = math.floor((end_value - start_value) / step_value) + 1
    size = max(0, int(size))

    if pin_memory is None:
        pin_memory = False

    if device is None:
        device = runtime.device.name

    result = torch.empty((size,), device=device, dtype=dtype, pin_memory=pin_memory)
    if size == 0:
        return result
    return _launch_range(result, start_value, step_value, size, device, dtype)


# ``torch.range`` dispatches to ``aten::range.step``; register the GCU device
# key so explicit-device calls reach this implementation (kunlunxin precedent,
# same allow_override pattern as the sparse bridge in this backend). The
# PrivateUse1 key only fires once BackendSelect has resolved the call to the
# GCU device: ``device=None`` keeps the native CPU route, exactly like every
# other factory op under use_gems (benchmark/test_range.py passes no device,
# so both of its legs are unaffected by this registration).
_RANGE_STEP_LIBRARY = None
