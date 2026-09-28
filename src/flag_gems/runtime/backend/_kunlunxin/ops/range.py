# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _range_kernel(out_ptr, start, size, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + offsets, offsets + start, mask=offsets < size)


def range(start, end, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_KUNLUNXIN RANGE")
    if layout not in (None, torch.strided):
        raise RuntimeError("torch.range only supports strided layout")
    if pin_memory:
        raise RuntimeError("torch.range does not support pinned memory on XPU")
    if dtype is None:
        dtype = (
            torch.float64
            if any(isinstance(value, float) for value in (start, end))
            else torch.int64
        )
    if dtype == torch.bfloat16:
        raise RuntimeError("torch.range does not support bfloat16 on XPU")
    if dtype not in (
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
    ):
        raise RuntimeError(f"torch.range does not support dtype {dtype}")

    integer_output = dtype in (torch.int32, torch.int64)
    start_value = int(start) if integer_output else float(start)
    end_value = int(end) if integer_output else float(end)
    size = (
        end_value - start_value + 1
        if integer_output
        else math.ceil(end_value - start_value) + 1
    )
    size = max(0, int(size))
    output_device = device if device is not None else runtime.device.name
    out = torch.empty(size, dtype=dtype, device=output_device)
    if size == 0:
        return out
    with torch_device_fn.device(out.device):
        _range_kernel[(triton.cdiv(size, 256),)](out, start_value, size, BLOCK_SIZE=256)
    return out


_range_backend_select_lib = torch.library.Library("aten", "IMPL", "BackendSelect")
_range_backend_select_lib.impl("range", range, allow_override=True)


# ---------------------------------------------------------------------------
# aten::range.step -- this is the overload ``torch.range`` actually dispatches to
# ---------------------------------------------------------------------------
# ``torch.range(start, end, step=1, ...)`` is bound to the ``aten::range.step``
# schema; ``aten::range`` / ``range.default`` (handled above) is a *separate*
# overload that ``torch.range`` never reaches.  ``_FULL_CONFIG`` only carries
# ``("range", range)``, so before this override ``torch.range(...)`` bypassed
# Gems entirely: the native BackendSelect kernel resolved the device and
# redispatched to the native CompositeExplicitAutograd ``range.step`` kernel.
#
# The *device* dispatch key is reachable for this overload (verified with an
# override spy on ``CUDA``), so a device-key registration is sufficient here and
# no BackendSelect hijack is needed.  As with the ``adaptive_max_pool2d`` /
# ``adaptive_max_pool3d`` patches in ``_kunlunxin/ops/__init__.py`` the
# registration is process global; it only takes effect for calls that explicitly
# select the XPU device, while ``device=None`` keeps resolving to the native CPU
# path (which is what ``benchmark/test_range.py`` exercises -- see
# ``harness/solution/range/README.md``).

_INT_OUT_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
_FLOAT_OUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_SUPPORTED_OUT_DTYPES = _INT_OUT_DTYPES + _FLOAT_OUT_DTYPES


@libentry()
@triton.jit
def _range_step_float_kernel(out_ptr, start, step, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)
    idx = offset + cols
    # ``cols * step`` (rather than ``idx * step``) keeps the fp32 multiply on a
    # small operand and mirrors the proven sibling kernel in
    # ``_kunlunxin/ops/arange.py``.
    base = (offset * step + start).to(tl.float32)
    value = cols.to(tl.float32) * step + base
    tl.store(out_ptr + idx, value, mask=idx < size)


@libentry()
@triton.jit
def _range_step_int_kernel(
    out_ptr, start, step, size, WIDE: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)
    idx = offset + cols
    if WIDE:
        # int64 output: ``cols``/``offset`` are i32 on this backend, so the
        # arithmetic has to be widened *before* ``start`` is added -- otherwise
        # values crossing 2**31 wrap silently (e.g. range(2**31-3, 2**31+2,
        # int64) gave -2147483646 instead of 2147483650).  Only the value is
        # widened; keeping ``idx``/the mask in i32 avoids the int64 comparison
        # lowering that crashes this backend.
        value = (cols.to(tl.int64) + offset) * step + start
    else:
        value = cols * step + offset * step + start
    tl.store(out_ptr + idx, value, mask=idx < size)


def _range_step_block(size):
    # Same size-based (BLOCK_SIZE, num_warps) heuristic as the sibling
    # ``_kunlunxin/ops/arange.py``; every block size is a multiple of 64 so the
    # ``tl.arange`` widening on this backend is a no-op.
    if size <= 1024:
        return 256, 2
    if size <= 8192:
        return 1024, 4
    if size <= 65536:
        return 4096, 8
    return 16384, 8


def range_step(
    start, end, step=1, *, dtype=None, layout=None, device=None, pin_memory=None
):
    """XPU implementation of ``aten::range.step`` (i.e. of ``torch.range``).

    The semantics reproduce the ATen reference measured on this stack:

    * default ``dtype`` is ``torch.float32`` (regardless of the argument types);
    * length is ``floor((end - start) / step) + 1`` -- ``end`` is included when
      it lands on the grid and is never exceeded otherwise;
    * integer outputs truncate ``start`` / ``end`` / ``step`` towards zero and
      then use ``floor((end - start) / step) + 1``;
    * ``step`` equal to ``0`` or ``NaN`` -> ``RuntimeError("step must be nonzero")``;
    * non-finite ``start`` / ``end`` -> ``RuntimeError("unsupported range: ...")``;
    * ``(end - start)`` and ``step`` of opposite signs (with ``end != start``)
      -> ``RuntimeError("upper bound and lower bound inconsistent with step sign")``.
    """
    logger.debug("GEMS_KUNLUNXIN RANGE_STEP")
    if layout not in (None, torch.strided):
        raise NotImplementedError("torch.range only supports strided layout")
    if pin_memory:
        raise RuntimeError("torch.range does not support pinned memory on XPU")
    if dtype is None:
        dtype = torch.float32
    if dtype not in _SUPPORTED_OUT_DTYPES:
        raise NotImplementedError(f'"range" not implemented for {dtype}')

    start_value = float(start)
    end_value = float(end)
    step_value = float(step)
    # Error precedence follows ATen (probed): step, then bound finiteness, then
    # the sign consistency check.
    if step_value == 0 or math.isnan(step_value):
        raise RuntimeError("step must be nonzero")
    if not (math.isfinite(start_value) and math.isfinite(end_value)):
        raise RuntimeError(f"unsupported range: {start} -> {end}")
    if end_value != start_value and (end_value - start_value) * step_value < 0:
        raise RuntimeError("upper bound and lower bound inconsistent with step sign")

    integer_output = dtype in _INT_OUT_DTYPES
    if integer_output:
        start_value = int(start)
        end_value = int(end)
        step_value = int(step)
        if step_value == 0:
            # ATen truncates the step for integer outputs but only validates the
            # floating point value, so e.g. ``step=0.5`` makes the native loop
            # run forever (reproduced on the CPU reference).  Raising keeps the
            # failure local instead of hanging the device.
            raise RuntimeError("step must be nonzero")
        size = (end_value - start_value) // step_value + 1
    else:
        size = math.floor((end_value - start_value) / step_value) + 1
    size = max(0, int(size))

    output_device = device if device is not None else runtime.device.name
    out = torch.empty((size,), dtype=dtype, device=output_device)
    if size == 0:
        return out

    block_size, num_warps = _range_step_block(size)
    grid = (triton.cdiv(size, block_size),)
    with torch_device_fn.device(out.device):
        if integer_output:
            _range_step_int_kernel[grid](
                out,
                start_value,
                step_value,
                size,
                WIDE=dtype == torch.int64,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
        else:
            _range_step_float_kernel[grid](
                out,
                start_value,
                step_value,
                size,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
    return out


_range_step_lib = torch.library.Library("aten", "IMPL")
_range_step_lib.impl("range.step", range_step, "CUDA", allow_override=True)
