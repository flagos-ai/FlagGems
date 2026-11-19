# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Group reference-sized matrix multiplications without changing their accumulation policy."""

import logging
from threading import Lock

import torch
import triton

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import get_device_properties

from .mm import (
    _PARTIAL_KERNELS,
    _VECTOR_ALIGNMENT_BYTES,
    _dispatch_mm,
    mm,
    mm_kernel,
    mm_kernel_nt_rows,
    mm_kernel_splitk_reduce,
    mm_kernel_wide,
)

logger = logging.getLogger(__name__)
_plans = {}
_plan_lock = Lock()
_MAX_PLANS = 256
_NOT_CACHED = object()


def _remember(key, plan):
    # Another thread may have prepared the same immutable metadata in parallel.
    with _plan_lock:
        if key not in _plans and len(_plans) >= _MAX_PLANS:
            _plans.pop(next(iter(_plans)))
        return _plans.setdefault(key, plan)


def _launch_grid(m, n, splits, meta):
    rm, cn = (n, m) if meta.get("TRANSPOSE", False) else (m, n)
    return (splits * triton.cdiv(rm, meta["BM"]) * triton.cdiv(cn, meta["BN"]),)


def _plan(a, b, out):
    m = 64
    k, n = b.shape
    aligned = all(t.data_ptr() % _VECTOR_ALIGNMENT_BYTES == 0 for t in (a, b, out))
    key = (a.device.index, k, n, a.stride(0), aligned)
    cached = _plans.get(key, _NOT_CACHED)
    if cached is not _NOT_CACHED:
        return cached
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm the row-tile plan before Graph capture")
    prop = get_device_properties(a.device.index)
    tuned, splits, pack = _dispatch_mm(
        m,
        n,
        k,
        a.stride(),
        b.stride(),
        out.stride(),
        a.dtype,
        out.dtype,
        aligned,
        False,
        prop.multi_processor_count,
        prop.shared_memory_per_block,
        prop.L2_cache_size,
    )
    if pack or tuned not in (mm_kernel, mm_kernel_nt_rows, mm_kernel_wide):
        return _remember(key, None)
    target = (
        out[:m]
        if splits == 1
        else torch.empty((splits, m, n), dtype=torch.float32, device=a.device)
    )
    if splits > 1:
        tuned = _PARTIAL_KERNELS[tuned]
    _, meta = tuned[lambda cfg: _launch_grid(m, n, splits, cfg)](
        a[:m], b, target, m, n, k, *a.stride(), *b.stride(), n, 1, SPLIT_K=splits
    )
    meta = dict(meta)
    meta.pop("SPLIT_K", None)
    reduce_meta = None
    if splits > 1:
        _, red = mm_kernel_splitk_reduce[
            lambda cfg: (triton.cdiv(m * n, cfg["BLOCK"]),)
        ](target, out[:m], n, m * n, *out.stride(), SPLIT_K=splits)
        reduce_meta = dict(red)
        reduce_meta.pop("SPLIT_K", None)
    row_tile = meta["BN"] if meta.get("TRANSPOSE", False) else meta["BM"]
    plan = (
        (tuned.jit_function, splits, meta, reduce_meta) if m % row_tile == 0 else None
    )
    return _remember(key, plan)


def row_tiled_mm(a, b, *, rows_per_launch=256):
    """Return BF16 A @ B with the native 64-row mm_out reduction order.

    A is row-major (padded rows are supported) and B is a transposed contiguous
    matrix. rows_per_launch is a multiple of 64 and bounds the FP32 workspace.
    """
    logger.debug("GEMS_METAX ROW_TILED_MM")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible matrix shapes")
    if (
        not isinstance(rows_per_launch, int)
        or rows_per_launch < 64
        or rows_per_launch % 64
    ):
        raise ValueError("rows_per_launch must be a positive multiple of 64")
    m, k = a.shape
    n = b.shape[1]
    if not (
        a.is_cuda and b.device == a.device and a.dtype == b.dtype == torch.bfloat16
    ):
        raise ValueError("Expected two BF16 matrices on the same MetaX GPU")
    if not m or not n:
        return torch.empty((m, n), device=a.device, dtype=a.dtype)
    if not k:
        return torch.zeros((m, n), device=a.device, dtype=a.dtype)
    if a.stride(1) != 1 or a.stride(0) < k or b.stride(0) != 1 or b.stride(1) != k:
        raise ValueError("Expected row-major A and transposed contiguous B")
    if max(a.stride(0) * m, k * n, m * n) >= 2**31:
        raise ValueError("Matrix exceeds the qualified address range")
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    full = m // 64 * 64
    device = a.device
    if torch_device_fn.current_device() != device.index:
        with torch_device_fn.device(device):
            return _launch_row_tiles(a, b, out, rows_per_launch, full)
    return _launch_row_tiles(a, b, out, rows_per_launch, full)


def _launch_row_tiles(a, b, out, rows_per_launch, full):
    m, k = a.shape
    n = b.shape[1]
    # Native MM can select SYRK for an input and its own transpose.
    disjoint = a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
    plan = (
        _plan(a, b, out)
        if full >= 128 and n > 1 and rows_per_launch > 64 and disjoint
        else None
    )
    if plan is None:
        for start in range(0, m, 64):
            mm(a[start : start + 64], b, out=out[start : start + 64])
        return out
    kernel, splits, meta, reduce_meta = plan
    tmp = (
        torch.empty(
            (splits, min(rows_per_launch, full), n),
            device=a.device,
            dtype=torch.float32,
        )
        if splits > 1
        else None
    )
    for start in range(0, full, rows_per_launch):
        count = min(rows_per_launch, full - start)
        dst = out[start : start + count]
        kernel[_launch_grid(count, n, splits, meta)](
            a[start : start + count],
            b,
            tmp if splits > 1 else dst,
            count,
            n,
            k,
            *a.stride(),
            *b.stride(),
            n,
            1,
            SPLIT_K=splits,
            **meta,
        )
        if splits > 1:
            mm_kernel_splitk_reduce.jit_function[
                (triton.cdiv(count * n, reduce_meta["BLOCK"]),)
            ](tmp, dst, n, count * n, *dst.stride(), SPLIT_K=splits, **reduce_meta)
    if full < m:
        mm(a[full:], b, out=out[full:])
    return out
