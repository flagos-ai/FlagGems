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

import functools
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_max, get_dtype_min

logger = logging.getLogger(__name__)


_PPU_SM_FALLBACK = 64
_PPU_MAX_GRID = 65535
_PPU_TARGET_CTAS_PER_SM = 4

# PPU reductions are substantially cheaper when a CTA stays at four warps and
# the live tile is bounded.  These are starting points for a shape sweep, not
# permanent hardware constants.
_PPU_DIRECT_ROW_BLOCK = 2048
_PPU_ROW_BLOCK = 1024
_PPU_MAX_ROW_SPLITS = 1024
_PPU_COLUMN_BLOCK_K = 64
_PPU_COLUMN_BLOCK_N = 128
_PPU_MAX_COLUMN_SPLITS = 32
_PPU_ROW_FINALIZE_ROWS = 4
_PPU_STRIDED_BLOCK = 1024
_PPU_MAX_STRIDED_SPLITS = 4096
_PPU_COLUMN_GENERIC_OUTPUTS = 1 << 18
_PPU_ZERO_BLOCK = 128
_PPU_ZERO_BLOCK_LARGE = 256
_PPU_ZERO_LARGE_THRESHOLD = 1 << 16
_PPU_ZERO_WARPS = 4

_PPU_ARGMAX_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


@functools.lru_cache(maxsize=8)
def _cached_num_sms(device_index):
    try:
        props = torch_device_fn.get_device_properties(device_index)
        return max(1, int(props.multi_processor_count))
    except Exception:
        return _PPU_SM_FALLBACK


def _num_sms(device):
    return _cached_num_sms(device.index if device.index is not None else 0)


def _partial_dtype(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if dtype in (torch.bool, torch.uint8, torch.int8, torch.int16):
        return torch.int32
    return dtype


def _plan_row(rows, n, sms):
    if n <= _PPU_DIRECT_ROW_BLOCK:
        return "direct", triton.next_power_of_2(n), 1, 0

    block = _PPU_ROW_BLOCK
    target_splits = triton.cdiv(_PPU_TARGET_CTAS_PER_SM * sms, rows)
    splits = min(
        _PPU_MAX_ROW_SPLITS,
        triton.cdiv(n, block),
        max(1, target_splits),
    )
    chunk = triton.cdiv(n, block * splits) * block
    splits = triton.cdiv(n, chunk)
    return "split", block, splits, chunk


def _plan_column(m, n, k, sms):
    next_k = triton.next_power_of_2(k)
    next_n = triton.next_power_of_2(n)
    if n <= 32:
        block_n = min(64, next_n)
        block_k = min(32, next_k)
    elif n <= 512:
        block_n = min(64, next_n)
        block_k = min(128, next_k)
    else:
        block_n = min(_PPU_COLUMN_BLOCK_N, next_n)
        block_k = min(_PPU_COLUMN_BLOCK_K, next_k)
    column_tiles = m * triton.cdiv(k, block_k)
    target_splits = triton.cdiv(_PPU_TARGET_CTAS_PER_SM * sms, column_tiles)
    splits = min(
        _PPU_MAX_COLUMN_SPLITS,
        triton.cdiv(n, block_n),
        max(1, target_splits),
    )
    chunk = triton.cdiv(n, block_n * splits) * block_n
    splits = triton.cdiv(n, chunk)
    return block_n, block_k, splits, chunk


def _output_shape(inp, dim, keepdim):
    if dim is None:
        return [1] * inp.ndim if keepdim else []
    shape = list(inp.shape)
    if keepdim:
        shape[dim] = 1
    else:
        del shape[dim]
    return shape


def _generic_argmax(inp, dim, keepdim, dtype):
    from flag_gems.ops.argmax import argmax as generic_argmax

    return generic_argmax(inp, dim=dim, keepdim=keepdim, dtype=dtype)


@triton.jit
def _argmax_maximum_nan(a, b):
    return tl.maximum(a, b, tl.PropagateNan.ALL)


@triton.jit
def _argmax_reduce(values, indices, AXIS: tl.constexpr):
    if values.dtype.is_floating():
        maximum = tl.reduce(values, AXIS, _argmax_maximum_nan)
        expanded = tl.expand_dims(maximum, AXIS)
        equal = (values == expanded) | ((values != values) & (expanded != expanded))
    else:
        maximum = tl.max(values, AXIS)
        equal = values == tl.expand_dims(maximum, AXIS)
    index = tl.min(tl.where(equal, indices, get_dtype_max(indices.dtype)), AXIS)
    return maximum, index


@triton.jit
def _argmax_merge(best_values, best_indices, values, indices):
    replace = values > best_values
    tied = values == best_values
    if values.dtype.is_floating():
        replace = replace | ((values != values) & (best_values == best_values))
        tied = tied | ((values != values) & (best_values != best_values))
    replace = replace | (tied & (indices < best_indices))
    return (
        tl.where(replace, values, best_values),
        tl.where(replace, indices, best_indices),
    )


@libentry()
@triton.jit
def _argmax_zero_kernel(out, size, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    for start in tl.range(pid * BLOCK, size, num_programs * BLOCK):
        offsets = start + lane
        tl.store(out + offsets, 0, offsets < size)


@libentry()
@triton.jit
def _argmax_row_direct_kernel(
    inp,
    out,
    M,
    N,
    BLOCK: tl.constexpr,
    OFFSET64: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    value_dtype: tl.constexpr = inp.dtype.element_ty
    if value_dtype == tl.float16 or value_dtype == tl.bfloat16:
        value_dtype = tl.float32
    if (
        value_dtype == tl.int1
        or value_dtype == tl.uint8
        or value_dtype == tl.int8
        or value_dtype == tl.int16
    ):
        value_dtype = tl.int32
    minimum = get_dtype_min(value_dtype)
    sentinel: tl.constexpr = 0x7FFFFFFF

    for row in tl.range(pid, M, num_programs):
        if OFFSET64:
            base = row.to(tl.int64) * N.to(tl.int64)
        else:
            base = row * N
        mask = lane < N
        values = tl.load(inp + base + lane, mask, other=minimum).to(value_dtype)
        indices = tl.where(mask, lane, sentinel)
        _, index = _argmax_reduce(values, indices, 0)
        tl.store(out + row, index)


@libentry()
@triton.jit
def _argmax_row_partial_kernel(
    inp,
    partial_values,
    partial_indices,
    out,
    M,
    N,
    SPLITS,
    CHUNK,
    BLOCK: tl.constexpr,
    INDEX64: tl.constexpr,
    OFFSET64: tl.constexpr,
    HAS_SPLIT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    value_dtype: tl.constexpr = inp.dtype.element_ty
    if value_dtype == tl.float16 or value_dtype == tl.bfloat16:
        value_dtype = tl.float32
    if (
        value_dtype == tl.int1
        or value_dtype == tl.uint8
        or value_dtype == tl.int8
        or value_dtype == tl.int16
    ):
        value_dtype = tl.int32
    index_dtype: tl.constexpr = tl.int64 if INDEX64 else tl.int32
    sentinel: tl.constexpr = 0x7FFFFFFFFFFFFFFF if INDEX64 else 0x7FFFFFFF
    minimum = get_dtype_min(value_dtype)

    total_jobs = M * SPLITS
    for job in tl.range(pid, total_jobs, num_programs):
        row = job // SPLITS
        split_id = job % SPLITS
        start = split_id.to(tl.int64) * CHUNK if INDEX64 else split_id * CHUNK
        best_values = tl.full((BLOCK,), minimum, value_dtype)
        best_indices = tl.full((BLOCK,), sentinel, index_dtype)

        if OFFSET64:
            base = row.to(tl.int64) * N.to(tl.int64)
        else:
            base = row * N
        for offset in tl.range(0, CHUNK, BLOCK):
            col = start + offset + lane
            mask = col < N
            values = tl.load(inp + base + col, mask, other=minimum).to(value_dtype)
            indices = tl.where(mask, col, sentinel).to(index_dtype)
            best_values, best_indices = _argmax_merge(
                best_values, best_indices, values, indices
            )

        value, index = _argmax_reduce(best_values, best_indices, 0)
        if HAS_SPLIT:
            tl.store(partial_indices + row.to(tl.int64) * SPLITS + split_id, index)
            tl.store(partial_values + row.to(tl.int64) * SPLITS + split_id, value)
        else:
            tl.store(out + row, index)


@libentry()
@triton.jit
def _argmax_row_finalize_kernel(
    partial_values,
    partial_indices,
    out,
    M,
    SPLITS,
    BS: tl.constexpr,
    R: tl.constexpr,
    INDEX64: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    col = tl.arange(0, BS)
    sentinel: tl.constexpr = 0x7FFFFFFFFFFFFFFF if INDEX64 else 0x7FFFFFFF

    for row_start in tl.range(pid * R, M, num_programs * R):
        rows = row_start + tl.arange(0, R)
        mask = (rows[:, None] < M) & (col[None, :] < SPLITS)
        values = tl.load(
            partial_values + rows[:, None].to(tl.int64) * SPLITS + col[None, :],
            mask,
            other=get_dtype_min(partial_values.dtype.element_ty),
        )
        indices = tl.load(
            partial_indices + rows[:, None].to(tl.int64) * SPLITS + col[None, :],
            mask,
            other=sentinel,
        )
        _, index = _argmax_reduce(values, indices, 1)
        tl.store(out + rows, index, rows < M)


@libentry()
@triton.jit
def _argmax_column_kernel(
    inp,
    partial_values,
    partial_indices,
    out,
    M,
    N,
    K,
    SPLITS,
    CHUNK,
    BK: tl.constexpr,
    BN: tl.constexpr,
    INDEX64: tl.constexpr,
    OFFSET64: tl.constexpr,
    HAS_SPLIT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    lane_n = tl.arange(0, BN)
    lane_k = tl.arange(0, BK)
    value_dtype: tl.constexpr = inp.dtype.element_ty
    if value_dtype == tl.float16 or value_dtype == tl.bfloat16:
        value_dtype = tl.float32
    if (
        value_dtype == tl.int1
        or value_dtype == tl.uint8
        or value_dtype == tl.int8
        or value_dtype == tl.int16
    ):
        value_dtype = tl.int32
    index_dtype: tl.constexpr = tl.int64 if INDEX64 else tl.int32
    sentinel: tl.constexpr = 0x7FFFFFFFFFFFFFFF if INDEX64 else 0x7FFFFFFF
    minimum = get_dtype_min(value_dtype)

    k_tiles = tl.cdiv(K, BK)
    total_jobs = M * k_tiles * SPLITS
    for job in tl.range(pid, total_jobs, num_programs):
        split_id = job % SPLITS
        job_no_split = job // SPLITS
        k_tile = job_no_split % k_tiles
        m_id = job_no_split // k_tiles
        kk = k_tile * BK + lane_k
        k_mask = kk < K

        if INDEX64:
            start = split_id.to(tl.int64) * CHUNK
        else:
            start = split_id * CHUNK
        best_values = tl.full((BK,), minimum, value_dtype)
        best_indices = tl.full((BK,), sentinel, index_dtype)

        if OFFSET64:
            base = m_id.to(tl.int64) * N.to(tl.int64) * K.to(tl.int64)
        else:
            base = m_id * N * K
        for offset in tl.range(0, CHUNK, BN):
            nn = start + offset + lane_n
            n_mask = nn < N
            mask = n_mask[:, None] & k_mask[None, :]
            if OFFSET64:
                positions = base + nn[:, None].to(tl.int64) * K + kk[None, :]
            else:
                positions = base + nn[:, None] * K + kk[None, :]
            values = tl.load(inp + positions, mask, other=minimum).to(value_dtype)
            indices = tl.where(mask, nn[:, None], sentinel).to(index_dtype)
            tile_values, tile_indices = _argmax_reduce(values, indices, 0)
            best_values, best_indices = _argmax_merge(
                best_values, best_indices, tile_values, tile_indices
            )

        if HAS_SPLIT:
            workspace_offset = (m_id.to(tl.int64) * SPLITS + split_id) * K + kk
            tl.store(partial_values + workspace_offset, best_values, k_mask)
            tl.store(partial_indices + workspace_offset, best_indices, k_mask)
        elif OFFSET64:
            tl.store(out + m_id.to(tl.int64) * K + kk, best_indices, k_mask)
        else:
            tl.store(out + m_id * K + kk, best_indices, k_mask)


@libentry()
@triton.jit
def _argmax_column_finalize_kernel(
    partial_values,
    partial_indices,
    out,
    M,
    K,
    SPLITS,
    BS: tl.constexpr,
    BK: tl.constexpr,
    INDEX64: tl.constexpr,
    OFFSET64: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    lane_k = tl.arange(0, BK)
    lane_s = tl.arange(0, BS)
    sentinel: tl.constexpr = 0x7FFFFFFFFFFFFFFF if INDEX64 else 0x7FFFFFFF
    k_tiles = tl.cdiv(K, BK)
    total_jobs = M * k_tiles

    for job in tl.range(pid, total_jobs, num_programs):
        k_tile = job % k_tiles
        m_id = job // k_tiles
        kk = k_tile * BK + lane_k
        mask = (lane_s[:, None] < SPLITS) & (kk[None, :] < K)
        workspace_offset = (m_id.to(tl.int64) * SPLITS + lane_s[:, None]) * K + kk[
            None, :
        ]
        values = tl.load(
            partial_values + workspace_offset,
            mask,
            other=get_dtype_min(partial_values.dtype.element_ty),
        )
        indices = tl.load(partial_indices + workspace_offset, mask, other=sentinel)
        _, index = _argmax_reduce(values, indices, 0)
        if OFFSET64:
            tl.store(out + m_id.to(tl.int64) * K + kk, index, kk < K)
        else:
            tl.store(out + m_id * K + kk, index, kk < K)


@libentry()
@triton.jit
def _argmax_strided_global_kernel(
    inp,
    partial_values,
    partial_indices,
    out,
    SIZE,
    SPLITS,
    CHUNK,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
    INDEX64: tl.constexpr,
    HAS_SPLIT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    value_dtype: tl.constexpr = inp.dtype.element_ty
    if value_dtype == tl.float16 or value_dtype == tl.bfloat16:
        value_dtype = tl.float32
    if (
        value_dtype == tl.int1
        or value_dtype == tl.uint8
        or value_dtype == tl.int8
        or value_dtype == tl.int16
    ):
        value_dtype = tl.int32
    index_dtype: tl.constexpr = tl.int64 if INDEX64 else tl.int32
    sentinel: tl.constexpr = 0x7FFFFFFFFFFFFFFF if INDEX64 else 0x7FFFFFFF
    minimum = get_dtype_min(value_dtype)

    for split_id in tl.range(pid, SPLITS, num_programs):
        start = split_id.to(tl.int64) * CHUNK
        best_values = tl.full((BLOCK,), minimum, value_dtype)
        best_indices = tl.full((BLOCK,), sentinel, index_dtype)

        for offset in tl.range(0, CHUNK, BLOCK):
            scan = start + offset + lane
            remaining = scan
            physical = tl.zeros((BLOCK,), tl.int64)
            for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
                coordinate = remaining % SHAPE[axis]
                remaining = remaining // SHAPE[axis]
                physical += coordinate * STRIDES[axis]
            mask = scan < SIZE
            values = tl.load(inp + physical, mask, other=minimum).to(value_dtype)
            indices = tl.where(mask, scan, sentinel).to(index_dtype)
            best_values, best_indices = _argmax_merge(
                best_values, best_indices, values, indices
            )

        value, index = _argmax_reduce(best_values, best_indices, 0)
        if HAS_SPLIT:
            tl.store(partial_values + split_id, value)
            tl.store(partial_indices + split_id, index)
        else:
            tl.store(out, index)


def _argmax_global_strided(inp, out, sms):
    size = inp.numel()
    block = _PPU_STRIDED_BLOCK
    target_splits = min(_PPU_MAX_STRIDED_SPLITS, _PPU_TARGET_CTAS_PER_SM * sms)
    splits = min(target_splits, triton.cdiv(size, block))
    chunk = triton.cdiv(size, block * splits) * block
    splits = triton.cdiv(size, chunk)
    index64 = size > 0x7FFFFFFF

    if splits > 1:
        values = torch.empty(
            (1, splits), dtype=_partial_dtype(inp.dtype), device=inp.device
        )
        indices = torch.empty(
            (1, splits),
            dtype=torch.int64 if index64 else torch.int32,
            device=inp.device,
        )
    else:
        values = indices = out

    with torch_device_fn.device(inp.device):
        _argmax_strided_global_kernel[(min(splits, _PPU_MAX_GRID),)](
            inp,
            values,
            indices,
            out,
            size,
            splits,
            chunk,
            tuple(inp.shape),
            tuple(inp.stride()),
            BLOCK=block,
            INDEX64=index64,
            HAS_SPLIT=splits > 1,
            num_warps=4,
        )
        if splits > 1:
            _argmax_row_finalize_kernel[(1,)](
                values,
                indices,
                out,
                1,
                splits,
                BS=triton.next_power_of_2(splits),
                R=1,
                INDEX64=index64,
                num_warps=4,
            )


def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_THEAD ARGMAX")
    ndim = inp.ndim

    if dim is not None:
        original_dim = dim
        if ndim == 0:
            if original_dim not in (0, -1):
                raise IndexError(
                    "Dimension out of range (expected to be in range of [-1, 0], "
                    f"but got {original_dim})"
                )
            return torch.zeros([], dtype=torch.int64, device=inp.device)
        if original_dim < -ndim or original_dim >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-ndim}, {ndim - 1}], but got {original_dim})"
            )
        dim %= ndim

    if dim is None:
        if inp.numel() == 0:
            raise IndexError(
                "argmax(): Expected reduction dim to be specified for "
                "input.numel() == 0."
            )
        n = inp.numel()
        m, k = 1, 1
    else:
        n = inp.shape[dim]
        if n == 0:
            raise IndexError(
                f"argmax(): Expected reduction dim {dim} to have non-zero size."
            )
        m = math.prod(inp.shape[:dim]) if dim else 1
        k = math.prod(inp.shape[dim + 1 :]) if dim + 1 < ndim else 1

    out_shape = _output_shape(inp, dim, keepdim)
    if inp.numel() == 0:
        return torch.empty(out_shape, dtype=torch.int64, device=inp.device)
    if n == 1:
        out_numel = math.prod(out_shape)
        if (
            dim is not None
            and k > 1
            and _PPU_ZERO_BLOCK < out_numel <= _PPU_ZERO_LARGE_THRESHOLD
            and inp.dtype in _PPU_ARGMAX_DTYPES
        ):
            return _generic_argmax(inp, dim, keepdim, dtype)
        out = torch.empty(out_shape, dtype=torch.int64, device=inp.device)
        zero_block = (
            _PPU_ZERO_BLOCK
            if out_numel <= _PPU_ZERO_LARGE_THRESHOLD
            else _PPU_ZERO_BLOCK_LARGE
        )
        with torch_device_fn.device(inp.device):
            grid = (min(triton.cdiv(out_numel, zero_block), _PPU_MAX_GRID),)
            _argmax_zero_kernel[grid](
                out,
                out_numel,
                BLOCK=zero_block,
                num_warps=_PPU_ZERO_WARPS,
            )
        return out

    if inp.dtype not in _PPU_ARGMAX_DTYPES:
        return _generic_argmax(inp, dim, keepdim, dtype)
    if not inp.is_contiguous():
        if dim is not None:
            return _generic_argmax(inp, dim, keepdim, dtype)
        out = torch.empty(out_shape, dtype=torch.int64, device=inp.device)
        _argmax_global_strided(inp, out, _num_sms(inp.device))
        return out

    if dim is None:
        return _generic_argmax(inp, dim, keepdim, dtype)
    if k > 1:
        if (
            n < 512
            or m * k >= _PPU_COLUMN_GENERIC_OUTPUTS
            or (inp.dtype == torch.float32 and n <= 512)
        ):
            return _generic_argmax(inp, dim, keepdim, dtype)
    elif (
        (m <= 64 and n <= 64)
        or (n <= 4096 and m >= 2048)
        or (inp.dtype == torch.float32 and ((n >= 32768) or (n <= 4096 and m >= 1024)))
    ):
        return _generic_argmax(inp, dim, keepdim, dtype)

    out = torch.empty(out_shape, dtype=torch.int64, device=inp.device)
    sms = _num_sms(inp.device)
    index64 = n > 0x7FFFFFFF
    offset64 = inp.numel() > 0x7FFFFFFF

    with torch_device_fn.device(inp.device):
        if dim is None or k == 1:
            rows = m
            mode, block, splits, chunk = _plan_row(rows, n, sms)
            if mode == "direct":
                grid = (min(rows, _PPU_MAX_GRID),)
                _argmax_row_direct_kernel[grid](
                    inp,
                    out,
                    rows,
                    n,
                    BLOCK=block,
                    OFFSET64=offset64,
                    num_warps=4,
                )
            else:
                if splits > 1:
                    values = torch.empty(
                        (rows, splits),
                        dtype=_partial_dtype(inp.dtype),
                        device=inp.device,
                    )
                    indices = torch.empty(
                        (rows, splits),
                        dtype=torch.int64 if index64 else torch.int32,
                        device=inp.device,
                    )
                else:
                    values = indices = out
                jobs = rows * splits
                grid = (min(jobs, _PPU_MAX_GRID),)
                _argmax_row_partial_kernel[grid](
                    inp,
                    values,
                    indices,
                    out,
                    rows,
                    n,
                    splits,
                    chunk,
                    BLOCK=block,
                    INDEX64=index64,
                    OFFSET64=offset64,
                    HAS_SPLIT=splits > 1,
                    num_warps=4,
                )
                if splits > 1:
                    finish_rows = min(
                        _PPU_ROW_FINALIZE_ROWS, triton.next_power_of_2(rows)
                    )
                    finish_grid = (min(triton.cdiv(rows, finish_rows), _PPU_MAX_GRID),)
                    _argmax_row_finalize_kernel[finish_grid](
                        values,
                        indices,
                        out,
                        rows,
                        splits,
                        BS=triton.next_power_of_2(splits),
                        R=finish_rows,
                        INDEX64=index64,
                        num_warps=4,
                    )
        else:
            block_n, block_k, splits, chunk = _plan_column(m, n, k, sms)
            if splits > 1:
                values = torch.empty(
                    (m, splits, k),
                    dtype=_partial_dtype(inp.dtype),
                    device=inp.device,
                )
                indices = torch.empty(
                    (m, splits, k),
                    dtype=torch.int64 if index64 else torch.int32,
                    device=inp.device,
                )
            else:
                values = indices = out
            jobs = m * triton.cdiv(k, block_k) * splits
            grid = (min(jobs, _PPU_MAX_GRID),)
            _argmax_column_kernel[grid](
                inp,
                values,
                indices,
                out,
                m,
                n,
                k,
                splits,
                chunk,
                BK=block_k,
                BN=block_n,
                INDEX64=index64,
                OFFSET64=offset64,
                HAS_SPLIT=splits > 1,
                num_warps=4,
            )
            if splits > 1:
                final_jobs = m * triton.cdiv(k, block_k)
                final_grid = (min(final_jobs, _PPU_MAX_GRID),)
                _argmax_column_finalize_kernel[final_grid](
                    values,
                    indices,
                    out,
                    m,
                    k,
                    splits,
                    BS=triton.next_power_of_2(splits),
                    BK=block_k,
                    INDEX64=index64,
                    OFFSET64=offset64,
                    num_warps=4,
                )
    return out
