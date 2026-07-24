import logging
import math

import torch
import triton
import triton.language as tl
from packaging import version

from flag_gems.ops.nanmedian import (
    LONG_RADIX_REDUCTION_N,
    MAX_BLOCK_N,
    NanMedian,
    _check_supported_dtype,
    _empty_flat_value,
)
from flag_gems.ops.nanmedian import _nanmedian_dim_impl as _generic_nanmedian_dim_impl
from flag_gems.ops.nanmedian import (
    _normalize_dim,
    _radix_block_n,
    nanmedian_ascend_byte_histogram_count_kernel,
    nanmedian_ascend_byte_histogram_find_index_kernel,
    nanmedian_ascend_byte_histogram_init_kernel,
    nanmedian_ascend_byte_histogram_select_kernel,
    nanmedian_ascend_byte_histogram_update_kernel,
    nanmedian_ascend_histogram_count_kernel,
    nanmedian_ascend_histogram_reduce_kernel,
    nanmedian_ascend_histogram_select_kernel,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend.utils import CORE_NUM
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_ASCEND_FLOAT_SELECT_DTYPES = (torch.float16, torch.float32)
_ASCEND_HISTOGRAM_SELECT_DTYPES = (torch.int8, torch.uint8)
_ASCEND_BYTE_HISTOGRAM_SELECT_DTYPES = (torch.int16, torch.int32)
_ASCEND_INTEGER_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
)
_ASCEND_SELECT_DTYPES = _ASCEND_FLOAT_SELECT_DTYPES + _ASCEND_INTEGER_DTYPES
_ASCEND_SMALL_INTEGER_SORT_DTYPES = _ASCEND_INTEGER_DTYPES
_ASCEND_HISTOGRAM_BINS = 256
_ASCEND_MULTI_HISTOGRAM_MIN_N = 8192
_ASCEND_FLAT_SEARCH_FANOUT = 16
_ASCEND_FLAT_SEARCH_FANOUT_BITS = 4
_ASCEND_TOPK_FUSED_MAX_N = 4096
_ASCEND_TOPK_SHORT_N = 128
_TORCH_VERSION = version.parse(torch.__version__.split("+", 1)[0])
_TRITON_VERSION = version.parse(triton.__version__.split("+", 1)[0])
_ASCEND_910B2C = torch.npu.get_device_name().startswith("Ascend910B2C")
_ASCEND_CANN85_TOPK_SUPPORTED = (
    _ASCEND_910B2C
    and _TORCH_VERSION.release[:2] == (2, 9)
    and _TRITON_VERSION.release[:2] == (3, 2)
)
_ASCEND_CANN9_TOPK_SUPPORTED = (
    _ASCEND_910B2C
    and _TORCH_VERSION.release[:2] == (2, 10)
    and _TRITON_VERSION.release[:2] == (3, 5)
)


@libentry()
@triton.jit
def nanmedian_ascend_store_zero_kernel(out):
    tl.store(out, 0)


@libentry()
@triton.jit
def nanmedian_ascend_float_count_kernel(
    inp,
    valid_counts,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    for row_offset in tl.range(0, ROWS_PER_PROGRAM):
        row = pid * ROWS_PER_PROGRAM + row_offset
        row_mask = row < M
        count = tl.full((), 0, dtype=tl.int32)

        for col_start in tl.range(0, N, BLOCK_N, loop_unroll_factor=1):
            cols = col_start + offsets
            mask = row_mask & (cols < N)
            values = tl.load(
                inp + row * N + cols,
                mask=mask,
                other=float("nan"),
            )
            count += tl.sum((mask & (values == values)).to(tl.int32), axis=0)

        tl.store(valid_counts + row, count, mask=row_mask)


@libentry()
@triton.jit
def nanmedian_ascend_float_topk_gather_kernel(
    top_values,
    top_indices,
    valid_counts,
    out_values,
    out_indices,
    M: tl.constexpr,
    K: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)

    for row_offset in tl.range(0, ROWS_PER_PROGRAM):
        row = pid * ROWS_PER_PROGRAM + row_offset
        row_mask = row < M
        count = tl.load(valid_counts + row, mask=row_mask, other=0)
        has_valid = row_mask & (count > 0)
        rank = tl.where(has_valid, (count - 1) // 2, 0)
        value = tl.load(
            top_values + row * K + rank,
            mask=has_valid,
            other=float("nan"),
        )
        index = tl.load(top_indices + row * K + rank, mask=has_valid, other=0)
        tl.store(
            out_values + row,
            tl.where(has_valid, value, float("nan")),
            mask=row_mask,
        )
        tl.store(
            out_indices + row,
            tl.where(has_valid, index, 0),
            mask=row_mask,
        )


@libentry()
@triton.jit
def nanmedian_ascend_float_fused_topk_gather_kernel(
    inp,
    top_values,
    top_indices,
    out_values,
    out_indices,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    for row_offset in tl.range(0, ROWS_PER_PROGRAM):
        row = pid * ROWS_PER_PROGRAM + row_offset
        mask = (row < M) & (offsets < N)
        values = tl.load(inp + row * N + offsets, mask=mask, other=float("nan"))
        valid_count = tl.sum((mask & (values == values)).to(tl.int32), axis=0)
        has_valid = (row < M) & (valid_count > 0)
        rank = tl.where(has_valid, (valid_count - 1) // 2, 0)
        result = tl.load(
            top_values + row * K + rank,
            mask=has_valid,
            other=float("nan"),
        )
        result_idx = tl.load(top_indices + row * K + rank, mask=has_valid, other=0)
        result = tl.where(has_valid, result, float("nan"))
        result_idx = tl.where(has_valid, result_idx, 0)
        tl.store(out_values + row, result, mask=row < M)
        tl.store(out_indices + row, result_idx, mask=row < M)


@libentry()
@triton.jit
def nanmedian_ascend_float_sorted_search_gather_kernel(
    sorted_values,
    sorted_indices,
    out_values,
    out_indices,
    M: tl.constexpr,
    N: tl.constexpr,
    SEARCH_FANOUT: tl.constexpr,
    SEARCH_ROUNDS: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tle.program_id(0)
    probe_ids = tl.arange(0, SEARCH_FANOUT)

    for row_offset in tl.range(0, ROWS_PER_PROGRAM):
        row = pid * ROWS_PER_PROGRAM + row_offset
        row_mask = row < M
        lo = tl.full((), 0, dtype=tl.int32)
        hi = tl.full((), N, dtype=tl.int32)

        for _ in tl.static_range(0, SEARCH_ROUNDS):
            span = hi - lo
            block_n = tl.maximum((span + SEARCH_FANOUT - 1) // SEARCH_FANOUT, 1)
            probe_mask = row_mask & ((probe_ids * block_n) < span)
            probe_offsets = tl.minimum(lo + (probe_ids + 1) * block_n, hi) - 1
            probe_offsets = tl.where(probe_mask, probe_offsets, 0)
            probe_values = tl.load(
                sorted_values + row * N + probe_offsets,
                mask=probe_mask,
                other=float("nan"),
            )
            valid_blocks = tl.sum(
                (probe_mask & (probe_values == probe_values)).to(tl.int32),
                axis=0,
            )
            next_lo = tl.minimum(lo + valid_blocks * block_n, hi)
            next_hi = tl.minimum(next_lo + block_n, hi)
            lo = tl.where(span > 0, next_lo, lo)
            hi = tl.where(span > 0, next_hi, hi)

        active = row_mask & (lo < hi)
        val = tl.load(
            sorted_values + row * N + lo,
            mask=active,
            other=float("nan"),
        )
        lo += (active & (val == val)).to(tl.int32)

        has_valid = row_mask & (lo != 0)
        rank = tl.where(has_valid, (lo - 1) // 2, 0)
        result_val = tl.load(
            sorted_values + row * N + rank, mask=has_valid, other=float("nan")
        )
        result_idx = tl.load(sorted_indices + row * N + rank, mask=has_valid, other=0)
        result_val = tl.where(has_valid, result_val, float("nan"))
        result_idx = tl.where(has_valid, result_idx, 0)
        tl.store(out_values + row, result_val, mask=row_mask)
        tl.store(out_indices + row, result_idx, mask=row_mask)


@libentry()
@triton.jit
def nanmedian_ascend_float_flat_sorted_search_kernel(
    sorted_values,
    out,
    N: tl.constexpr,
    SEARCH_FANOUT: tl.constexpr,
    SEARCH_ROUNDS: tl.constexpr,
):
    probe_ids = tl.arange(0, SEARCH_FANOUT)
    lo = tl.full((), 0, dtype=tl.int32)
    hi = tl.full((), N, dtype=tl.int32)

    for _ in tl.static_range(0, SEARCH_ROUNDS):
        span = hi - lo
        block_n = tl.maximum((span + SEARCH_FANOUT - 1) // SEARCH_FANOUT, 1)
        probe_mask = (probe_ids * block_n) < span
        probe_offsets = tl.minimum(lo + (probe_ids + 1) * block_n, hi) - 1
        probe_offsets = tl.where(probe_mask, probe_offsets, 0)
        probe_values = tl.load(
            sorted_values + probe_offsets,
            mask=probe_mask,
            other=float("nan"),
        )
        valid_blocks = tl.sum(
            (probe_mask & (probe_values == probe_values)).to(tl.int32), axis=0
        )
        next_lo = tl.minimum(lo + valid_blocks * block_n, hi)
        next_hi = tl.minimum(next_lo + block_n, hi)
        lo = tl.where(span > 0, next_lo, lo)
        hi = tl.where(span > 0, next_hi, hi)

    active = lo < hi
    val = tl.load(sorted_values + lo, mask=active, other=float("nan"))
    lo += (active & (val == val)).to(tl.int32)

    has_valid = lo != 0
    rank = tl.where(has_valid, (lo - 1) // 2, 0)
    result = tl.load(sorted_values + rank, mask=has_valid, other=float("nan"))
    tl.store(out, tl.where(has_valid, result, float("nan")))


def _nanmedian_float_sort_select(inp, M, N, values, indices):
    rows = inp.reshape(M, N)
    sorted_values, sorted_indices = torch.sort(rows, dim=1)
    search_rounds = (
        (N - 1).bit_length() + _ASCEND_FLAT_SEARCH_FANOUT_BITS - 1
    ) // _ASCEND_FLAT_SEARCH_FANOUT_BITS
    search_grid = min(M, CORE_NUM)
    rows_per_program = triton.cdiv(M, search_grid)
    with torch_device_fn.device(inp.device):
        nanmedian_ascend_float_sorted_search_gather_kernel[(search_grid,)](
            sorted_values,
            sorted_indices,
            values,
            indices,
            M,
            N,
            _ASCEND_FLAT_SEARCH_FANOUT,
            search_rounds,
            rows_per_program,
            num_warps=1,
            num_stages=1,
        )


def _nanmedian_float_topk_supported(N):
    return _ASCEND_CANN9_TOPK_SUPPORTED or (
        _ASCEND_CANN85_TOPK_SUPPORTED and N <= _ASCEND_TOPK_FUSED_MAX_N
    )


def _nanmedian_float_topk_select(inp, M, N, values, indices):
    rows = inp.reshape(M, N)
    k = (N + 1) // 2

    if _ASCEND_CANN9_TOPK_SUPPORTED and N > _ASCEND_TOPK_SHORT_N:
        grid = min(M, CORE_NUM)
        rows_per_program = triton.cdiv(M, grid)
        valid_counts = torch.empty_strided(
            (M,), (1,), dtype=torch.int32, device=inp.device
        )
        with torch_device_fn.device(inp.device):
            nanmedian_ascend_float_count_kernel[(grid,)](
                rows,
                valid_counts,
                M,
                N,
                1024,
                rows_per_program,
                num_warps=1,
                num_stages=1,
            )
        top_values, top_indices = torch.topk(
            rows, k, dim=-1, largest=False, sorted=True
        )
        with torch_device_fn.device(inp.device):
            nanmedian_ascend_float_topk_gather_kernel[(grid,)](
                top_values,
                top_indices,
                valid_counts,
                values,
                indices,
                M,
                k,
                rows_per_program,
                num_warps=1,
                num_stages=1,
            )
        return

    top_values, top_indices = torch.topk(rows, k, dim=-1, largest=False, sorted=True)
    if _ASCEND_CANN85_TOPK_SUPPORTED:
        grid = M
        rows_per_program = 1
        num_warps = 4
    else:
        grid = min(M, CORE_NUM)
        rows_per_program = triton.cdiv(M, grid)
        num_warps = 1
    with torch_device_fn.device(inp.device):
        nanmedian_ascend_float_fused_topk_gather_kernel[(grid,)](
            rows,
            top_values,
            top_indices,
            values,
            indices,
            M,
            N,
            k,
            triton.next_power_of_2(N),
            rows_per_program,
            num_warps=num_warps,
            num_stages=1,
        )


def _nanmedian_float_flat_sort(inp, out=None):
    flat = inp.reshape(-1)
    values = inp.new_empty(()) if out is None else out
    search_rounds = (
        (flat.numel() - 1).bit_length() + _ASCEND_FLAT_SEARCH_FANOUT_BITS - 1
    ) // _ASCEND_FLAT_SEARCH_FANOUT_BITS
    sorted_values, _ = torch.sort(flat.reshape(1, flat.numel()), dim=1)
    with torch_device_fn.device(inp.device):
        nanmedian_ascend_float_flat_sorted_search_kernel[(1,)](
            sorted_values,
            values,
            flat.numel(),
            _ASCEND_FLAT_SEARCH_FANOUT,
            search_rounds,
            num_warps=4,
            num_stages=1,
        )
    return values


def _nanmedian_float_flat_topk(inp, out=None):
    flat = inp.reshape(-1)
    values = (
        torch.empty_strided((), (), dtype=inp.dtype, device=inp.device)
        if out is None
        else out
    )
    indices = torch.empty_strided((), (), dtype=torch.long, device=inp.device)
    _nanmedian_float_topk_select(flat, 1, flat.numel(), values, indices)
    return values


def _nanmedian_histogram_select(inp, M, N, values, indices):
    block_n = _radix_block_n(inp, N)
    num_warps = 4 if block_n <= 512 else 8

    if N >= _ASCEND_MULTI_HISTOGRAM_MIN_N:
        num_chunks = triton.cdiv(N, block_n)
        partial_counts = inp.new_empty(
            (M, num_chunks, _ASCEND_HISTOGRAM_BINS), dtype=torch.int32
        )
        with torch_device_fn.device(inp.device):
            nanmedian_ascend_histogram_count_kernel[(M, num_chunks)](
                inp,
                partial_counts,
                M,
                N,
                block_n,
                num_chunks,
                _ASCEND_HISTOGRAM_BINS,
                num_warps=num_warps,
                num_stages=1,
            )
            nanmedian_ascend_histogram_reduce_kernel[(M,)](
                inp,
                partial_counts,
                values,
                indices,
                M,
                N,
                block_n,
                num_chunks,
                _ASCEND_HISTOGRAM_BINS,
                num_warps=num_warps,
                num_stages=1,
            )
        return

    with torch_device_fn.device(inp.device):
        nanmedian_ascend_histogram_select_kernel[(M,)](
            inp,
            values,
            indices,
            M,
            N,
            block_n,
            _ASCEND_HISTOGRAM_BINS,
            num_warps=num_warps,
            num_stages=1,
        )


def _nanmedian_byte_histogram_select(inp, M, N, values, indices):
    block_n = _radix_block_n(inp, N)
    num_warps = 4 if block_n <= 512 else 8

    if N >= _ASCEND_MULTI_HISTOGRAM_MIN_N:
        num_chunks = triton.cdiv(N, block_n)
        partial_counts = inp.new_empty(
            (M, num_chunks, _ASCEND_HISTOGRAM_BINS), dtype=torch.int32
        )
        state = inp.new_empty((M, 3), dtype=torch.int64)
        nbits = inp.element_size() * 8
        with torch_device_fn.device(inp.device):
            nanmedian_ascend_byte_histogram_init_kernel[(M,)](
                state,
                M,
                N,
                num_warps=1,
                num_stages=1,
            )
            for digit_pos in range(nbits - 8, -1, -8):
                nanmedian_ascend_byte_histogram_count_kernel[(M, num_chunks)](
                    inp,
                    state,
                    partial_counts,
                    M,
                    N,
                    block_n,
                    num_chunks,
                    _ASCEND_HISTOGRAM_BINS,
                    digit_pos,
                    num_warps=num_warps,
                    num_stages=1,
                )
                nanmedian_ascend_byte_histogram_update_kernel[(M,)](
                    inp,
                    partial_counts,
                    state,
                    M,
                    num_chunks,
                    _ASCEND_HISTOGRAM_BINS,
                    digit_pos,
                    num_warps=num_warps,
                    num_stages=1,
                )
            nanmedian_ascend_byte_histogram_find_index_kernel[(M,)](
                inp,
                state,
                values,
                indices,
                M,
                N,
                block_n,
                num_warps=num_warps,
                num_stages=1,
            )
        return

    with torch_device_fn.device(inp.device):
        nanmedian_ascend_byte_histogram_select_kernel[(M,)](
            inp,
            values,
            indices,
            M,
            N,
            block_n,
            _ASCEND_HISTOGRAM_BINS,
            num_warps=num_warps,
            num_stages=1,
        )


def _nanmedian_integer_sort_select(inp, M, N, values, indices):
    rows = inp.reshape(M, N)
    sorted_values, sorted_indices = torch.sort(rows, dim=1)
    kth = (N + 1) // 2 - 1
    values.copy_(sorted_values[:, kth])
    indices.copy_(sorted_indices[:, kth])


def _nanmedian_dim_impl(inp, dim, keepdim, out=None):
    dim = _normalize_dim(dim, inp.ndim)
    if inp.ndim == 0:
        return _generic_nanmedian_dim_impl(inp, dim, keepdim, out=out)

    N = inp.shape[dim]
    if (
        inp.numel() == 0
        or inp.dtype not in _ASCEND_SELECT_DTYPES
        or (
            N <= MAX_BLOCK_N
            and inp.dtype not in _ASCEND_FLOAT_SELECT_DTYPES
            and inp.dtype not in _ASCEND_SMALL_INTEGER_SORT_DTYPES
        )
    ):
        return _generic_nanmedian_dim_impl(inp, dim, keepdim, out=out)

    shape = list(inp.shape)
    out_shape = shape[:dim] + shape[dim + 1 :]
    M = math.prod(out_shape)
    keepdim_shape = shape.copy()
    keepdim_shape[dim] = 1
    output_shape = keepdim_shape if keepdim else out_shape
    compute_shape = output_shape if out is not None else keepdim_shape

    if out is None:
        values = inp.new_empty(compute_shape)
        indices = inp.new_empty(compute_shape, dtype=torch.long)
    else:
        values, indices = out

    inp = dim_compress(inp, dim)
    values_contiguous = values.is_contiguous()
    indices_contiguous = indices.is_contiguous()
    flat_values = (
        values.reshape(M)
        if values_contiguous
        else values.new_empty((M,), dtype=values.dtype)
    )
    flat_indices = (
        indices.reshape(M)
        if indices_contiguous
        else indices.new_empty((M,), dtype=indices.dtype)
    )

    if inp.dtype in _ASCEND_FLOAT_SELECT_DTYPES:
        if _nanmedian_float_topk_supported(N):
            _nanmedian_float_topk_select(inp, M, N, flat_values, flat_indices)
        else:
            _nanmedian_float_sort_select(inp, M, N, flat_values, flat_indices)
    elif N <= MAX_BLOCK_N and inp.dtype in _ASCEND_SMALL_INTEGER_SORT_DTYPES:
        _nanmedian_integer_sort_select(inp, M, N, flat_values, flat_indices)
    elif inp.dtype in _ASCEND_HISTOGRAM_SELECT_DTYPES and N <= LONG_RADIX_REDUCTION_N:
        _nanmedian_histogram_select(inp, M, N, flat_values, flat_indices)
    elif (
        inp.dtype in _ASCEND_BYTE_HISTOGRAM_SELECT_DTYPES
        and N <= LONG_RADIX_REDUCTION_N
    ):
        _nanmedian_byte_histogram_select(inp, M, N, flat_values, flat_indices)
    else:
        _nanmedian_integer_sort_select(inp, M, N, flat_values, flat_indices)

    if not values_contiguous:
        values.copy_(flat_values.reshape(values.shape))
    if not indices_contiguous:
        indices.copy_(flat_indices.reshape(indices.shape))

    if out is None and not keepdim:
        values = torch.squeeze(values, dim)
        indices = torch.squeeze(indices, dim)
    return NanMedian(values=values, indices=indices)


def _nanmedian_flat_impl(inp, out=None):
    if inp.numel() == 0:
        if not torch.is_floating_point(inp):
            result = inp.new_empty(()) if out is None else out
            with torch_device_fn.device(inp.device):
                nanmedian_ascend_store_zero_kernel[(1,)](
                    result,
                    num_warps=1,
                    num_stages=1,
                )
            return result
        result = _empty_flat_value(inp)
        if out is not None:
            out.copy_(result)
            return out
        return result

    if inp.numel() == 1:
        scalar = inp.reshape(())
        if out is not None:
            out.copy_(scalar)
            return out
        return scalar.clone()

    if inp.dtype in _ASCEND_FLOAT_SELECT_DTYPES:
        if _nanmedian_float_topk_supported(inp.numel()):
            return _nanmedian_float_flat_topk(inp, out=out)
        return _nanmedian_float_flat_sort(inp, out=out)

    flat = inp.reshape(-1)
    if out is None:
        return _nanmedian_dim_impl(flat, 0, False).values

    indices = inp.new_empty((), dtype=torch.long)
    _nanmedian_dim_impl(flat, 0, False, out=(out, indices))
    return out


def nanmedian(inp):
    logger.debug("GEMS_ASCEND NANMEDIAN")
    _check_supported_dtype(inp)
    return _nanmedian_flat_impl(inp)


def nanmedian_out(inp, *, out):
    logger.debug("GEMS_ASCEND NANMEDIAN OUT")
    _check_supported_dtype(inp)
    return _nanmedian_flat_impl(inp, out=out)


def nanmedian_dim(inp, dim=-1, keepdim=False):
    logger.debug("GEMS_ASCEND NANMEDIAN DIM")
    _check_supported_dtype(inp)
    return _nanmedian_dim_impl(inp, dim, keepdim)


def nanmedian_dim_values(inp, dim=-1, keepdim=False, *, values, indices):
    logger.debug("GEMS_ASCEND NANMEDIAN DIM VALUES")
    return _nanmedian_dim_impl(inp, dim, keepdim, out=(values, indices))
