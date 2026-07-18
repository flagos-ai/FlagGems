import logging
import math

import torch
import triton

from flag_gems.ops.nanmedian import (
    LARGE_FLOAT_REDUCTION_N,
    LONG_RADIX_REDUCTION_N,
    MAX_BLOCK_N,
    RADIX_BLOCK_N,
    NanMedian,
    _check_supported_dtype,
    _count_block_n,
    _empty_flat_value,
)
from flag_gems.ops.nanmedian import _nanmedian_dim_impl as _generic_nanmedian_dim_impl
from flag_gems.ops.nanmedian import (
    _normalize_dim,
    _radix_block_n,
    count_valid_kernel,
    nanmedian_ascend_byte_histogram_count_kernel,
    nanmedian_ascend_byte_histogram_find_index_kernel,
    nanmedian_ascend_byte_histogram_init_kernel,
    nanmedian_ascend_byte_histogram_select_kernel,
    nanmedian_ascend_byte_histogram_update_kernel,
    nanmedian_ascend_histogram_count_kernel,
    nanmedian_ascend_histogram_reduce_kernel,
    nanmedian_ascend_histogram_select_kernel,
    nanmedian_float_clean_count_kernel,
    nanmedian_float_sorted_gather_kernel,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress

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
_ASCEND_HISTOGRAM_BINS = 256
_ASCEND_MULTI_HISTOGRAM_MIN_N = 8192


def _nanmedian_float_sort_select(inp, M, N, values, indices):
    rows = inp.reshape(M, N)
    valid_counts = inp.new_empty((M,), dtype=torch.int32)

    if N <= LARGE_FLOAT_REDUCTION_N:
        sort_input = torch.empty_like(rows)
        block_n = min(triton.next_power_of_2(N), RADIX_BLOCK_N)
        with torch_device_fn.device(inp.device):
            nanmedian_float_clean_count_kernel[(M,)](
                rows,
                sort_input,
                valid_counts,
                N,
                block_n,
                num_warps=4 if block_n <= 512 else 8,
                num_stages=1,
            )
    else:
        sort_input = rows
        block_n = _count_block_n(rows, N)
        with torch_device_fn.device(inp.device):
            count_valid_kernel[(M,)](
                rows,
                valid_counts,
                M,
                N,
                block_n,
                False,
                num_warps=4 if block_n <= 512 else 8,
                num_stages=1,
            )

    sorted_values, sorted_indices = torch.sort(sort_input, dim=1)
    with torch_device_fn.device(inp.device):
        nanmedian_float_sorted_gather_kernel[(M,)](
            sorted_values,
            sorted_indices,
            valid_counts,
            values,
            indices,
            N,
            num_warps=1,
            num_stages=1,
        )


def _nanmedian_float_flat_sort(inp, out=None):
    flat = inp.reshape(-1).contiguous()
    values = inp.new_empty(()) if out is None else out
    indices = inp.new_empty((), dtype=torch.long)
    _nanmedian_float_sort_select(flat, 1, flat.numel(), values, indices)
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
    if inp.numel() == 0 or N <= MAX_BLOCK_N or inp.dtype not in _ASCEND_SELECT_DTYPES:
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
        _nanmedian_float_sort_select(inp, M, N, flat_values, flat_indices)
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
        result = _empty_flat_value(inp)
        if out is not None:
            out.copy_(result)
            return out
        return result

    if inp.dtype in _ASCEND_FLOAT_SELECT_DTYPES and inp.numel() > MAX_BLOCK_N:
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
