import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.nanmedian import (
    INT32_MAX,
    MAX_BLOCK_N,
    RADIX_SELECT_DTYPES,
    NanMedian,
    _check_supported_dtype,
    _empty_flat_value,
    _nanmedian_cuda_flat_radix_select,
    _normalize_dim,
    nanmedian_radix_select_kernel,
    nanmedian_select_kernel,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .sort import sort as mthreads_sort

logger = logging.getLogger(__name__)

_SORT_BLOCK_N = 1024


@libentry()
@triton.jit
def _clean_nan_and_count_kernel(
    inp,
    cleaned,
    valid_counts,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    cols = block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N
    offsets = row * N + cols
    values = tl.load(inp + offsets, mask=mask, other=float("nan"))
    valid = mask & (values == values)
    cleaned_values = tl.where(valid, values, float("inf"))
    tl.store(cleaned + offsets, cleaned_values, mask=mask)
    count = tl.sum(valid.to(tl.int32), axis=0)
    tl.atomic_add(valid_counts + row, count, sem="relaxed")


@libentry()
@triton.jit
def _select_sorted_median_kernel(
    inp,
    sorted_values,
    valid_counts,
    out_values,
    out_indices,
    N: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if IS_FLOAT:
        valid_count = tl.load(valid_counts + row)
        rank = tl.where(valid_count > 0, (valid_count - 1) // 2, 0)
    else:
        valid_count = N
        rank = (N - 1) // 2

    selected = tl.load(
        sorted_values + row * N + rank,
        mask=valid_count > 0,
        other=float("nan") if IS_FLOAT else 0,
    )
    first_index = tl.full((), N, dtype=tl.int64)
    offsets = tl.arange(0, BLOCK_N)
    for start in range(0, N, BLOCK_N):
        cols = start + offsets
        mask = cols < N
        values = tl.load(inp + row * N + cols, mask=mask, other=0)
        matches = mask & (values == selected)
        if IS_FLOAT:
            matches &= values == values
        local_index = tl.min(tl.where(matches, cols.to(tl.int64), N), axis=0)
        first_index = tl.minimum(first_index, local_index)

    if IS_FLOAT:
        all_nan = valid_count == 0
        selected = tl.where(all_nan, float("nan"), selected)
        first_index = tl.where(all_nan, 0, first_index)

    tl.store(out_values + row, selected)
    tl.store(out_indices + row, first_index)


def _large_nanmedian(rows, values, indices):
    M, N = rows.shape
    is_float = rows.dtype.is_floating_point
    if is_float:
        cleaned = torch.empty_like(rows)
        valid_counts = torch.zeros((M,), dtype=torch.int32, device=rows.device)
        num_blocks = triton.cdiv(N, _SORT_BLOCK_N)
        with torch_device_fn.device(rows.device):
            _clean_nan_and_count_kernel[(M, num_blocks)](
                rows,
                cleaned,
                valid_counts,
                N,
                _SORT_BLOCK_N,
                num_warps=8,
                num_stages=1,
            )
        sort_input = cleaned
    else:
        valid_counts = torch.empty((1,), dtype=torch.int32, device=rows.device)
        sort_input = rows

    sorted_values, _ = mthreads_sort(sort_input, dim=-1, descending=False)
    with torch_device_fn.device(rows.device):
        _select_sorted_median_kernel[(M,)](
            rows,
            sorted_values,
            valid_counts,
            values,
            indices,
            N,
            is_float,
            _SORT_BLOCK_N,
            num_warps=8,
            num_stages=1,
        )


def _nanmedian_dim_impl(inp, dim, keepdim, out=None):
    dim = _normalize_dim(dim, inp.ndim)

    if inp.ndim == 0:
        if out is None:
            values = inp.clone()
            indices = torch.zeros((), dtype=torch.long, device=inp.device)
        else:
            values, indices = out
            values.copy_(inp)
            indices.zero_()
        return NanMedian(values=values, indices=indices)

    shape = list(inp.shape)
    N = shape[dim]
    out_shape = shape[:dim] + shape[dim + 1 :]
    M = math.prod(out_shape)
    keepdim_shape = shape.copy()
    keepdim_shape[dim] = 1
    output_shape = keepdim_shape if keepdim else out_shape
    compute_shape = output_shape if out is not None else keepdim_shape

    if N == 0:
        if M != 0:
            raise IndexError(
                f"median(): Expected reduction dim {dim} to have non-zero size."
            )
        if out is None:
            values = torch.empty(compute_shape, dtype=inp.dtype, device=inp.device)
            indices = torch.empty(compute_shape, dtype=torch.long, device=inp.device)
            if not keepdim:
                values = torch.squeeze(values, dim)
                indices = torch.squeeze(indices, dim)
        else:
            values, indices = out
        return NanMedian(values=values, indices=indices)

    if out is None:
        values = torch.empty(compute_shape, dtype=inp.dtype, device=inp.device)
        indices = torch.empty(compute_shape, dtype=torch.long, device=inp.device)
    else:
        values, indices = out

    if M == 0:
        if out is None and not keepdim:
            values = torch.squeeze(values, dim)
            indices = torch.squeeze(indices, dim)
        return NanMedian(values=values, indices=indices)

    rows = torch.movedim(inp, dim, -1).contiguous().reshape(M, N)
    values_contiguous = values.is_contiguous()
    indices_contiguous = indices.is_contiguous()
    flat_values = (
        values.reshape(M)
        if values_contiguous
        else torch.empty((M,), dtype=values.dtype, device=values.device)
    )
    flat_indices = (
        indices.reshape(M)
        if indices_contiguous
        else torch.empty((M,), dtype=indices.dtype, device=indices.device)
    )
    if N <= MAX_BLOCK_N and inp.dtype is not torch.float64:
        block_n = triton.next_power_of_2(N)
        with torch_device_fn.device(inp.device):
            nanmedian_select_kernel[(M,)](
                rows,
                flat_values,
                flat_indices,
                M,
                N,
                block_n,
                False,
                num_warps=4,
                num_stages=1,
            )
    elif inp.dtype in RADIX_SELECT_DTYPES and N <= INT32_MAX:
        if N <= 1024:
            block_n = triton.next_power_of_2(N)
        elif N <= 8192:
            block_n = 1024
        else:
            block_n = 4096
        with torch_device_fn.device(inp.device):
            nanmedian_radix_select_kernel[(M,)](
                rows,
                flat_values,
                flat_indices,
                M,
                N,
                block_n,
                4,
                False,
                True,
                num_warps=8,
                num_stages=1,
            )
    else:
        _large_nanmedian(rows, flat_values, flat_indices)

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

    if (
        inp.numel() > MAX_BLOCK_N
        and inp.numel() <= INT32_MAX
        and inp.dtype in RADIX_SELECT_DTYPES
    ):
        return _nanmedian_cuda_flat_radix_select(inp, out=out)

    flat = inp.reshape(-1)
    if out is None:
        return _nanmedian_dim_impl(flat, 0, False).values

    indices = torch.empty((), dtype=torch.long, device=inp.device)
    _nanmedian_dim_impl(flat, 0, False, out=(out, indices))
    return out


def nanmedian(inp):
    logger.debug("GEMS_MTHREADS NANMEDIAN")
    _check_supported_dtype(inp)
    return _nanmedian_flat_impl(inp)


def nanmedian_out(inp, *, out):
    logger.debug("GEMS_MTHREADS NANMEDIAN OUT")
    _check_supported_dtype(inp)
    return _nanmedian_flat_impl(inp, out=out)


def nanmedian_dim(inp, dim=-1, keepdim=False):
    logger.debug("GEMS_MTHREADS NANMEDIAN DIM")
    _check_supported_dtype(inp)
    return _nanmedian_dim_impl(inp, dim, keepdim)


def nanmedian_dim_values(inp, dim=-1, keepdim=False, *, values, indices):
    logger.debug("GEMS_MTHREADS NANMEDIAN DIM VALUES")
    _check_supported_dtype(inp)
    return _nanmedian_dim_impl(inp, dim, keepdim, out=(values, indices))
