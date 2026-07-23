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
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_max, get_dtype_min

from .sort import convert_to_uint_preverse_order

logger = logging.getLogger(__name__)

NanMedian = namedtuple("nanmedian", ["values", "indices"])
MAX_BLOCK_N = 128
FLOAT_SELECT_BLOCK_N = 128
MAX_NDIM = 8


@triton.jit
def _is_not_nan(vals):
    vals_fp32 = vals.to(tl.float32)
    return vals_fp32 == vals_fp32


@libentry()
@triton.jit
def nanmedian_direct_select_kernel(
    inp,
    out_values,
    out_indices,
    N: tl.constexpr,
    STRIDE_DIM: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    S4: tl.constexpr,
    S5: tl.constexpr,
    S6: tl.constexpr,
    S7: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    T2: tl.constexpr,
    T3: tl.constexpr,
    T4: tl.constexpr,
    T5: tl.constexpr,
    T6: tl.constexpr,
    T7: tl.constexpr,
    DIM: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    dtype = inp.dtype.element_ty
    max_value = get_dtype_max(dtype)
    fallback_value = get_dtype_min(dtype)

    idx = pid
    base = tl.full((), 0, dtype=tl.int64)
    if NDIM >= 8:
        if DIM != 7:
            coord = idx % S7
            idx = idx // S7
            base += coord * T7
    if NDIM >= 7:
        if DIM != 6:
            coord = idx % S6
            idx = idx // S6
            base += coord * T6
    if NDIM >= 6:
        if DIM != 5:
            coord = idx % S5
            idx = idx // S5
            base += coord * T5
    if NDIM >= 5:
        if DIM != 4:
            coord = idx % S4
            idx = idx // S4
            base += coord * T4
    if NDIM >= 4:
        if DIM != 3:
            coord = idx % S3
            idx = idx // S3
            base += coord * T3
    if NDIM >= 3:
        if DIM != 2:
            coord = idx % S2
            idx = idx // S2
            base += coord * T2
    if NDIM >= 2:
        if DIM != 1:
            coord = idx % S1
            idx = idx // S1
            base += coord * T1
    if NDIM >= 1:
        if DIM != 0:
            coord = idx % S0
            base += coord * T0
    vals = tl.load(inp + base + offsets * STRIDE_DIM, mask=mask, other=max_value)

    if dtype.is_floating():
        valid = mask & _is_not_nan(vals)
    else:
        valid = mask
    valid_count = tl.sum(valid.to(tl.int32), axis=0)
    median_rank = (valid_count - 1) // 2

    active = valid
    median_val = tl.full((), fallback_value, dtype=vals.dtype)
    median_idx = tl.full((), 0, dtype=tl.int32)
    for select_iter in tl.static_range(0, BLOCK_N):
        select_vals = tl.where(active, vals, max_value)
        cur_val = tl.min(select_vals, axis=0)
        cur_idx = tl.min(tl.where(active & (vals == cur_val), offsets, BLOCK_N), axis=0)
        take = select_iter == median_rank
        median_val = tl.where(take, cur_val, median_val)
        median_idx = tl.where(take, cur_idx, median_idx)
        active = active & (offsets != cur_idx)

    if dtype.is_floating():
        all_nan = valid_count == 0
        median_val = tl.where(all_nan, float("nan"), median_val)
        median_idx = tl.where(all_nan, 0, median_idx)

    tl.store(out_values + pid, median_val)
    tl.store(out_indices + pid, median_idx)


@libentry()
@triton.jit
def nanmedian_float_key_select_kernel(
    inp,
    out_values,
    out_indices,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KEY_BITS: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    count = tl.full((), 0, dtype=tl.int32)
    if KEY_BITS == 64:
        zero_key = tl.full((), 0, dtype=tl.uint64)
        one_key = tl.full((), 1, dtype=tl.uint64)
        two_key = tl.full((), 2, dtype=tl.uint64)
        max_key = tl.full((), 0xFFFFFFFFFFFFFFFF, dtype=tl.uint64)
    else:
        zero_key = tl.full((), 0, dtype=tl.uint32)
        one_key = tl.full((), 1, dtype=tl.uint32)
        two_key = tl.full((), 2, dtype=tl.uint32)
        max_key = tl.full((), 0xFFFFFFFF, dtype=tl.uint32)
    min_key = max_key
    upper_key = zero_key

    for start in tl.range(0, N, BLOCK_N):
        cols = start + offsets
        mask = cols < N
        vals = tl.load(inp + pid * N + cols, mask=mask, other=0.0)
        valid = mask & _is_not_nan(vals)
        count += tl.sum(valid.to(tl.int32), axis=0)
        key_vals = vals if KEY_BITS == 64 else vals.to(tl.float32)
        keys = convert_to_uint_preverse_order(key_vals, False)
        keys = keys.to(tl.uint64) if KEY_BITS == 64 else keys.to(tl.uint32)
        min_key = tl.minimum(min_key, tl.min(tl.where(valid, keys, max_key), axis=0))
        upper_key = tl.maximum(
            upper_key, tl.max(tl.where(valid, keys, zero_key), axis=0)
        )

    target = tl.maximum((count - 1) // 2, 0)
    lower_key = min_key
    for _ in tl.static_range(0, KEY_BITS):
        active = lower_key < upper_key
        mid_key = lower_key + ((upper_key - lower_key) // two_key)
        le_count = tl.full((), 0, dtype=tl.int32)

        for start in tl.range(0, N, BLOCK_N):
            cols = start + offsets
            mask = cols < N
            vals = tl.load(inp + pid * N + cols, mask=mask, other=0.0)
            valid = mask & _is_not_nan(vals)
            key_vals = vals if KEY_BITS == 64 else vals.to(tl.float32)
            keys = convert_to_uint_preverse_order(key_vals, False)
            keys = keys.to(tl.uint64) if KEY_BITS == 64 else keys.to(tl.uint32)
            le_count += tl.sum((valid & (keys <= mid_key)).to(tl.int32), axis=0)

        go_left = le_count > target
        lower_key = tl.where(active & ~go_left, mid_key + one_key, lower_key)
        upper_key = tl.where(active & go_left, mid_key, upper_key)

    result_idx = tl.full((), 0, dtype=tl.int32)
    first_idx = tl.full((), N, dtype=tl.int32)
    for start in tl.range(0, N, BLOCK_N):
        cols = start + offsets
        mask = cols < N
        vals = tl.load(inp + pid * N + cols, mask=mask, other=0.0)
        valid = mask & _is_not_nan(vals)
        key_vals = vals if KEY_BITS == 64 else vals.to(tl.float32)
        keys = convert_to_uint_preverse_order(key_vals, False)
        keys = keys.to(tl.uint64) if KEY_BITS == 64 else keys.to(tl.uint32)
        local_idx = tl.min(tl.where(valid & (keys == lower_key), cols, N), axis=0)
        first_idx = tl.minimum(first_idx, local_idx)

    result_idx = tl.where(count > 0, first_idx, result_idx)
    result_val = tl.load(inp + pid * N + result_idx, mask=count > 0, other=float("nan"))
    result_val = tl.where(count > 0, result_val, float("nan"))

    tl.store(out_values + pid, result_val)
    tl.store(out_indices + pid, result_idx)


@libentry()
@triton.jit
def nanmedian_sorted_gather_kernel(
    sorted_values,
    sorted_indices,
    valid_counts,
    out_values,
    out_indices,
    N: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    pid = ext.program_id(0)
    if IS_FLOAT:
        count = tl.load(valid_counts + pid)
        rank = tl.where(count > 0, (count - 1) // 2, 0)
        result_val = tl.load(
            sorted_values + pid * N + rank, mask=count > 0, other=float("nan")
        )
        result_idx = tl.load(sorted_indices + pid * N + rank, mask=count > 0, other=0)
        result_val = tl.where(count > 0, result_val, float("nan"))
        result_idx = tl.where(count > 0, result_idx, 0)
    else:
        rank = (N - 1) // 2
        result_val = tl.load(sorted_values + pid * N + rank)
        result_idx = tl.load(sorted_indices + pid * N + rank)

    tl.store(out_values + pid, result_val)
    tl.store(out_indices + pid, result_idx)


def _check_supported_dtype(inp):
    if inp.dtype is torch.bool:
        raise NotImplementedError("\"median_out_impl\" not implemented for 'Bool'")


def _normalize_dim(dim, ndim):
    if ndim == 0:
        if dim in (0, -1):
            return 0
    elif -ndim <= dim < ndim:
        return dim % ndim
    raise IndexError(
        f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
    )


def _pad_meta(values, fill):
    if len(values) > MAX_NDIM:
        raise NotImplementedError(
            f"nanmedian supports input rank <= {MAX_NDIM} on Kunlunxin"
        )
    return tuple(values) + (fill,) * (MAX_NDIM - len(values))


def _empty_flat_value(inp):
    result = torch.empty((), dtype=inp.dtype, device=inp.device)
    if inp.dtype.is_floating_point:
        result.fill_(float("nan"))
    else:
        result.fill_(torch.iinfo(inp.dtype).min)
    return result


def _reduction_rows(inp, dim, M, N):
    if dim == inp.ndim - 1:
        return inp.reshape(M, N)
    return torch.movedim(inp, dim, -1).reshape(M, N)


def _nanmedian_sort_fallback(inp, dim, M, N, values, indices):
    rows = _reduction_rows(inp, dim, M, N)
    if torch.is_floating_point(rows):
        key_bits = 64 if rows.dtype is torch.float64 else 32
        with torch_device_fn.device(inp.device):
            nanmedian_float_key_select_kernel[(M,)](
                rows,
                values,
                indices,
                N,
                FLOAT_SELECT_BLOCK_N,
                key_bits,
                num_warps=4,
                num_stages=1,
                buffer_size_limit=2048,
            )
        return
    else:
        # Kunlunxin native sort cannot return (uint8 values, int64 indices).
        sort_rows = rows.to(torch.int32) if rows.dtype is torch.uint8 else rows
        sorted_values, sorted_indices = torch.sort(sort_rows, dim=1)
        valid_counts = sorted_indices
        is_float = False

    with torch_device_fn.device(inp.device):
        nanmedian_sorted_gather_kernel[(M,)](
            sorted_values,
            sorted_indices,
            valid_counts,
            values,
            indices,
            N,
            is_float,
            num_warps=1,
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

    flat_values = values.reshape(M)
    flat_indices = indices.reshape(M)

    if N <= MAX_BLOCK_N:
        stride_tuple = tuple(inp.stride())
        stride_dim = stride_tuple[dim]
        shape_meta = _pad_meta(shape, 1)
        stride_meta = _pad_meta(stride_tuple, 0)
        block_n = triton.next_power_of_2(N)
        num_warps = 4
        with torch_device_fn.device(inp.device):
            nanmedian_direct_select_kernel[(M,)](
                inp,
                flat_values,
                flat_indices,
                N,
                stride_dim,
                *shape_meta,
                *stride_meta,
                dim,
                inp.ndim,
                block_n,
                num_warps=num_warps,
                num_stages=1,
                buffer_size_limit=2048,
            )
    else:
        # Avoid the Kunlunxin TritonXPU large-N radix lowering crash.
        _nanmedian_sort_fallback(inp, dim, M, N, flat_values, flat_indices)

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

    flat = inp.reshape(-1)
    if out is None:
        return _nanmedian_dim_impl(flat, 0, False).values

    indices = torch.empty((), dtype=torch.long, device=inp.device)
    _nanmedian_dim_impl(flat, 0, False, out=(out, indices))
    return out


def nanmedian(inp):
    logger.debug("GEMS_KUNLUNXIN NANMEDIAN")
    _check_supported_dtype(inp)
    return _nanmedian_flat_impl(inp)


def nanmedian_out(inp, *, out):
    logger.debug("GEMS_KUNLUNXIN NANMEDIAN_OUT")
    _check_supported_dtype(inp)
    return _nanmedian_flat_impl(inp, out=out)


def nanmedian_dim(inp, dim=-1, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN NANMEDIAN_DIM")
    _check_supported_dtype(inp)
    return _nanmedian_dim_impl(inp, dim, keepdim)


def nanmedian_dim_values(inp, dim=-1, keepdim=False, *, values, indices):
    logger.debug("GEMS_KUNLUNXIN NANMEDIAN_DIM_VALUES")
    _check_supported_dtype(inp)
    return _nanmedian_dim_impl(inp, dim, keepdim, out=(values, indices))
