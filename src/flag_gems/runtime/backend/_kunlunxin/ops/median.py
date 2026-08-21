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

MedianResult = namedtuple("median", ["values", "indices"])
MAX_BLOCK_N = 128
BOOL_BLOCK_N = 1024
MAX_NDIM = 8
KEY_BLOCK_LIMIT = 8192


@triton.jit
def _median_is_nan(vals):
    vals_fp32 = vals.to(tl.float32)
    return vals_fp32 != vals_fp32


@triton.jit
def _median_keys(vals, KEY_BITS: tl.constexpr):
    if KEY_BITS == 64:
        if not vals.dtype.is_floating() and vals.dtype.primitive_bitwidth < 64:
            w = vals.to(tl.int64)
        else:
            w = vals
        return convert_to_uint_preverse_order(w, False)
    if vals.dtype.is_floating():
        w = vals
    elif vals.dtype.primitive_bitwidth < 32:
        w = vals.to(tl.int32)
    else:
        w = vals
    k = convert_to_uint_preverse_order(w, False)
    return k.to(tl.uint32)


@libentry()
@triton.jit
def median_direct_select_kernel(
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
    data = tl.load(inp + base + offsets * STRIDE_DIM, mask=mask, other=max_value)

    if data.dtype.is_floating():
        nan_mask = mask & _median_is_nan(data)
        has_nan = tl.max(nan_mask.to(tl.int32), axis=0) != 0
        first_nan_idx = tl.min(tl.where(nan_mask, offsets, BLOCK_N), axis=0)
    else:
        has_nan = False
        first_nan_idx = tl.full((), 0, dtype=tl.int32)

    median_rank = (N - 1) // 2

    active = mask
    median_val = tl.full((), fallback_value, dtype=data.dtype)
    median_idx = tl.full((), 0, dtype=tl.int32)
    for select_iter in tl.static_range(0, BLOCK_N):
        select_vals = tl.where(active, data, max_value)
        cur_val = tl.min(select_vals, axis=0)
        cur_idx = tl.min(tl.where(active & (data == cur_val), offsets, BLOCK_N), axis=0)
        take = select_iter == median_rank
        median_val = tl.where(take, cur_val, median_val)
        median_idx = tl.where(take, cur_idx, median_idx)
        active = active & (offsets != cur_idx)

    if data.dtype.is_floating():
        median_val = tl.where(has_nan, float("nan"), median_val)
        median_idx = tl.where(has_nan, first_nan_idx, median_idx)

    tl.store(out_values + pid, median_val)
    tl.store(out_indices + pid, median_idx.to(tl.int64))


@libentry()
@triton.jit
def median_key_info_kernel(
    inp,
    keybuf,
    mins,
    maxs,
    nan_flags,
    nan_firsts,
    N: tl.constexpr,
    KEY_BITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PREORDERED: tl.constexpr,
):
    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    if PREORDERED:
        vals = tl.load(inp + pid * N + cols, mask=mask, other=0)
        keys = vals
        vals_lo = tl.load(inp + pid * N + cols, mask=mask, other=0xFFFFFFFFFFFFFFFF)
        vals_hi = tl.load(inp + pid * N + cols, mask=mask, other=0)
        keys_lo = vals_lo
        keys_hi = vals_hi
    else:
        dtype = inp.dtype.element_ty
        is_float: tl.constexpr = dtype.is_floating()
        if is_float:
            min_fill = float("-inf")
            max_fill = float("inf")
        else:
            min_fill = get_dtype_min(dtype)
            max_fill = get_dtype_max(dtype)
        vals = tl.load(inp + pid * N + cols, mask=mask, other=max_fill)
        keys = _median_keys(vals, KEY_BITS)
        vals_lo = tl.load(inp + pid * N + cols, mask=mask, other=min_fill)
        vals_hi = tl.load(inp + pid * N + cols, mask=mask, other=max_fill)
        keys_lo = _median_keys(vals_lo, KEY_BITS)
        keys_hi = _median_keys(vals_hi, KEY_BITS)
    tl.store(keybuf + pid * BLOCK_N + cols, keys, mask=mask)
    lo = tl.min(keys_lo, axis=0)
    hi = tl.max(keys_hi, axis=0)
    tl.store(mins + pid, lo)
    tl.store(maxs + pid, hi)
    if not PREORDERED and vals.dtype.is_floating():
        nan = mask & _median_is_nan(vals)
        has_nan = tl.max(nan.to(tl.int32), axis=0) != 0
        first_nan = tl.min(tl.where(nan, cols, BLOCK_N), axis=0)
        tl.store(nan_flags + pid, has_nan.to(tl.int32))
        tl.store(nan_firsts + pid, first_nan)


@libentry()
@triton.jit
def median_count_le_kernel(
    keys,
    lo,
    hi,
    counts,
    TARGET: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    keys_v = tl.load(keys + pid * BLOCK_N + cols, mask=mask, other=0)
    lo_v = tl.load(lo + pid)
    hi_v = tl.load(hi + pid)
    mid = lo_v + ((hi_v - lo_v) >> 1)
    le = tl.sum((mask & (keys_v <= mid)).to(tl.int32), axis=0)
    go_left = le > TARGET
    active = lo_v < hi_v
    new_hi = tl.where(go_left & active, mid, hi_v)
    new_lo = tl.where(~go_left & active, mid + 1, lo_v)
    tl.store(counts + pid, le)
    tl.store(lo + pid, new_lo)
    tl.store(hi + pid, new_hi)


@libentry()
@triton.jit
def median_select_kernel(
    inp,
    keybuf,
    sel_keys,
    nan_flags,
    nan_firsts,
    out_values,
    out_indices,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    sel = tl.load(sel_keys + pid)
    keys_v = tl.load(keybuf + pid * BLOCK_N + cols, mask=mask, other=0)
    km = mask & (keys_v == sel)
    first = tl.min(tl.where(km, cols, BLOCK_N), axis=0)
    has_nan = tl.load(nan_flags + pid) != 0
    first_nan = tl.load(nan_firsts + pid)
    ridx = tl.where(has_nan, first_nan, first)
    rval = tl.load(inp + pid * N + ridx, mask=ridx < N, other=0.0)
    if inp.dtype.element_ty.is_floating():
        rval = tl.where(has_nan, float("nan"), rval)
    tl.store(out_values + pid, rval)
    tl.store(out_indices + pid, ridx.to(tl.int64))


@libentry()
@triton.jit
def median_bool_row_kernel(
    inp,
    out_values,
    out_indices,
    N,
    BLOCK_N: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    true_count = tl.full((), 0, dtype=tl.int32)
    first_false = tl.full((), 2147483647, dtype=tl.int32)
    first_true = tl.full((), 2147483647, dtype=tl.int32)

    for start in tl.range(0, N, BLOCK_N):
        cols = start + offsets
        mask = cols < N
        vals = tl.load(inp + pid * N + cols, mask=mask, other=False)
        true_count += tl.sum((vals & mask).to(tl.int32), axis=0)
        first_false = tl.minimum(
            first_false,
            tl.min(tl.where(mask & ~vals, cols, 2147483647), axis=0),
        )
        first_true = tl.minimum(
            first_true,
            tl.min(tl.where(mask & vals, cols, 2147483647), axis=0),
        )

    false_count = N - true_count
    rank = (N - 1) // 2
    take_true = rank >= false_count
    median_val = take_true
    median_idx = tl.where(take_true, first_true, first_false)

    tl.store(out_values + pid, median_val)
    tl.store(out_indices + pid, median_idx.to(tl.int64))


def _check_supported_dtype(inp):
    if inp.dtype is torch.complex64 or inp.dtype is torch.complex128:
        raise NotImplementedError("\"median_out_impl\" not implemented for complex")


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
            f"median supports input rank <= {MAX_NDIM} on Kunlunxin"
        )
    return tuple(values) + (fill,) * (MAX_NDIM - len(values))


def _empty_flat_result(inp):
    result = torch.empty((), dtype=inp.dtype, device=inp.device)
    if inp.dtype.is_complex:
        result.real.fill_(float("nan"))
        result.imag.fill_(0.0)
        return result
    if inp.dtype.is_floating_point:
        result.fill_(float("nan"))
    elif inp.dtype == torch.bool:
        result.fill_(True)
    elif inp.dtype in (torch.int32, torch.int64):
        result.fill_(torch.iinfo(inp.dtype).min)
    else:
        result.fill_(0)
    return result


def _reduction_rows(inp, dim, M, N):
    if dim == inp.ndim - 1:
        return inp.reshape(M, N)
    return torch.movedim(inp, dim, -1).contiguous().reshape(M, N)


def _key_bits(dtype):
    if dtype in (torch.float64, torch.int64):
        return 64
    return 32


def _order_keys64(t):
    bits = t.view(torch.int64)
    key = bits ^ (0x8000000000000000 | (bits >> 63))
    return key.to(torch.uint64)


def _median_key_select(rows, N):
    M = rows.shape[0]
    key_bits = _key_bits(rows.dtype)
    block_n = triton.next_power_of_2(N)
    if block_n > KEY_BLOCK_LIMIT:
        raise NotImplementedError(
            f"median reduction width {N} exceeds Kunlunxin limit"
        )
    key_dtype = torch.uint64 if key_bits == 64 else torch.uint32
    keybuf = torch.empty((M, block_n), dtype=key_dtype, device=rows.device)
    mins = torch.empty((M,), dtype=key_dtype, device=rows.device)
    maxs = torch.empty((M,), dtype=key_dtype, device=rows.device)
    nan_flags = torch.empty((M,), dtype=torch.int32, device=rows.device)
    nan_firsts = torch.empty((M,), dtype=torch.int32, device=rows.device)
    counts = torch.empty((M,), dtype=torch.int32, device=rows.device)

    preordered = rows.dtype in (torch.float64, torch.int64)
    if preordered:
        work = _order_keys64(rows)
    elif rows.dtype in (torch.float16, torch.bfloat16):
        work = rows.to(torch.float32)
    elif rows.dtype in (torch.int8, torch.uint8, torch.int16):
        work = rows.to(torch.int32)
    else:
        work = rows
    if preordered and rows.dtype == torch.float64:
        nanf = rows.isnan()
        nan_flags = nanf.any(dim=1).to(torch.int32).contiguous()
        nan_firsts = nanf.to(torch.int64).argmax(dim=1).to(torch.int32).contiguous()
    elif preordered:
        nan_flags.zero_()
        nan_firsts.zero_()

    with torch_device_fn.device(work.device):
        median_key_info_kernel[(M,)](
            work,
            keybuf,
            mins,
            maxs,
            nan_flags,
            nan_firsts,
            N,
            key_bits,
            block_n,
            preordered,
            num_warps=4,
            num_stages=1,
            buffer_size_limit=2048,
        )
        lo = mins
        hi = maxs
        target = (N - 1) // 2
        for _ in range(key_bits):
            median_count_le_kernel[(M,)](
                keybuf,
                lo,
                hi,
                counts,
                target,
                N,
                block_n,
                num_warps=4,
                num_stages=1,
                buffer_size_limit=2048,
            )
        sel_keys = lo
        out_values = torch.empty((M,), dtype=rows.dtype, device=rows.device)
        out_indices = torch.empty((M,), dtype=torch.long, device=rows.device)
        median_select_kernel[(M,)](
            rows,
            keybuf,
            sel_keys,
            nan_flags,
            nan_firsts,
            out_values,
            out_indices,
            N,
            block_n,
            num_warps=4,
            num_stages=1,
            buffer_size_limit=2048,
        )
    return out_values, out_indices


def _median_dim_impl(inp, dim, keepdim, out=None):
    dim = _normalize_dim(dim, inp.ndim)

    if inp.ndim == 0:
        if out is None:
            values = inp.clone()
            indices = torch.zeros((), dtype=torch.long, device=inp.device)
        else:
            values, indices = out
            values.copy_(inp)
            indices.zero_()
        return MedianResult(values=values, indices=indices)

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
        return MedianResult(values=values, indices=indices)

    if out is None:
        values = torch.empty(compute_shape, dtype=inp.dtype, device=inp.device)
        indices = torch.empty(compute_shape, dtype=torch.long, device=inp.device)
    else:
        values, indices = out

    if M == 0:
        if out is None and not keepdim:
            values = torch.squeeze(values, dim)
            indices = torch.squeeze(indices, dim)
        return MedianResult(values=values, indices=indices)

    flat_values = values.reshape(M)
    flat_indices = indices.reshape(M)

    with torch_device_fn.device(inp.device):
        if inp.dtype == torch.bool:
            rows = _reduction_rows(inp, dim, M, N)
            median_bool_row_kernel[(M,)](
                rows,
                flat_values,
                flat_indices,
                N,
                BOOL_BLOCK_N,
                num_warps=4,
                num_stages=1,
                buffer_size_limit=2048,
            )
        elif N <= MAX_BLOCK_N and inp.dtype != torch.int64:
            stride_tuple = tuple(inp.stride())
            stride_dim = stride_tuple[dim]
            shape_meta = _pad_meta(shape, 1)
            stride_meta = _pad_meta(stride_tuple, 0)
            block_n = triton.next_power_of_2(N)
            num_warps = 4 if block_n > 32 else 1
            median_direct_select_kernel[(M,)](
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
            rows = _reduction_rows(inp, dim, M, N)
            out_values, out_indices = _median_key_select(rows, N)
            flat_values.copy_(out_values)
            flat_indices.copy_(out_indices)

    if out is None and not keepdim:
        values = torch.squeeze(values, dim)
        indices = torch.squeeze(indices, dim)

    return MedianResult(values=values, indices=indices)


def _median_flat_impl(inp, out=None):
    if inp.numel() == 0:
        result = _empty_flat_result(inp)
        if out is not None:
            out.copy_(result)
            return out
        return result

    flat = inp.reshape(-1)
    if out is None:
        return _median_dim_impl(flat, 0, False).values

    indices = torch.empty((), dtype=torch.long, device=inp.device)
    _median_dim_impl(flat, 0, False, out=(out, indices))
    return out


def median(inp):
    logger.debug("GEMS_KUNLUNXIN MEDIAN")
    _check_supported_dtype(inp)
    return _median_flat_impl(inp)


def median_out(inp, *, out):
    logger.debug("GEMS_KUNLUNXIN MEDIAN_OUT")
    _check_supported_dtype(inp)
    return _median_flat_impl(inp, out=out)


def median_dim(inp, dim=-1, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN MEDIAN_DIM")
    _check_supported_dtype(inp)
    return _median_dim_impl(inp, dim, keepdim)


def median_dim_values(inp, dim=-1, keepdim=False, *, values, indices):
    logger.debug("GEMS_KUNLUNXIN MEDIAN_DIM_VALUES")
    _check_supported_dtype(inp)
    return _median_dim_impl(inp, dim, keepdim, out=(values, indices))
