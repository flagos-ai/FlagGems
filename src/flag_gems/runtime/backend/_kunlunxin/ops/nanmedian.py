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
import warnings
from collections import namedtuple

import numpy as np
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
KEY_BLOCK_LIMIT = 32768


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
    elif inp.dtype in (torch.int8, torch.int16, torch.uint8):
        # torch's empty-input semantics: 0 for 8/16-bit ints, iinfo.min for
        # 32/64-bit ints (see CPU nanmedian behavior).
        result.fill_(0)
    else:
        result.fill_(torch.iinfo(inp.dtype).min)
    return result


def _reduction_rows(inp, dim, M, N):
    if dim == inp.ndim - 1:
        return inp.reshape(M, N)
    return torch.movedim(inp, dim, -1).reshape(M, N)


def _nanmedian_sort_fallback(inp, dim, M, N, values, indices):
    rows = _reduction_rows(inp, dim, M, N)
    key_bits = 64 if rows.dtype in (torch.float64, torch.int64) else 32
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


def _nanmedian_dim_legacy(inp, dim, M, N, values, indices):
    shape = list(inp.shape)
    stride_tuple = tuple(inp.stride())
    stride_dim = stride_tuple[dim]
    shape_meta = _pad_meta(shape, 1)
    stride_meta = _pad_meta(stride_tuple, 0)
    block_n = triton.next_power_of_2(N)
    with torch_device_fn.device(inp.device):
        nanmedian_direct_select_kernel[(M,)](
            inp,
            values,
            indices,
            N,
            stride_dim,
            *shape_meta,
            *stride_meta,
            dim,
            inp.ndim,
            block_n,
            num_warps=4,
            num_stages=1,
            buffer_size_limit=2048,
        )


def _check_dim_out(out, inp, output_shape):
    # `nanmedian.dim_values` is registered on the Autograd key, so ATen's own
    # structured-kernel meta checks for the `out=` pair never run.  Reproduce
    # them here (measured against ATen CPU, not inferred): mismatching dtypes
    # raise, mismatching shapes are resized with a UserWarning.
    values, indices = out
    if values.dtype is not inp.dtype:
        raise RuntimeError(
            "Expected tensor for 'values' to have the same type as tensor for "
            f"argument #2 'self'; but type {values.dtype} does not equal "
            f"{inp.dtype} (while checking arguments for median)"
        )
    if indices.dtype is not torch.long:
        raise RuntimeError(
            "Expected tensor for argument #1 'indices' to have scalar type Long; "
            f"but got {indices.dtype} instead (while checking arguments for median)"
        )
    shape = tuple(output_shape)
    for tensor in (values, indices):
        if tuple(tensor.shape) == shape:
            continue
        if tensor.numel() > 0:
            warnings.warn(
                "An output with one or more elements was resized since it had "
                f"shape {list(tensor.shape)}, which does not match the required "
                f"output shape {list(shape)}. This behavior is deprecated, and "
                "in a future PyTorch release outputs will not be resized unless "
                "they have zero elements.",
                UserWarning,
                stacklevel=4,
            )
        tensor.resize_(shape)
    return values, indices


def _dim_out_row_view(tensor, M):
    # The dim kernels address their outputs as a flat, element-contiguous run
    # (`tl.store(out + pid, ...)`), so a non-contiguous `out=` tensor has to be
    # staged through a contiguous scratch and copied back afterwards.  Handing
    # the strided tensor straight to the kernel writes M consecutive slots and
    # therefore both mislays the results and clobbers whatever shares the
    # buffer (measured: a (4,2) buffer's second column was overwritten).
    if tensor.is_contiguous():
        return tensor.reshape(M), None
    scratch = torch.empty(M, dtype=tensor.dtype, device=tensor.device)
    return scratch, tensor


def _nanmedian_dim_impl(inp, dim, keepdim, out=None):
    dim = _normalize_dim(dim, inp.ndim)

    if inp.ndim == 0:
        if out is None:
            values = inp.clone()
            indices = torch.zeros((), dtype=torch.long, device=inp.device)
        else:
            values, indices = _check_dim_out(out, inp, ())
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
            values, indices = _check_dim_out(out, inp, output_shape)
        return NanMedian(values=values, indices=indices)

    if out is None:
        values = torch.empty(compute_shape, dtype=inp.dtype, device=inp.device)
        indices = torch.empty(compute_shape, dtype=torch.long, device=inp.device)
    else:
        values, indices = _check_dim_out(out, inp, output_shape)

    if M == 0:
        if out is None and not keepdim:
            values = torch.squeeze(values, dim)
            indices = torch.squeeze(indices, dim)
        return NanMedian(values=values, indices=indices)

    flat_values, values_dst = _dim_out_row_view(values, M)
    flat_indices, indices_dst = _dim_out_row_view(indices, M)

    if inp.dtype is torch.float64:
        # No native float64 on this platform (storage is 32-bit), so the key
        # arithmetic of the fast path does not apply; keep the legacy route.
        if N <= MAX_BLOCK_N:
            _nanmedian_dim_legacy(inp, dim, M, N, flat_values, flat_indices)
        else:
            _nanmedian_sort_fallback(inp, dim, M, N, flat_values, flat_indices)
    elif N == 1:
        # Closed form: a length-1 reduction is the element itself (NaN
        # included), so there is nothing to search for.
        rows = _nmdim_rows(inp, dim, M, N)
        torch.ops.aten._copy_from(rows.reshape(M), flat_values, False)
        flat_indices.zero_()
    else:
        rows = _nmdim_rows(inp, dim, M, N)
        _nmdim_launch(rows, M, N, flat_values, flat_indices)

    if values_dst is not None:
        torch.ops.aten._copy_from(
            flat_values.reshape(values_dst.shape), values_dst, False
        )
    if indices_dst is not None:
        torch.ops.aten._copy_from(
            flat_indices.reshape(indices_dst.shape), indices_dst, False
        )

    if out is None and not keepdim:
        values = torch.squeeze(values, dim)
        indices = torch.squeeze(indices, dim)

    return NanMedian(values=values, indices=indices)


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
def median_key_info_chunk_kernel(
    inp,
    keybuf,
    chunk_mins,
    chunk_maxs,
    chunk_nan,
    N,
    NCHUNK,
    BLOCK_R: tl.constexpr,
    CHUNK: tl.constexpr,
    KEY_BITS: tl.constexpr,
    PREORDERED: tl.constexpr,
):
    # grid: M * NCHUNK; each program handles one CHUNK-sized slice of one row.
    pid = ext.program_id(0)
    chunk = pid % NCHUNK
    cols = tl.arange(0, CHUNK)
    offsets = chunk * CHUNK + cols
    mask = offsets < N
    if PREORDERED:
        keys = tl.load(inp + pid // NCHUNK * N + offsets, mask=mask, other=0)
        keys_lo = tl.load(
            inp + pid // NCHUNK * N + offsets, mask=mask, other=0xFFFFFFFFFFFFFFFF
        )
        keys_hi = tl.load(inp + pid // NCHUNK * N + offsets, mask=mask, other=0)
    else:
        dtype = inp.dtype.element_ty
        is_float: tl.constexpr = dtype.is_floating()
        if is_float:
            min_fill = float("-inf")
            max_fill = float("inf")
        else:
            min_fill = get_dtype_min(dtype)
            max_fill = get_dtype_max(dtype)
        vals = tl.load(inp + pid // NCHUNK * N + offsets, mask=mask, other=max_fill)
        keys = _median_keys(vals, KEY_BITS)
        vals_lo = tl.load(
            inp + pid // NCHUNK * N + offsets, mask=mask, other=min_fill
        )
        vals_hi = tl.load(
            inp + pid // NCHUNK * N + offsets, mask=mask, other=max_fill
        )
        keys_lo = _median_keys(vals_lo, KEY_BITS)
        keys_hi = _median_keys(vals_hi, KEY_BITS)
    # keybuf layout: (M * NCHUNK, CHUNK); pad lanes keep the wrapper's
    # all-ones-key sentinel (only the last chunk of a row has pads).
    tl.store(keybuf + pid * CHUNK + cols, keys, mask=mask)
    cidx = pid // NCHUNK * BLOCK_R + chunk
    if not PREORDERED and vals.dtype.is_floating():
        # NaN first-index must be computed before the uint32 min/max
        # reductions below (XPU miscompile otherwise, see the single-block
        # key info kernel).  Pack = global index, sentinel = 0x7FFFFFFF.
        nan = mask & _median_is_nan(vals)
        local_first = tl.min(tl.where(nan, cols, CHUNK), axis=0)
        pack = tl.where(
            local_first < CHUNK, local_first + chunk * CHUNK, 2147483647
        )
        tl.store(chunk_nan + cidx, pack)
    else:
        tl.store(chunk_nan + cidx, 2147483647)
    lo = tl.min(keys_lo, axis=0)
    hi = tl.max(keys_hi, axis=0)
    tl.store(chunk_mins + cidx, lo)
    tl.store(chunk_maxs + cidx, hi)


@libentry()
@triton.jit
def median_key_info_partial_kernel(
    inp,
    keybuf,
    chunk_mins,
    chunk_maxs,
    chunk_nan,
    N,
    NCHUNK,
    START,
    PARTIAL,
    BLOCK_R: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    TAIL_BASE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    KEY_BITS: tl.constexpr,
    PREORDERED: tl.constexpr,
):
    # grid: (M); handles the leftover tail of a row (last slice), which is
    # at most a full CHUNK but here never padded to a power of two.
    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_P)
    offsets = START + cols
    mask = cols < PARTIAL
    if PREORDERED:
        keys = tl.load(inp + pid * N + offsets, mask=mask, other=0)
        keys_lo = tl.load(
            inp + pid * N + offsets, mask=mask, other=0xFFFFFFFFFFFFFFFF
        )
        keys_hi = tl.load(inp + pid * N + offsets, mask=mask, other=0)
    else:
        dtype = inp.dtype.element_ty
        is_float: tl.constexpr = dtype.is_floating()
        if is_float:
            min_fill = float("-inf")
            max_fill = float("inf")
        else:
            min_fill = get_dtype_min(dtype)
            max_fill = get_dtype_max(dtype)
        vals = tl.load(inp + pid * N + offsets, mask=mask, other=max_fill)
        keys = _median_keys(vals, KEY_BITS)
        vals_lo = tl.load(inp + pid * N + offsets, mask=mask, other=min_fill)
        vals_hi = tl.load(inp + pid * N + offsets, mask=mask, other=max_fill)
        keys_lo = _median_keys(vals_lo, KEY_BITS)
        keys_hi = _median_keys(vals_hi, KEY_BITS)
    tl.store(keybuf + pid * ROW_STRIDE + TAIL_BASE + cols, keys, mask=mask)
    if not PREORDERED and vals.dtype.is_floating():
        nan = mask & _median_is_nan(vals)
        local_first = tl.min(tl.where(nan, cols, BLOCK_P), axis=0)
        pack = tl.where(local_first < BLOCK_P, local_first + START, 2147483647)
        tl.store(chunk_nan + pid * BLOCK_R + NCHUNK - 1, pack)
    else:
        tl.store(chunk_nan + pid * BLOCK_R + NCHUNK - 1, 2147483647)
    lo = tl.min(keys_lo, axis=0)
    hi = tl.max(keys_hi, axis=0)
    tl.store(chunk_mins + pid * BLOCK_R + NCHUNK - 1, lo)
    tl.store(chunk_maxs + pid * BLOCK_R + NCHUNK - 1, hi)


@libentry()
@triton.jit
def median_row_reduce_kernel(
    chunk_data,
    out,
    NCHUNK,
    OTHER: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MODE: tl.constexpr,
):
    # grid: M; MODE 0 = min, 1 = max, 2 = sum (int32).
    # OTHER is the masked-lane fill (max-key for min, 0 for max/sum).
    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < NCHUNK
    v = tl.load(chunk_data + pid * BLOCK_N + cols, mask=mask, other=OTHER)
    if MODE == 0:
        r = tl.min(v, axis=0)
    elif MODE == 1:
        r = tl.max(v, axis=0)
    else:
        r = tl.sum(v.to(tl.int32), axis=0)
    tl.store(out + pid, r)


@libentry()
@triton.jit
def median_count_chunk_kernel(
    keys,
    mid,
    chunk_counts,
    N,
    NCHUNK,
    BLOCK_R: tl.constexpr,
    CHUNK: tl.constexpr,
):
    # grid: M * NCHUNK; count keys <= mid within one chunk slice.
    pid = ext.program_id(0)
    chunk = pid % NCHUNK
    cols = tl.arange(0, CHUNK)
    offsets = chunk * CHUNK + cols
    keys_v = tl.load(keys + pid * CHUNK + cols)
    mid_v = tl.load(mid + pid // NCHUNK)
    le = tl.sum((keys_v <= mid_v).to(tl.int32), axis=0)
    tl.store(chunk_counts + pid // NCHUNK * BLOCK_R + chunk, le)


@libentry()
@triton.jit
def median_update_step_kernel(
    lo,
    hi,
    counts,
    mid_buf,
    TARGET: tl.constexpr,
    KEY_BITS: tl.constexpr,
    FIRST: tl.constexpr,
):
    # grid: (M); one binary-search step from the per-row total count.
    # FIRST only materializes the initial mid.
    pid = ext.program_id(0)
    lo_v = tl.load(lo + pid)
    hi_v = tl.load(hi + pid)
    if KEY_BITS == 64:
        mid = lo_v + ((hi_v - lo_v) // 2)
    else:
        mid = ((lo_v.to(tl.int64) + hi_v.to(tl.int64)) // 2).to(tl.uint32)
    if not FIRST:
        le = tl.load(counts + pid)
        go_left = le > TARGET
        active = lo_v < hi_v
        new_hi = tl.where(go_left & active, mid, hi_v)
        new_lo = tl.where(~go_left & active, mid + 1, lo_v)
        tl.store(lo + pid, new_lo)
        tl.store(hi + pid, new_hi)
    tl.store(mid_buf + pid, mid)


@libentry()
@triton.jit
def median_count_partial_kernel(
    keys,
    mid,
    chunk_counts,
    N,
    NCHUNK,
    PARTIAL,
    BLOCK_R: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    TAIL_BASE: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # grid: (M); count keys <= mid within the leftover tail slice.
    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_P)
    mask = cols < PARTIAL
    keys_v = tl.load(keys + pid * ROW_STRIDE + TAIL_BASE + cols, mask=mask, other=0)
    mid_v = tl.load(mid + pid)
    le = tl.sum((mask & (keys_v <= mid_v)).to(tl.int32), axis=0)
    tl.store(chunk_counts + pid * BLOCK_R + NCHUNK - 1, le)


@libentry()
@triton.jit
def median_set_scalar_kernel(buf, idx, val):
    tl.store(buf + idx, val)


@libentry()
@triton.jit
def median_select_chunk_kernel(
    keybuf,
    sel_keys,
    chunk_first,
    N,
    NCHUNK,
    BLOCK_R: tl.constexpr,
    CHUNK: tl.constexpr,
):
    # grid: M * NCHUNK; find the earliest matching key within one slice.
    # chunk_first is encoded as the GLOBAL index (chunk*CHUNK + local),
    # or 0x7FFFFFFF when the slice has no match.
    pid = ext.program_id(0)
    chunk = pid % NCHUNK
    cols = tl.arange(0, CHUNK)
    offsets = chunk * CHUNK + cols
    sel = tl.load(sel_keys + pid // NCHUNK)
    keys_v = tl.load(keybuf + pid * CHUNK + cols)
    km = keys_v == sel
    first = tl.min(tl.where(km, cols, CHUNK), axis=0)
    pack = tl.where(first < CHUNK, first + chunk * CHUNK, 2147483647)
    tl.store(chunk_first + pid // NCHUNK * BLOCK_R + chunk, pack)


@libentry()
@triton.jit
def median_select_partial_kernel(
    keybuf,
    sel_keys,
    chunk_first,
    N,
    NCHUNK,
    START,
    PARTIAL,
    BLOCK_R: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    TAIL_BASE: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # grid: (M); find the earliest matching key within the tail slice.
    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_P)
    mask = cols < PARTIAL
    sel = tl.load(sel_keys + pid)
    keys_v = tl.load(keybuf + pid * ROW_STRIDE + TAIL_BASE + cols, mask=mask, other=0)
    km = mask & (keys_v == sel)
    first = tl.min(tl.where(km, cols, BLOCK_P), axis=0)
    pack = tl.where(first < BLOCK_P, first + START, 2147483647)
    tl.store(chunk_first + pid * BLOCK_R + NCHUNK - 1, pack)


@libentry()
@triton.jit
def median_merge_select_chunk_kernel(
    inp,
    chunk_first,
    chunk_nan,
    row_nan_first,
    out_values,
    out_indices,
    N,
    NCHUNK,
    USE_ROW_NAN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    # grid: (M); pick the earliest matching key over all slices, and for
    # float rows the earliest NaN position.  chunk_first/chunk_nan hold
    # GLOBAL positions (chunk*CHUNK + local) or CHUNK when absent, so the
    # row-wide minimum directly yields the requested index.
    pid = ext.program_id(0)
    cc = tl.arange(0, BLOCK_N)
    mask = cc < NCHUNK
    cf = tl.load(chunk_first + pid * BLOCK_N + cc, mask=mask, other=2147483647)
    cn = tl.load(chunk_nan + pid * BLOCK_N + cc, mask=mask, other=2147483647)
    is_float: tl.constexpr = inp.dtype.element_ty.is_floating()
    has_nan = tl.max((cn < 2147483647).to(tl.int32), axis=0) != 0
    nan_best = tl.min(cn, axis=0)
    if USE_ROW_NAN:
        row_first = tl.load(row_nan_first + pid)
        has_nan = has_nan | (row_first >= 0)
        nan_best = tl.minimum(nan_best, tl.where(row_first >= 0, row_first, 0x7FFFFFFF))
    match_best = tl.min(cf, axis=0)
    best = tl.where(has_nan, nan_best, match_best)
    ridx = tl.minimum(best, N - 1)
    if is_float:
        rval = tl.load(inp + pid * N + ridx, mask=ridx < N, other=0.0)
        rval = tl.where(has_nan, float("nan"), rval)
    else:
        rval = tl.load(inp + pid * N + ridx, mask=ridx < N, other=0)
    tl.store(out_values + pid, rval)
    tl.store(out_indices + pid, ridx.to(tl.int64))


@libentry()
@triton.jit
def nanmedian_flat_keysel_kernel(
    orig,
    keys,
    out_values,
    out_indices,
    TARGET,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KEY_BITS: tl.constexpr,
):
    # Flat nanmedian key-select (single-block path, N <= KEY_BLOCK_LIMIT).
    # `keys` holds host pre-keyed lanes (floats: NaN already replaced by
    # +inf before keying, so NaN lanes sit at the top of the key range and
    # are never counted before rank `TARGET`); the binary search runs fully
    # in-register on uint32/uint64 keys; pad lanes are re-sanitized to the
    # all-ones sentinel because XPU masked loads ignore `other` and leak
    # neighboring memory (pad is thus inert for counts and selects).
    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    pad = cols >= N
    if KEY_BITS == 64:
        kv = tl.load(keys + pid * N + cols, mask=~pad, other=0xFFFFFFFFFFFFFFFF)
        kv = tl.where(pad, 0xFFFFFFFFFFFFFFFF, kv)
    else:
        kv = tl.load(keys + pid * N + cols, mask=~pad, other=0xFFFFFFFF)
        kv = tl.where(pad, 0xFFFFFFFF, kv)
    lo = tl.min(kv, axis=0)
    hi = tl.max(kv, axis=0)
    for _ in tl.range(0, KEY_BITS):
        if KEY_BITS == 64:
            mid = lo + ((hi - lo) // 2)
        else:
            mid = ((lo.to(tl.int64) + hi.to(tl.int64)) // 2).to(tl.uint32)
        le = tl.sum((kv <= mid).to(tl.int32), axis=0)
        go_left = le > TARGET
        active = lo < hi
        hi = tl.where(go_left & active, mid, hi)
        lo = tl.where(~go_left & active, mid + 1, lo)
    km = (~pad) & (kv == lo)
    first = tl.min(tl.where(km, cols, BLOCK_N), axis=0)
    ridx = tl.minimum(first, N - 1)
    rval = tl.load(orig + pid * N + ridx, mask=ridx < N, other=0)
    tl.store(out_values + pid, rval)
    tl.store(out_indices + pid, ridx.to(tl.int64))


def _nanmedian_prekey(flat, key_bits):
    """Pre-order a flat vector into ascending uint32/uint64 keys on the host.

    Floats get NaN replaced by +inf first (so NaN keys land at the very top
    of the real key range and stay inert below rank TARGET), then the
    IEEE-style sign-flip preorder.  Signed ints get the two's-complement
    sign-bit flip; uint8 stays in plain rank order.

    Note: on the Kunlunxin XPU runtime float64 tensors are physically stored
    as 32-bit elements (torch reports float64 but storage is 4B/element), so
    float64 rows are pre-keyed with 32-bit keys (KEY_BITS=32) and the kernel
    value loads run on the converted fp32 work tensor.  int64 is stored
    natively and keeps 64-bit keys.
    """
    dt = flat.dtype
    if key_bits == 64:
        bits = flat.view(torch.int64)
        return (bits ^ (-(1 << 63))).view(torch.uint64)
    if dt in (torch.float16, torch.bfloat16):
        w = flat.to(torch.float32)
    elif dt in (torch.int8, torch.uint8, torch.int16):
        w = flat.to(torch.int32)
    else:
        w = flat
    if w.dtype.is_floating_point:
        w = w.nan_to_num(nan=float("inf"))
        b = w.contiguous().view(torch.int32)
        return (b ^ (-(1 << 31) | (b >> 31))).view(torch.uint32)
    if dt == torch.uint8:
        return w.view(torch.uint32)
    return (w.view(torch.int32) ^ (-(1 << 31))).view(torch.uint32)


def _nanmedian_valid_target(flat, N):
    if flat.dtype.is_floating_point:
        nan_cnt = int(torch.isnan(flat).sum().item())
    else:
        nan_cnt = 0
    valid = N - nan_cnt
    return max((valid - 1) // 2, 0)


def _nanmedian_flat_single(flat, N, key_bits):
    pre = _nanmedian_prekey(flat, key_bits)
    target = _nanmedian_valid_target(flat, N)
    block_n = max(64, triton.next_power_of_2(N))
    values = torch.empty((1,), dtype=flat.dtype, device=flat.device)
    indices = torch.empty((1,), dtype=torch.long, device=flat.device)
    with torch_device_fn.device(flat.device):
        nanmedian_flat_keysel_kernel[(1,)](
            flat,
            pre,
            values,
            indices,
            target,
            N,
            block_n,
            key_bits,
            num_warps=4,
            num_stages=1,
            buffer_size_limit=2048,
        )
    return values, indices


def _nanmedian_flat_chunked(flat, N, key_bits):
    """Key-sort select for one flat row wider than KEY_BLOCK_LIMIT lanes.

    The same chunks + unpadded tail + host-side binary-search pipeline as
    the shared median chunked path (kernels reused from `median`); the only
    nanmedian delta is the target: (valid-1)//2 over non-NaN entries, with
    NaN lanes carrying the +inf (top-ranked) key so they never satisfy the
    counts or the selection below TARGET (all-NaN rows converge on the
    first +inf lane, i.e. index 0 / NaN, matching torch semantics).
    """
    M = 1
    CHUNK = KEY_BLOCK_LIMIT
    nfull = N // CHUNK
    partial = N - nfull * CHUNK
    nall = nfull + (1 if partial else 0)
    nchunks = nall
    if nchunks > KEY_BLOCK_LIMIT:
        raise NotImplementedError(
            f"nanmedian reduction width {N} exceeds Kunlunxin limit"
        )
    tail_start = nfull * CHUNK
    pre = _nanmedian_prekey(flat, key_bits)
    key_dtype = torch.uint64 if key_bits == 64 else torch.uint32
    keybuf = torch.full(
        (M * nall, CHUNK),
        -1,
        device=flat.device,
    ).view(key_dtype)
    reduce_block = triton.next_power_of_2(nall)
    chunk_mins = torch.empty((M * reduce_block,), dtype=key_dtype, device=flat.device)
    chunk_maxs = torch.empty((M * reduce_block,), dtype=key_dtype, device=flat.device)
    chunk_nan = torch.full(
        (M * reduce_block,), 2147483647, dtype=torch.int32, device=flat.device
    )
    chunk_counts = torch.empty((M * reduce_block,), dtype=torch.int32, device=flat.device)
    counts = torch.empty((M,), dtype=torch.int32, device=flat.device)
    lo = torch.empty((M,), dtype=key_dtype, device=flat.device)
    hi = torch.empty((M,), dtype=key_dtype, device=flat.device)
    mid = torch.empty((M,), dtype=key_dtype, device=flat.device)
    tail_blk = triton.next_power_of_2(partial) if partial else 1
    row_stride = nall * CHUNK
    tail_base = (nall - 1) * CHUNK
    out_values = torch.empty((M,), dtype=flat.dtype, device=flat.device)
    out_indices = torch.empty((M,), dtype=torch.long, device=flat.device)
    with torch_device_fn.device(flat.device):
        nan_kw = dict(num_warps=4, num_stages=1, buffer_size_limit=2048)
        if nfull:
            median_key_info_chunk_kernel[(M * nfull,)](
                pre,
                keybuf,
                chunk_mins,
                chunk_maxs,
                chunk_nan,
                N,
                nchunks,
                reduce_block,
                CHUNK,
                key_bits,
                True,
                **nan_kw,
            )
        if partial:
            median_key_info_partial_kernel[(M,)](
                pre,
                keybuf,
                chunk_mins,
                chunk_maxs,
                chunk_nan,
                N,
                nchunks,
                tail_start,
                partial,
                reduce_block,
                row_stride,
                tail_base,
                tail_blk,
                key_bits,
                True,
                **nan_kw,
            )
        max_key = (1 << key_bits) - 1
        median_row_reduce_kernel[(M,)](
            chunk_mins, lo, nchunks, max_key, reduce_block, 0, **nan_kw
        )
        median_row_reduce_kernel[(M,)](
            chunk_maxs, hi, nchunks, 0, reduce_block, 1, **nan_kw
        )
        target = _nanmedian_valid_target(flat, N)

        def _u64_of(t, i):
            if key_bits == 64:
                return int(t.view(torch.int64)[i].item()) & 0xFFFFFFFFFFFFFFFF
            return int(np.asarray(np.float32(t.view(torch.float32)[i].item())).view(np.uint32))

        lo_h = [_u64_of(lo, r) for r in range(M)]
        hi_h = [_u64_of(hi, r) for r in range(M)]
        for _ in range(key_bits + 6):
            mid_h = [(l + h) // 2 for l, h in zip(lo_h, hi_h)]
            if key_bits == 64:
                mid_h = [m & 0xFFFFFFFFFFFFFFFF for m in mid_h]
            for r in range(M):
                median_set_scalar_kernel[(1,)](mid, r, mid_h[r], **nan_kw)
            if nfull:
                median_count_chunk_kernel[(M * nfull,)](
                    keybuf, mid, chunk_counts, N, nchunks, reduce_block, CHUNK, **nan_kw
                )
            if partial:
                median_count_partial_kernel[(M,)](
                    keybuf, mid, chunk_counts, N, nchunks, partial,
                    reduce_block, row_stride, tail_base, tail_blk, **nan_kw
                )
            median_row_reduce_kernel[(M,)](
                chunk_counts, counts, nchunks, 0, reduce_block, 2, **nan_kw
            )
            for r in range(M):
                if int(counts[r].item()) > target:
                    hi_h[r] = mid_h[r]
                else:
                    lo_h[r] = mid_h[r] + 1
        sel_keys = lo
        for r in range(M):
            median_set_scalar_kernel[(1,)](sel_keys, r, lo_h[r], **nan_kw)
        chunk_first = torch.empty((M * reduce_block,), dtype=torch.int32, device=flat.device)
        if nfull:
            median_select_chunk_kernel[(M * nfull,)](
                keybuf, sel_keys, chunk_first, N, nchunks, reduce_block, CHUNK, **nan_kw
            )
        if partial:
            median_select_partial_kernel[(M,)](
                keybuf, sel_keys, chunk_first, N, nchunks, tail_start, partial,
                reduce_block, row_stride, tail_base, tail_blk, **nan_kw
            )
        median_merge_select_chunk_kernel[(M,)](
            flat,
            chunk_first,
            chunk_nan,
            chunk_nan,
            out_values,
            out_indices,
            N,
            nchunks,
            False,
            reduce_block,
            CHUNK,
            **nan_kw,
        )
    return out_values, out_indices


# ---------------------------------------------------------------------------
# Fast flat nanmedian pipeline (Kunlunxin, 32-bit-keyable dtypes).
#
# Differences against the legacy chunked pipeline kept below/above:
#  * keys are produced inside the kernels, so the host no longer materializes
#    an N-element uint32 pre-key tensor (that chain was 3-5 full gems
#    elementwise passes and, for fp16/bf16/int8/int16/uint8, it also doubled or
#    quadrupled the bytes every later pass had to re-read);
#  * no `keybuf` scratch at all - count/select recompute the key from the raw
#    input, which keeps the hot bisection passes at the input's native element
#    width;
#  * zero `other=` fills: full tiles are exactly in bounds, the tail tile is
#    re-based to `N - CHUNK` so every lane is a legal address and the overlap
#    is removed by a lane predicate, and the tiny fold tiles use a clamped
#    address plus an explicit `tl.where` gate (masked `other=` silently
#    pollutes live lanes on this backend);
#  * the NaN count and the binary-search state live on the device, so a whole
#    reduction needs exactly one host synchronization instead of one per
#    bisection step;
#  * narrow dtypes keep narrow keys (fp16/bf16 -> 16 bit, int8/uint8 -> 8 bit),
#    and the step budget is derived from the observed key span, so int inputs
#    finish in ~7-8 steps and fp16/bf16 in ~16 instead of a fixed 38.
# ---------------------------------------------------------------------------
NMFLAT_CHUNK = 32768
# A *clamped* (per-lane gather) load faults on this backend once the tile grows
# past 64 KiB - measured: float32 with 32768 lanes (128 KiB) raises an illegal
# memory access, 16384 lanes (64 KiB) and float16 with 32768 lanes (64 KiB) are
# fine.  The fused single-tile path is the only place that needs a clamp, so it
# is capped there; plain stride-1 tiles are fine at 128 KiB, which keeps the
# chunked tile at NMFLAT_CHUNK elements for every dtype.
NMFLAT_SINGLE_BYTES = 65536
NMFLAT_MAX_KEY = tl.constexpr(0xFFFFFFFF)
NMFLAT_IDX_SENTINEL = tl.constexpr(2147483647)
NMFLAT_MAX_STEPS = 34


@triton.jit
def _nmflat_keys(vals):
    """Order-preserving unsigned key for one value tile, widened to uint32.

    Narrow dtypes keep their native key width (float16 -> 16 bit, int8/uint8
    -> 8 bit, int16 -> 16 bit), which is what lets the bisection finish in far
    fewer than 32 steps.  bfloat16 is the exception: it is upcast to float32
    and therefore spreads over the full 32-bit key space.  Deriving a 16-bit
    bf16 key by bitcasting to int16/uint16/float16 was measured to abort
    lowering on this backend (`PassManager::run failed`, reported as
    `out of resource: uni_sram`), so the proven upcast is kept.
    """
    return convert_to_uint_preverse_order(vals, False).to(tl.uint32)


@triton.jit
def _nmflat_gated_keys(vals, inb):
    """Keys with NaN lanes forced to the top of the key domain.

    NaN must outrank every real value (including +inf) so it is never selected
    below rank TARGET; the raw float preorder puts *negative* NaN at the very
    bottom instead, which is why the remap is explicit here.  Returns
    (keys, valid) where `valid` is "in bounds and not NaN".
    """
    keys = _nmflat_keys(vals)
    if vals.dtype.is_floating():
        nan = _median_is_nan(vals)
        keys = tl.where(nan, NMFLAT_MAX_KEY, keys)
        valid = inb & (~nan)
    else:
        valid = inb
    return keys, valid


@libentry()
@triton.jit
def nmflat_stats_kernel(
    inp,
    part_min,
    part_max,
    part_nan,
    CHUNK: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    # grid: (NFULL,); one fully in-bounds CHUNK tile per program.
    pid = ext.program_id(0)
    cols = tl.arange(0, CHUNK)
    vals = tl.load(inp + pid * CHUNK + cols)
    keys, valid = _nmflat_gated_keys(vals, cols >= 0)
    tl.store(part_min + pid, tl.min(tl.where(valid, keys, NMFLAT_MAX_KEY), axis=0))
    tl.store(part_max + pid, tl.max(tl.where(valid, keys, 0), axis=0))
    if IS_FLOAT:
        tl.store(part_nan + pid, tl.sum((~valid).to(tl.int32), axis=0))


@libentry()
@triton.jit
def nmflat_stats_tail_kernel(
    tail_inp,
    part_min,
    part_max,
    part_nan,
    SLOT,
    PARTIAL,
    CHUNK: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    # grid: (1,); the leftover lanes live in a freshly allocated CHUNK-wide
    # scratch tile, so the load is an aligned, fully in-bounds, stride-1 tile
    # with no mask, no `other=` and no address clamping (both of those fault or
    # silently corrupt on this backend); `PARTIAL` is only ever a *lane*
    # predicate, never part of an address.
    cols = tl.arange(0, CHUNK)
    inb = cols < PARTIAL
    vals = tl.load(tail_inp + cols)
    keys, valid = _nmflat_gated_keys(vals, inb)
    tl.store(part_min + SLOT, tl.min(tl.where(valid, keys, NMFLAT_MAX_KEY), axis=0))
    tl.store(part_max + SLOT, tl.max(tl.where(valid, keys, 0), axis=0))
    if IS_FLOAT:
        tl.store(part_nan + SLOT, tl.sum((inb & (~valid)).to(tl.int32), axis=0))


@libentry()
@triton.jit
def nmflat_fold_stats_kernel(
    part_min,
    part_max,
    part_nan,
    lo_buf,
    hi_buf,
    mid_buf,
    info,
    N,
    BLOCK_R: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    # grid: (1,); folds the per-tile partials and seeds the search state.
    # The partial buffers are sentinel-filled by the wrapper and sized exactly
    # BLOCK_R, so this is a plain in-bounds stride-1 tile: no mask, no `other=`
    # and no gather.
    cols = tl.arange(0, BLOCK_R)
    lo = tl.min(tl.load(part_min + cols), axis=0)
    hi = tl.max(tl.load(part_max + cols), axis=0)
    if IS_FLOAT:
        nan_total = tl.sum(tl.load(part_nan + cols).to(tl.int32), axis=0)
    else:
        nan_total = 0
    all_nan = (N - nan_total) <= 0
    lo = tl.where(all_nan, NMFLAT_MAX_KEY, lo)
    hi = tl.where(all_nan, NMFLAT_MAX_KEY, hi)
    tl.store(lo_buf + 0, lo)
    tl.store(hi_buf + 0, hi)
    tl.store(mid_buf + 0, ((lo.to(tl.int64) + hi.to(tl.int64)) // 2).to(tl.uint32))
    slots = tl.arange(0, 64)
    packed = tl.where(
        slots == 0,
        lo.to(tl.int64),
        tl.where(slots == 1, hi.to(tl.int64), nan_total.to(tl.int64)),
    )
    tl.store(info + slots, packed)


@libentry()
@triton.jit
def nmflat_count_kernel(inp, mid_buf, part_cnt, CHUNK: tl.constexpr):
    # grid: (NFULL,); count keys <= mid inside one fully in-bounds tile.
    pid = ext.program_id(0)
    cols = tl.arange(0, CHUNK)
    vals = tl.load(inp + pid * CHUNK + cols)
    keys, _ = _nmflat_gated_keys(vals, cols >= 0)
    mid = tl.load(mid_buf + 0)
    tl.store(part_cnt + pid, tl.sum((keys <= mid).to(tl.int32), axis=0))


@libentry()
@triton.jit
def nmflat_count_tail_kernel(
    tail_inp, mid_buf, part_cnt, SLOT, PARTIAL, CHUNK: tl.constexpr
):
    # grid: (1,); same scratch tile as above.
    cols = tl.arange(0, CHUNK)
    inb = cols < PARTIAL
    vals = tl.load(tail_inp + cols)
    keys, _ = _nmflat_gated_keys(vals, inb)
    mid = tl.load(mid_buf + 0)
    tl.store(part_cnt + SLOT, tl.sum((inb & (keys <= mid)).to(tl.int32), axis=0))


@libentry()
@triton.jit
def nmflat_step_kernel(
    part_cnt,
    lo_buf,
    hi_buf,
    mid_buf,
    BLOCK_R: tl.constexpr,
    TARGET: tl.constexpr,
):
    # grid: (1,); one device-resident bisection step (no host round trip).
    # TARGET is a constexpr on purpose: an extra runtime scalar argument is a
    # measured performance cliff on this backend.
    cols = tl.arange(0, BLOCK_R)
    total = tl.sum(tl.load(part_cnt + cols).to(tl.int32), axis=0)
    lo = tl.load(lo_buf + 0)
    hi = tl.load(hi_buf + 0)
    mid = tl.load(mid_buf + 0)
    go_left = total > TARGET
    active = lo < hi
    new_hi = tl.where(go_left & active, mid, hi)
    new_lo = tl.where((~go_left) & active, mid + 1, lo)
    tl.store(lo_buf + 0, new_lo)
    tl.store(hi_buf + 0, new_hi)
    tl.store(
        mid_buf + 0, ((new_lo.to(tl.int64) + new_hi.to(tl.int64)) // 2).to(tl.uint32)
    )


@libentry()
@triton.jit
def nmflat_select_kernel(inp, sel_buf, part_first, CHUNK: tl.constexpr):
    # grid: (NFULL,); earliest global index whose key equals the selected key.
    pid = ext.program_id(0)
    cols = tl.arange(0, CHUNK)
    off = pid * CHUNK + cols
    vals = tl.load(inp + off)
    keys, _ = _nmflat_gated_keys(vals, cols >= 0)
    sel = tl.load(sel_buf + 0)
    hit = keys == sel
    tl.store(
        part_first + pid,
        tl.min(tl.where(hit, off, NMFLAT_IDX_SENTINEL), axis=0),
    )


@libentry()
@triton.jit
def nmflat_select_tail_kernel(
    tail_inp, sel_buf, part_first, TAIL_START, SLOT, PARTIAL, CHUNK: tl.constexpr
):
    # grid: (1,); same scratch tile as above.  TAIL_START only shifts the
    # reported *index*, it is not part of any load address.
    cols = tl.arange(0, CHUNK)
    inb = cols < PARTIAL
    vals = tl.load(tail_inp + cols)
    keys, _ = _nmflat_gated_keys(vals, inb)
    sel = tl.load(sel_buf + 0)
    hit = inb & (keys == sel)
    tl.store(
        part_first + SLOT,
        tl.min(tl.where(hit, TAIL_START + cols, NMFLAT_IDX_SENTINEL), axis=0),
    )


@libentry()
@triton.jit
def nmflat_finish_kernel(
    inp, part_first, out_values, out_indices, N, BLOCK_R: tl.constexpr
):
    # grid: (1,); fold the per-tile winners and fetch the value.
    cols = tl.arange(0, BLOCK_R)
    best = tl.min(tl.load(part_first + cols), axis=0)
    ridx = tl.minimum(best, N - 1)
    tl.store(out_values + 0, tl.load(inp + ridx))
    tl.store(out_indices + 0, ridx.to(tl.int64))


@libentry()
@triton.jit
def nmflat_single_kernel(
    inp, out_values, out_indices, N, BLOCK_N: tl.constexpr, KEY_BITS: tl.constexpr
):
    # grid: (1,); whole reduction in registers for N <= NMFLAT_CHUNK.  Keys,
    # the NaN count, the rank target and the bisection all stay on device, so
    # this replaces ~9 host-side launches plus one device synchronization with
    # a single launch and no synchronization at all.
    cols = tl.arange(0, BLOCK_N)
    inb = cols < N
    vals = tl.load(inp + tl.minimum(cols, N - 1))
    keys, valid = _nmflat_gated_keys(vals, inb)
    nvalid = tl.sum(valid.to(tl.int32), axis=0)
    target = tl.maximum((nvalid - 1) // 2, 0)
    all_nan = nvalid <= 0
    lo = tl.where(
        all_nan, NMFLAT_MAX_KEY, tl.min(tl.where(valid, keys, NMFLAT_MAX_KEY), axis=0)
    )
    hi = tl.where(all_nan, NMFLAT_MAX_KEY, tl.max(tl.where(valid, keys, 0), axis=0))
    for _ in tl.range(0, KEY_BITS):
        mid = ((lo.to(tl.int64) + hi.to(tl.int64)) // 2).to(tl.uint32)
        le = tl.sum((inb & (keys <= mid)).to(tl.int32), axis=0)
        go_left = le > target
        active = lo < hi
        hi = tl.where(go_left & active, mid, hi)
        lo = tl.where((~go_left) & active, mid + 1, lo)
    hit = inb & (keys == lo)
    ridx = tl.minimum(tl.min(tl.where(hit, cols, BLOCK_N), axis=0), N - 1)
    tl.store(out_values + 0, tl.load(inp + ridx))
    tl.store(out_indices + 0, ridx.to(tl.int64))


def _nmflat_launch_kw():
    return dict(num_warps=4, num_stages=1, buffer_size_limit=2048)


def _nmflat_key_bits(dtype):
    """Width of the key domain produced by `_nmflat_keys` for this dtype.

    Only used to size the in-register bisection of the fused single-tile
    kernel: the keys live in [0, 2**bits), so `bits` steps are always enough.
    bfloat16 is 32 because `convert_to_uint_preverse_order` upcasts it.
    """
    if dtype == torch.bfloat16:
        return 32
    return 8 * torch._utils._element_size(dtype)


def _nmflat_single_block(N, itemsize):
    block_n = max(64, triton.next_power_of_2(N))
    if block_n > NMFLAT_CHUNK or block_n * itemsize > NMFLAT_SINGLE_BYTES:
        return None
    return block_n


def _nmflat_single(flat, N, block_n, out_values=None):
    # `out_values` lets the caller hand in a user-visible 1-element destination
    # (the `nanmedian.out` fast path) so the scalar result is committed by the
    # kernel itself and the extra write-back launch disappears.  Verified with a
    # three-level sentinel probe (immediate neighbours / 640-fp32 over-allocated
    # prefix view / independent allocator neighbours): 35/35 dtype x N cases
    # clean, i.e. this store commits exactly one element.
    if out_values is None:
        values = torch.empty((1,), dtype=flat.dtype, device=flat.device)
    else:
        values = out_values
    indices = torch.empty((1,), dtype=torch.long, device=flat.device)
    with torch_device_fn.device(flat.device):
        nmflat_single_kernel[(1,)](
            flat,
            values,
            indices,
            N,
            block_n,
            _nmflat_key_bits(flat.dtype),
            **_nmflat_launch_kw(),
        )
    return values, indices


def _nmflat_chunked(flat, N):
    CHUNK = NMFLAT_CHUNK
    nfull = N // CHUNK
    partial = N - nfull * CHUNK
    nall = max(1, nfull + (1 if partial else 0))
    if nall > CHUNK:
        raise NotImplementedError(
            f"nanmedian reduction width {N} exceeds Kunlunxin limit"
        )
    tail_start = nfull * CHUNK
    slot = nfull
    block_r = max(64, triton.next_power_of_2(nall))
    dev = flat.device
    is_float = flat.dtype.is_floating_point

    tailbuf = None
    if partial:
        # One aligned CHUNK-wide scratch tile for the leftover lanes, filled by
        # the native ATen strided-copy primitive (`_copy_from`); gems only
        # overrides `copy_`/`copy`, so this stays off the slow gems copy path.
        # The lanes past `partial` keep uninitialized data on purpose - every
        # consumer gates them out with the `cols < PARTIAL` lane predicate.
        tailbuf = torch.empty((CHUNK,), dtype=flat.dtype, device=dev)
        torch.ops.aten._copy_from(flat[tail_start:], tailbuf[:partial], False)

    # +64 slots of head room: a scalar store can commit a full 64-element
    # vector on this backend, and the tail tile writes at index nall - 1.
    pad_r = block_r + 64
    part_min = torch.full((pad_r,), -1, dtype=torch.int32, device=dev).view(torch.uint32)
    part_max = torch.zeros((pad_r,), dtype=torch.int32, device=dev).view(torch.uint32)
    part_nan = torch.zeros((pad_r,), dtype=torch.int32, device=dev)
    part_cnt = torch.zeros((pad_r,), dtype=torch.int32, device=dev)
    part_first = torch.full(
        (pad_r,), NMFLAT_IDX_SENTINEL.value, dtype=torch.int32, device=dev
    )
    state = torch.empty((3, 64), dtype=torch.int32, device=dev).view(torch.uint32)
    lo_buf, hi_buf, mid_buf = state[0], state[1], state[2]
    info = torch.empty((64,), dtype=torch.int64, device=dev)
    values = torch.empty((1,), dtype=flat.dtype, device=dev)
    indices = torch.empty((1,), dtype=torch.long, device=dev)

    kw = _nmflat_launch_kw()
    with torch_device_fn.device(dev):
        if nfull:
            nmflat_stats_kernel[(nfull,)](
                flat, part_min, part_max, part_nan, CHUNK, is_float, **kw
            )
        if partial:
            nmflat_stats_tail_kernel[(1,)](
                tailbuf,
                part_min,
                part_max,
                part_nan,
                slot,
                partial,
                CHUNK,
                is_float,
                **kw,
            )
        nmflat_fold_stats_kernel[(1,)](
            part_min,
            part_max,
            part_nan,
            lo_buf,
            hi_buf,
            mid_buf,
            info,
            N,
            block_r,
            is_float,
            **kw,
        )
        # The one and only host synchronization of the whole reduction.
        lo0, hi0, nan_total = (int(v) for v in info.cpu()[:3].tolist())
        lo0 &= 0xFFFFFFFF
        hi0 &= 0xFFFFFFFF
        if N - nan_total <= 0:
            steps = 0
        else:
            steps = min(NMFLAT_MAX_STEPS, max(1, (hi0 - lo0).bit_length() + 1))
        target = max((N - nan_total - 1) // 2, 0)
        for _ in range(steps):
            if nfull:
                nmflat_count_kernel[(nfull,)](flat, mid_buf, part_cnt, CHUNK, **kw)
            if partial:
                nmflat_count_tail_kernel[(1,)](
                    tailbuf, mid_buf, part_cnt, slot, partial, CHUNK, **kw
                )
            nmflat_step_kernel[(1,)](
                part_cnt, lo_buf, hi_buf, mid_buf, block_r, target, **kw
            )
        if nfull:
            nmflat_select_kernel[(nfull,)](flat, lo_buf, part_first, CHUNK, **kw)
        if partial:
            nmflat_select_tail_kernel[(1,)](
                tailbuf, lo_buf, part_first, tail_start, slot, partial, CHUNK, **kw
            )
        nmflat_finish_kernel[(1,)](flat, part_first, values, indices, N, block_r, **kw)
    return values, indices


def _nanmedian_out_meta_gate(inp, out):
    """Replay the ATen structured-kernel `out=` meta checks for `nanmedian.out`.

    FlagGems binds its kernels to the *Autograd* dispatch key, which sits above
    ATen's structured-kernel wrappers, so none of the `out=` validation that
    `aten::nanmedian.out` normally performs ever runs.  Measured divergences
    against the CPU ATen oracle before this gate existed (probe
    `p2_gems_out_semantics_BASE.log`):

    * `out` with a shape other than `()` - ATen calls `resize_output()`, shrinks
      `out` to `()` in place (keeping its storage offset) and writes exactly one
      element.  The bare `out.copy_(result)` used before instead *broadcast* the
      scalar over every element of `out`: with `out = big[4:8]` inside a
      `(12,)` sentinel buffer, indices 5/6/7 were silently overwritten
      (reproduced for float32/float16/int16/uint8), and `out = torch.full((4,33))`
      had all 132 elements written.  It also returned a non-0-dim tensor.
    * `out` with a dtype different from the input - ATen raises
      ``RuntimeError: Expected out tensor to have dtype ...``; the old path
      silently cast instead.
    """
    if out.dtype != inp.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {inp.dtype}, "
            f"but got {out.dtype} instead"
        )
    if out.dim() != 0:
        # ATen's `resize_output()` shrinks `out` to `()` in place and keeps its
        # storage offset (CPU oracle: `big[4:8]` stays at offset 4 and only
        # `big[4]` is written).  `Tensor.resize_` cannot be used here: FlagGems
        # overrides `aten::resize_` (see `_FULL_CONFIG`) and that override resets
        # the storage offset to 0, which would move the write to `big[0]`.
        # `as_strided_` is a pure metadata operation, is not a gems op, and
        # preserves the offset.
        if (
            out.storage_offset() + 1
        ) * out.element_size() <= out.untyped_storage().nbytes():
            out.as_strided_((), ())
        else:
            # Nothing is allocated at the current offset (e.g. a freshly created
            # `torch.empty((0,))`), so the metadata-only path would write out of
            # bounds; this needs a real (re)allocation.  ATen grows the storage
            # to one element here too (CPU oracle S13).
            out.resize_(())


def _nanmedian_flat_impl(inp, out=None):
    if out is not None:
        _nanmedian_out_meta_gate(inp, out)
    if inp.numel() == 0:
        result = _empty_flat_value(inp)
        if out is not None:
            torch.ops.aten._copy_from(result, out, False)
            return out
        return result

    flat = inp.reshape(-1).contiguous()
    return_fp64 = flat.dtype == torch.float64
    if return_fp64:
        # Kunlunxin XPU stores float64 tensors with 32-bit elements (the
        # dtype reports float64 but the storage is 4B/element), so compute
        # on the fp32 work tensor and re-cast the scalar result.
        flat = flat.to(torch.float32)
    N = flat.numel()
    key_bits = 64 if flat.dtype == torch.int64 else 32
    # The meta gate above already asserts `out.dtype == inp.dtype`, so for the
    # fused single-tile pipeline the kernel can commit its scalar straight into
    # `out` and the write-back disappears entirely.  The other pipelines keep an
    # explicit copy (`aten::_copy_from`, the native strided-copy primitive -
    # `Tensor.copy_` is a gems op and costs 14.1 us against 5.0 us here).
    dest = out.view(1) if (out is not None and not return_fp64) else None
    if key_bits == 64:
        # int64 keeps the legacy host-pre-keyed pipeline (64-bit keys are not
        # covered by the uint32 fast path and int64 is not part of any
        # benchmark matrix).
        if N <= KEY_BLOCK_LIMIT:
            values, _ = _nanmedian_flat_single(flat, N, key_bits)
        else:
            values, _ = _nanmedian_flat_chunked(flat, N, key_bits)
    else:
        block_n = _nmflat_single_block(N, flat.element_size())
        if block_n is not None:
            values, _ = _nmflat_single(flat, N, block_n, out_values=dest)
            if dest is not None:
                return out
        else:
            values, _ = _nmflat_chunked(flat, N)
    result = values.reshape(())
    if return_fp64:
        result = result.to(torch.float64)
    if out is not None:
        torch.ops.aten._copy_from(result, out, False)
        return out
    return result



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


# ---------------------------------------------------------------------------
# Fast dim-wise nanmedian pipeline (Kunlunxin).
#
# The legacy dim path kept above (`nanmedian_direct_select_kernel` for
# N <= MAX_BLOCK_N and `nanmedian_float_key_select_kernel` beyond it) is wrong
# and, for some shapes, fatal on this backend:
#  * every one of its tile loads is a masked load with `other=`, which is not
#    honoured here - it silently pollutes live lanes.  That is what breaks
#    `(2, 3, 1031)` (a partial tail tile) for all dtypes: the per-pass "count
#    keys <= mid" reductions disagree between passes, the bisection converges
#    to a key that no lane owns, and the select pass returns the `N` sentinel,
#    so the reported index is out of range and the value is read out of bounds;
#  * `nanmedian_direct_select_kernel` runs a `tl.static_range(0, BLOCK_N)`
#    selection-sort over the tile.  With int64 tiles, and with BLOCK_N == N
#    (i.e. a power-of-two reduction width such as (64, 64)), it aborts with
#    `KL_XID_KERNEL_EXCEPTION` / status 719 and destroys the CUDA context.
#
# The replacement mirrors the flat `nmflat_*` pipeline that is already proven
# on this backend: one program per reduction row, order-preserving unsigned
# keys computed inside the kernel, a register-resident binary search over the
# key domain, and zero `other=` fills.  Address hygiene:
#  * full tiles are exact, in-bounds, stride-1 tiles with no mask at all;
#  * where a partial tile is unavoidable the address is clamped with a
#    *compile-time* bound (`tl.minimum(cols, N - 1)`, N is a `tl.constexpr`)
#    and the out-of-range lanes are removed afterwards by an explicit
#    `tl.where` lane predicate, never by `other=`;
#  * a clamped (per-lane gather) load faults above 64 KiB on this backend, so
#    the fused single-tile path is capped at NMDIM_TILE_BYTES.
#
# Tile widths are budgeted by the *key* width (4 bytes, 8 for int64), not by the
# input element width, so the widest live vector in any of these kernels stays
# at NMDIM_TILE_BYTES.  A 2-byte input with 32768 lanes keeps its values inside
# 64 KiB but blows the key vector up to 128 KiB, and int16 x 32768 in that shape
# was measured to wedge the card (`reason[29] task timeout` + a persistent NOC
# fault that survives the process and needs a soft reset), while the same shape
# with a 16384-lane tile is fine.
# ---------------------------------------------------------------------------
NMDIM_TILE_BYTES = 65536
NMDIM_MAX_LANES = 32768
NMDIM_IDX_SENTINEL = tl.constexpr(2147483647)


def _nmdim_tile_lanes(key_bits):
    key_bytes = 8 if key_bits == 64 else 4
    return min(NMDIM_MAX_LANES, NMDIM_TILE_BYTES // key_bytes)


@triton.jit
def _nmdim_gated_keys(vals, inb, KEY64: tl.constexpr, KEY_BITS: tl.constexpr):
    keys = convert_to_uint_preverse_order(vals, False)
    if KEY64:
        keys = keys.to(tl.uint64)
        valid = inb
    else:
        keys = keys.to(tl.uint32)
        if vals.dtype.is_floating():
            # NaN must outrank every real value; the raw float preorder maps
            # *negative* NaN to the bottom of the domain instead.  The top of
            # the key domain is never produced by a non-NaN float (it is itself
            # a NaN bit pattern), so it is a collision-free sentinel.
            nan = _median_is_nan(vals)
            keys = tl.where(nan, (1 << KEY_BITS) - 1, keys)
            valid = inb & (~nan)
        else:
            valid = inb
    return keys, valid


@triton.jit
def _nmdim_key_limits(KEY64: tl.constexpr):
    if KEY64:
        top = tl.full((), 0xFFFFFFFFFFFFFFFF, dtype=tl.uint64)
        bottom = tl.full((), 0, dtype=tl.uint64)
    else:
        top = tl.full((), 0xFFFFFFFF, dtype=tl.uint32)
        bottom = tl.full((), 0, dtype=tl.uint32)
    return top, bottom


@triton.jit
def _nmdim_key_vectors(KEY64: tl.constexpr, CHUNK: tl.constexpr):
    if KEY64:
        vtop = tl.full((CHUNK,), 0xFFFFFFFFFFFFFFFF, dtype=tl.uint64)
        vbottom = tl.full((CHUNK,), 0, dtype=tl.uint64)
    else:
        vtop = tl.full((CHUNK,), 0xFFFFFFFF, dtype=tl.uint32)
        vbottom = tl.full((CHUNK,), 0, dtype=tl.uint32)
    return vtop, vbottom


@libentry()
@triton.jit
def nmdim_single_kernel(
    inp,
    out_values,
    out_indices,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KEY_BITS: tl.constexpr,
    KEY64: tl.constexpr,
):
    # grid: (M,); one reduction row per program, entirely in registers.  The
    # row is read exactly once, then the whole binary search runs on the
    # register-resident key tile: no extra launch, no host synchronization.
    pid = ext.program_id(0)
    base = pid.to(tl.int64) * N
    cols = tl.arange(0, BLOCK_N)
    inb = cols < N
    vals = tl.load(inp + base + tl.minimum(cols, N - 1))
    keys, valid = _nmdim_gated_keys(vals, inb, KEY64, KEY_BITS)
    top, bottom = _nmdim_key_limits(KEY64)
    nvalid = tl.sum(valid.to(tl.int32), axis=0)
    target = tl.maximum((nvalid - 1) // 2, 0)
    all_nan = nvalid <= 0
    lo = tl.where(all_nan, top, tl.min(tl.where(valid, keys, top), axis=0))
    hi = tl.where(all_nan, top, tl.max(tl.where(valid, keys, bottom), axis=0))
    for _step in tl.range(0, KEY_BITS):
        mid = lo + ((hi - lo) >> 1)
        le = tl.sum((valid & (keys <= mid)).to(tl.int32), axis=0)
        go_left = le > target
        active = lo < hi
        hi = tl.where(go_left & active, mid, hi)
        lo = tl.where((~go_left) & active, mid + 1, lo)
    hit = inb & (keys == lo)
    ridx = tl.minimum(tl.min(tl.where(hit, cols, BLOCK_N), axis=0), N - 1)
    tl.store(out_values + pid, tl.load(inp + base + ridx))
    tl.store(out_indices + pid, ridx.to(tl.int64))


# The chunked path is a partial + combine pipeline of loop-free kernels rather
# than one kernel with a tile loop.  Three independent lowering failures were
# measured for the loop form once a row spans more than one tile (NTILES >= 2;
# NTILES == 1 compiles and is correct):
#   * `tl.reduce` inside the tile loop -> "failed to legalize operation
#     'tt.reduce'";
#   * a scalar `tt.addptr` (row base, or the final scalar value fetch) next to
#     CHUNK-wide tiles -> "'tt.addptr' op all non-scalar operands/results must
#     have the same shape and base type";
#   * after folding the base into the pointer outside the loops, the nested
#     `tl.range` bisection still died in `arith.addi`, and full `tl.static_range`
#     unrolling of the inner loop died in `ConvertTritonXPUToLLVM`.
# Every kernel below therefore handles exactly one tile (or one row of tile
# partials) with no loop at all, which is the same shape as the proven
# `nmflat_*` flat pipeline.  The bisection state lives in device memory, so the
# whole reduction still needs zero host synchronizations.


@libentry()
@triton.jit
def nmdim_tile_stats_kernel(
    inp,
    pmin,
    pmax,
    pcnt,
    N: tl.constexpr,
    ROW_PITCH: tl.constexpr,
    NT: tl.constexpr,
    PSTRIDE: tl.constexpr,
    CHUNK: tl.constexpr,
    KEY_BITS: tl.constexpr,
    KEY64: tl.constexpr,
    EXACT: tl.constexpr,
):
    # grid: (M * NT,); per-tile key span and valid count.
    pid = ext.program_id(0)
    row = pid // NT
    tile = pid % NT
    cols = tl.arange(0, CHUNK)
    off = tile * CHUNK
    vals = tl.load(inp + row.to(tl.int64) * ROW_PITCH + off + cols)
    if EXACT:
        inb = cols >= 0
    else:
        inb = off + cols < N
    keys, valid = _nmdim_gated_keys(vals, inb, KEY64, KEY_BITS)
    top, bottom = _nmdim_key_limits(KEY64)
    slot = row * PSTRIDE + tile
    tl.store(pmin + slot, tl.min(tl.where(valid, keys, top), axis=0))
    tl.store(pmax + slot, tl.max(tl.where(valid, keys, bottom), axis=0))
    tl.store(pcnt + slot, tl.sum(valid.to(tl.int32), axis=0))


@libentry()
@triton.jit
def nmdim_fold_kernel(
    pmin,
    pmax,
    pcnt,
    lo_buf,
    hi_buf,
    mid_buf,
    tgt_buf,
    NTP: tl.constexpr,
    PSTRIDE: tl.constexpr,
    KEY64: tl.constexpr,
):
    # grid: (M,); fold the per-tile partials of one row and seed the search.
    row = ext.program_id(0)
    slots = row * PSTRIDE + tl.arange(0, NTP)
    lo = tl.min(tl.load(pmin + slots), axis=0)
    hi = tl.max(tl.load(pmax + slots), axis=0)
    nvalid = tl.sum(tl.load(pcnt + slots), axis=0)
    top, _bottom = _nmdim_key_limits(KEY64)
    all_nan = nvalid <= 0
    lo = tl.where(all_nan, top, lo)
    hi = tl.where(all_nan, top, hi)
    tl.store(lo_buf + row, lo)
    tl.store(hi_buf + row, hi)
    tl.store(mid_buf + row, lo + ((hi - lo) >> 1))
    tl.store(tgt_buf + row, tl.maximum((nvalid - 1) // 2, 0))


@libentry()
@triton.jit
def nmdim_tile_count_kernel(
    inp,
    mid_buf,
    pcnt,
    N: tl.constexpr,
    ROW_PITCH: tl.constexpr,
    NT: tl.constexpr,
    PSTRIDE: tl.constexpr,
    CHUNK: tl.constexpr,
    KEY_BITS: tl.constexpr,
    KEY64: tl.constexpr,
    EXACT: tl.constexpr,
):
    # grid: (M * NT,); count keys <= mid[row] inside one tile.
    pid = ext.program_id(0)
    row = pid // NT
    tile = pid % NT
    cols = tl.arange(0, CHUNK)
    off = tile * CHUNK
    vals = tl.load(inp + row.to(tl.int64) * ROW_PITCH + off + cols)
    if EXACT:
        inb = cols >= 0
    else:
        inb = off + cols < N
    keys, valid = _nmdim_gated_keys(vals, inb, KEY64, KEY_BITS)
    mid = tl.load(mid_buf + row)
    tl.store(
        pcnt + row * PSTRIDE + tile,
        tl.sum((valid & (keys <= mid)).to(tl.int32), axis=0),
    )


@libentry()
@triton.jit
def nmdim_step_kernel(
    pcnt,
    lo_buf,
    hi_buf,
    mid_buf,
    tgt_buf,
    NTP: tl.constexpr,
    PSTRIDE: tl.constexpr,
):
    # grid: (M,); one device-resident bisection step for one row.
    row = ext.program_id(0)
    slots = row * PSTRIDE + tl.arange(0, NTP)
    total = tl.sum(tl.load(pcnt + slots), axis=0)
    lo = tl.load(lo_buf + row)
    hi = tl.load(hi_buf + row)
    mid = tl.load(mid_buf + row)
    target = tl.load(tgt_buf + row)
    go_left = total > target
    active = lo < hi
    new_hi = tl.where(go_left & active, mid, hi)
    new_lo = tl.where((~go_left) & active, mid + 1, lo)
    tl.store(lo_buf + row, new_lo)
    tl.store(hi_buf + row, new_hi)
    tl.store(mid_buf + row, new_lo + ((new_hi - new_lo) >> 1))


@libentry()
@triton.jit
def nmdim_tile_select_kernel(
    inp,
    lo_buf,
    pfirst,
    N: tl.constexpr,
    ROW_PITCH: tl.constexpr,
    NT: tl.constexpr,
    PSTRIDE: tl.constexpr,
    CHUNK: tl.constexpr,
    KEY_BITS: tl.constexpr,
    KEY64: tl.constexpr,
    EXACT: tl.constexpr,
):
    # grid: (M * NT,); earliest in-tile column whose key equals the answer.
    pid = ext.program_id(0)
    row = pid // NT
    tile = pid % NT
    cols = tl.arange(0, CHUNK)
    off = tile * CHUNK
    vals = tl.load(inp + row.to(tl.int64) * ROW_PITCH + off + cols)
    if EXACT:
        inb = cols >= 0
    else:
        inb = off + cols < N
    keys, _valid = _nmdim_gated_keys(vals, inb, KEY64, KEY_BITS)
    sel = tl.load(lo_buf + row)
    hit = inb & (keys == sel)
    tl.store(
        pfirst + row * PSTRIDE + tile,
        tl.min(tl.where(hit, off + cols, NMDIM_IDX_SENTINEL), axis=0),
    )


@libentry()
@triton.jit
def nmdim_finish_kernel(
    inp,
    pfirst,
    out_values,
    out_indices,
    N: tl.constexpr,
    ROW_PITCH: tl.constexpr,
    NTP: tl.constexpr,
    PSTRIDE: tl.constexpr,
):
    # grid: (M,); fold the per-tile winners and fetch the selected value.
    row = ext.program_id(0)
    slots = row * PSTRIDE + tl.arange(0, NTP)
    best = tl.min(tl.load(pfirst + slots), axis=0)
    ridx = tl.minimum(best, N - 1)
    tl.store(out_values + row, tl.load(inp + row.to(tl.int64) * ROW_PITCH + ridx))
    tl.store(out_indices + row, ridx.to(tl.int64))



def _nmdim_key_bits(dtype):
    """Width of the key domain produced by `_nmdim_gated_keys`."""
    if dtype is torch.bfloat16:
        # `convert_to_uint_preverse_order` upcasts bfloat16 to float32; a 16-bit
        # bf16 key needs a bitcast, which aborts lowering on this backend.
        return 32
    if dtype is torch.int64:
        return 64
    return 8 * torch._utils._element_size(dtype)


def _nmdim_rows(inp, dim, M, N):
    """A contiguous (M, N) view of the reduction rows.

    The fast case (last dim of a contiguous tensor) is a pure view.  Anything
    else is materialized once with the native strided-copy primitive
    `aten::_copy_from`; `Tensor.contiguous()` is deliberately avoided because
    inside `use_gems()` it becomes a gems strided pointwise copy (measured
    hundreds of times slower, and a strided *source* read has been observed to
    wedge this device).
    """
    if dim == inp.ndim - 1 and inp.is_contiguous():
        return inp.reshape(M, N)
    src = inp if dim == inp.ndim - 1 else torch.movedim(inp, dim, -1)
    rows = torch.empty((M, N), dtype=inp.dtype, device=inp.device)
    torch.ops.aten._copy_from(src, rows.view(src.shape), False)
    return rows


def _nmdim_launch(rows, M, N, out_values, out_indices):
    key_bits = _nmdim_key_bits(rows.dtype)
    key64 = key_bits == 64
    kw = dict(num_warps=4, num_stages=1, buffer_size_limit=2048)
    lanes = _nmdim_tile_lanes(key_bits)
    block_n = max(64, triton.next_power_of_2(N))
    dev = rows.device
    if block_n <= lanes:
        with torch_device_fn.device(dev):
            nmdim_single_kernel[(M,)](
                rows, out_values, out_indices, N, block_n, key_bits, key64, **kw
            )
        return

    # Wide rows: partial + combine.  Rows whose width is not a multiple of the
    # tile are padded once so that every tile load is a plain, in-bounds,
    # unmasked stride-1 tile (no `other=`, no clamped gather).  None of the
    # accuracy or benchmark shapes needs that copy: their N is either small
    # enough for the fused kernel above or an exact multiple of CHUNK.
    chunk = lanes
    ntiles = (N + chunk - 1) // chunk
    exact = N % chunk == 0
    row_pitch = N if exact else ntiles * chunk
    if not exact:
        work = torch.empty((M, row_pitch), dtype=rows.dtype, device=dev)
        torch.ops.aten._copy_from(rows, work[:, :N], False)
        rows = work

    ntp = max(64, triton.next_power_of_2(ntiles))
    # A scalar store can commit a whole 64-element vector on this backend, so
    # every row of partials gets 64 slots of head room past the NTP slots the
    # folds read back.
    pstride = ntp + 64
    ukey = torch.uint64 if key64 else torch.uint32
    skey = torch.int64 if key64 else torch.int32
    pmin = torch.full((M * pstride,), -1, dtype=skey, device=dev).view(ukey)
    pmax = torch.zeros((M * pstride,), dtype=skey, device=dev).view(ukey)
    pcnt = torch.zeros((M * pstride,), dtype=torch.int32, device=dev)
    pfirst = torch.full(
        (M * pstride,), NMDIM_IDX_SENTINEL.value, dtype=torch.int32, device=dev
    )
    state = torch.zeros((3, M * pstride), dtype=skey, device=dev).view(ukey)
    lo_buf, hi_buf, mid_buf = state[0], state[1], state[2]
    tgt_buf = torch.zeros((M * pstride,), dtype=torch.int32, device=dev)

    tile_grid = (M * ntiles,)
    with torch_device_fn.device(dev):
        nmdim_tile_stats_kernel[tile_grid](
            rows,
            pmin,
            pmax,
            pcnt,
            N,
            row_pitch,
            ntiles,
            pstride,
            chunk,
            key_bits,
            key64,
            exact,
            **kw,
        )
        nmdim_fold_kernel[(M,)](
            pmin,
            pmax,
            pcnt,
            lo_buf,
            hi_buf,
            mid_buf,
            tgt_buf,
            ntp,
            pstride,
            key64,
            **kw,
        )
        for _ in range(key_bits):
            nmdim_tile_count_kernel[tile_grid](
                rows,
                mid_buf,
                pcnt,
                N,
                row_pitch,
                ntiles,
                pstride,
                chunk,
                key_bits,
                key64,
                exact,
                **kw,
            )
            nmdim_step_kernel[(M,)](
                pcnt, lo_buf, hi_buf, mid_buf, tgt_buf, ntp, pstride, **kw
            )
        nmdim_tile_select_kernel[tile_grid](
            rows,
            lo_buf,
            pfirst,
            N,
            row_pitch,
            ntiles,
            pstride,
            chunk,
            key_bits,
            key64,
            exact,
            **kw,
        )
        nmdim_finish_kernel[(M,)](
            rows, pfirst, out_values, out_indices, N, row_pitch, ntp, pstride, **kw
        )
