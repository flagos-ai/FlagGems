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

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

NanMedian = namedtuple("nanmedian", ["values", "indices"])
MAX_BLOCK_N = 128
RADIX_BLOCK_N = 1024
RADIX_BITS = 2


@triton.jit
def _is_not_nan(vals):
    vals_fp32 = vals.to(tl.float32)
    return vals_fp32 == vals_fp32


@triton.jit
def _to_order_key(vals, valid):
    dtype = vals.dtype
    nbits: tl.constexpr = dtype.primitive_bitwidth
    utype = tl.dtype(f"uint{nbits}")
    top_mask: tl.constexpr = 1 << (nbits - 1)
    full_mask: tl.constexpr = (1 << nbits) - 1
    full = tl.full(vals.shape, full_mask, dtype=utype)

    if dtype.is_floating():
        bits = vals.to(utype, bitcast=True)
        sign_mask = tl.where((bits & top_mask) != 0, full_mask, top_mask)
        key = bits ^ sign_mask
    elif dtype.is_int_signed():
        bits = vals.to(utype, bitcast=True)
        key = bits ^ top_mask
    else:
        key = vals.to(utype)
    return tl.where(valid, key, full)


@triton.jit
def _base_offset(
    out_id,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    DIM: tl.constexpr,
    NDIM: tl.constexpr,
):
    idx = out_id
    base = tl.full((), 0, dtype=tl.int64)
    for dim in tl.static_range(NDIM - 1, -1, -1):
        if dim != DIM:
            coord = idx % SHAPE[dim]
            idx = idx // SHAPE[dim]
            base += coord * STRIDES[dim]
    return base


@libentry()
@triton.jit
def nanmedian_direct_select_kernel(
    inp,
    out_values,
    out_indices,
    N: tl.constexpr,
    STRIDE_DIM: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
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

    base = _base_offset(pid, SHAPE, STRIDES, DIM, NDIM)
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
def nanmedian_direct_radix_kernel(
    inp,
    out_values,
    out_indices,
    N: tl.constexpr,
    STRIDE_DIM: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    DIM: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RADIX_BITS_: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    dtype = inp.dtype.element_ty
    nbits: tl.constexpr = dtype.primitive_bitwidth
    utype = tl.dtype(f"uint{nbits}")
    radix_size: tl.constexpr = 1 << RADIX_BITS_
    radix_mask: tl.constexpr = radix_size - 1
    radix_bins = tl.arange(0, radix_size)
    radix_mask_val = tl.full((), radix_mask, dtype=utype)
    base = _base_offset(pid, SHAPE, STRIDES, DIM, NDIM)

    valid_count = tl.full((), 0, dtype=tl.int32)
    for start in tl.range(0, N, BLOCK_N):
        cols = start + offsets
        mask = cols < N
        vals = tl.load(inp + base + cols * STRIDE_DIM, mask=mask, other=0.0)
        if dtype.is_floating():
            valid = mask & _is_not_nan(vals)
        else:
            valid = mask
        valid_count += tl.sum(valid.to(tl.int32), axis=0)

    k_to_find = (valid_count + 1) // 2
    desired = tl.full((), 0, dtype=utype)
    desired_mask = tl.full((), 0, dtype=utype)

    for digit_pos in tl.static_range(nbits - RADIX_BITS_, -1, -RADIX_BITS_):
        counts = tl.zeros((radix_size,), dtype=tl.int32)
        for start in tl.range(0, N, BLOCK_N):
            cols = start + offsets
            mask = cols < N
            vals = tl.load(inp + base + cols * STRIDE_DIM, mask=mask, other=0.0)
            if dtype.is_floating():
                valid = mask & _is_not_nan(vals)
            else:
                valid = mask
            keys = _to_order_key(vals, valid)
            matches = (keys & desired_mask) == desired
            digit = ((keys >> digit_pos) & radix_mask_val).to(tl.int32)
            active = valid & matches
            for radix_bin in tl.static_range(0, radix_size):
                bin_count = tl.sum((active & (digit == radix_bin)).to(tl.int32), axis=0)
                counts += tl.where(radix_bins == radix_bin, bin_count, 0)

        cumsum = tl.cumsum(counts, axis=0)
        prev = cumsum - counts
        take = (cumsum >= k_to_find) & (prev < k_to_find)
        selected_bin = tl.min(tl.where(take, radix_bins, radix_size - 1), axis=0)
        counts_before = tl.max(tl.where(take, prev, 0), axis=0)

        selected_bin = selected_bin.to(utype)
        desired = desired | (selected_bin << digit_pos)
        desired_mask = desired_mask | (radix_mask_val << digit_pos)
        k_to_find = k_to_find - counts_before

    result_idx = tl.full((), N, dtype=tl.int32)
    for start in tl.range(0, N, BLOCK_N):
        cols = start + offsets
        mask = cols < N
        vals = tl.load(inp + base + cols * STRIDE_DIM, mask=mask, other=0.0)
        if dtype.is_floating():
            valid = mask & _is_not_nan(vals)
        else:
            valid = mask
        keys = _to_order_key(vals, valid)
        local_idx = tl.min(tl.where(valid & (keys == desired), cols, N), axis=0)
        result_idx = tl.where(local_idx < result_idx, local_idx, result_idx)

    fallback_value = get_dtype_min(dtype)
    result_val = tl.load(
        inp + base + result_idx * STRIDE_DIM,
        mask=valid_count > 0,
        other=fallback_value,
    )

    if dtype.is_floating():
        all_nan = valid_count == 0
        result_val = tl.where(all_nan, float("nan"), result_val)
        result_idx = tl.where(all_nan, 0, result_idx)

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
    shape_tuple = tuple(shape)
    stride_tuple = tuple(inp.stride())
    stride_dim = stride_tuple[dim]

    if N <= MAX_BLOCK_N:
        block_n = triton.next_power_of_2(N)
        num_warps = 4
        with torch_device_fn.device(inp.device):
            nanmedian_direct_select_kernel[(M,)](
                inp,
                flat_values,
                flat_indices,
                N,
                stride_dim,
                shape_tuple,
                stride_tuple,
                dim,
                inp.ndim,
                block_n,
                num_warps=num_warps,
                num_stages=1,
                buffer_size_limit=2048,
            )
    else:
        block_n = min(triton.next_power_of_2(N), RADIX_BLOCK_N)
        num_warps = 4 if block_n <= 512 else 8
        with torch_device_fn.device(inp.device):
            nanmedian_direct_radix_kernel[(M,)](
                inp,
                flat_values,
                flat_indices,
                N,
                stride_dim,
                shape_tuple,
                stride_tuple,
                dim,
                inp.ndim,
                block_n,
                RADIX_BITS,
                num_warps=num_warps,
                num_stages=1,
                buffer_size_limit=2048,
            )

    if out is None and not keepdim:
        values = torch.squeeze(values, dim)
        indices = torch.squeeze(indices, dim)

    return NanMedian(values=values, indices=indices)


def nanmedian_dim(inp, dim=-1, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN NANMEDIAN DIM")
    _check_supported_dtype(inp)
    return _nanmedian_dim_impl(inp, dim, keepdim)


def nanmedian_dim_values(inp, dim=-1, keepdim=False, *, values, indices):
    logger.debug("GEMS_KUNLUNXIN NANMEDIAN DIM VALUES")
    _check_supported_dtype(inp)
    return _nanmedian_dim_impl(inp, dim, keepdim, out=(values, indices))
