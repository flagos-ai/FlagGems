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
MAX_NDIM = 8


def _unwrap_if_constexpr(value):
    return value.value if isinstance(value, tl.constexpr) else value


@tl.constexpr
def _uint_dtype(num_bits: tl.constexpr) -> tl.dtype:
    return tl.core.get_int_dtype(_unwrap_if_constexpr(num_bits), False)


@tl.constexpr
def _sign_bit_mask(num_bits: tl.constexpr) -> int:
    return 1 << (_unwrap_if_constexpr(num_bits) - 1)


@triton.jit
def _is_not_nan(vals):
    vals_fp32 = vals.to(tl.float32)
    return vals_fp32 == vals_fp32


@triton.jit
def _to_order_key(vals, valid):
    dtype = vals.dtype
    nbits: tl.constexpr = dtype.primitive_bitwidth
    utype = _uint_dtype(nbits)
    top = tl.full(vals.shape, _sign_bit_mask(nbits), dtype=utype)
    full = ~tl.full(vals.shape, 0, dtype=utype)

    if dtype.is_floating():
        bits = vals.to(utype, bitcast=True)
        sign_mask = tl.where((bits & top) != 0, full, top)
        key = bits ^ sign_mask
    elif dtype.is_int_signed():
        bits = vals.to(utype, bitcast=True)
        key = bits ^ top
    else:
        key = vals.to(utype)
    return tl.where(valid, key, full)


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
def nanmedian_direct_radix_kernel(
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
    RADIX_BITS_: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    dtype = inp.dtype.element_ty
    nbits: tl.constexpr = dtype.primitive_bitwidth
    utype = _uint_dtype(nbits)
    radix_size: tl.constexpr = 1 << RADIX_BITS_
    radix_mask: tl.constexpr = radix_size - 1
    radix_bins = tl.arange(0, radix_size)
    radix_mask_val = tl.full((), radix_mask, dtype=utype)
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
    stride_tuple = tuple(inp.stride())
    stride_dim = stride_tuple[dim]
    shape_meta = _pad_meta(shape, 1)
    stride_meta = _pad_meta(stride_tuple, 0)

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
        block_n = min(triton.next_power_of_2(N), RADIX_BLOCK_N)
        num_warps = 4 if block_n <= 512 else 8
        with torch_device_fn.device(inp.device):
            nanmedian_direct_radix_kernel[(M,)](
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
                RADIX_BITS,
                num_warps=num_warps,
                num_stages=1,
                buffer_size_limit=2048,
            )

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
    logger.debug("GEMS_KUNLUNXIN NANMEDIAN OUT")
    _check_supported_dtype(inp)
    return _nanmedian_flat_impl(inp, out=out)


def nanmedian_dim(inp, dim=-1, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN NANMEDIAN DIM")
    _check_supported_dtype(inp)
    return _nanmedian_dim_impl(inp, dim, keepdim)


def nanmedian_dim_values(inp, dim=-1, keepdim=False, *, values, indices):
    logger.debug("GEMS_KUNLUNXIN NANMEDIAN DIM VALUES")
    _check_supported_dtype(inp)
    return _nanmedian_dim_impl(inp, dim, keepdim, out=(values, indices))
