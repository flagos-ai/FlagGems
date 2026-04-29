import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.topk import _get_finfo_val, _get_iinfo_val
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

MedianResult = namedtuple("median", ["values", "indices"])

_DIRECT_REDUCTION_LIMIT = 256
_DIRECT_FLAT_LIMIT = 256
_DIRECT_REDUCTION_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_FLAT_SORT_LIMIT = 1024
_LASTDIM_SORT_LIMIT = 1024
_BF16_LASTDIM_SORT_LIMIT = 2048
_LASTDIM_SORT_DTYPES = {torch.float16, torch.bfloat16}
_FLAT_SORT_DTYPES = _LASTDIM_SORT_DTYPES | {torch.float32}
_F16_KEY_SELECT_MIN = 513
_F16_KEY_SELECT_LIMIT = 2048
_F16_KEY_SELECT_DTYPES = {torch.float16, torch.bfloat16}
_FP32_KEY_SELECT_MIN = 257
_FP32_KEY_SELECT_LIMIT = 2048
_INT_LASTDIM_SELECT_LIMIT = 2048
_INT_LASTDIM_SELECT_DTYPES = {torch.int16, torch.int32}


@libentry()
@triton.jit
def median_small_dim_kernel(
    inp,
    values,
    indices,
    total_outputs,
    reduction_size,
    inner_size,
    BLOCK_N: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    out_offsets = tl.program_id(0) * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    out_mask = out_offsets < total_outputs
    inner_offsets = out_offsets % inner_size
    outer_offsets = out_offsets // inner_size

    reduction_offsets = tl.arange(0, BLOCK_N)
    sample_mask = (reduction_offsets[None, :] < reduction_size) & out_mask[:, None]
    sample_ptrs = (
        inp
        + outer_offsets[:, None] * reduction_size * inner_size
        + reduction_offsets[None, :] * inner_size
        + inner_offsets[:, None]
    )

    if inp.dtype.element_ty.is_floating():
        high = _get_finfo_val(inp.dtype.element_ty, return_max=True)
    else:
        high = _get_iinfo_val(inp.dtype.element_ty, return_max=True)

    samples = tl.load(sample_ptrs, mask=sample_mask, other=high)
    sortable = samples

    if inp.dtype.element_ty.is_floating():
        nan_mask = sample_mask & (samples != samples)
        sortable = tl.where(nan_mask, high, samples)

    ordered = tl.sort(sortable, dim=1)
    rank = (reduction_size - 1) // 2
    rank_mask = reduction_offsets[None, :] == rank
    median_values = tl.sum(tl.where(rank_mask, ordered, tl.zeros_like(ordered)), axis=1)

    first_match = tl.argmax(
        (sample_mask & (samples == median_values[:, None])).to(tl.int32), axis=1
    )

    if inp.dtype.element_ty.is_floating():
        nan_i32 = nan_mask.to(tl.int32)
        has_nan = tl.max(nan_i32, axis=1) != 0
        first_nan = tl.argmax(nan_i32, axis=1)
        nan_values = tl.load(
            inp
            + outer_offsets * reduction_size * inner_size
            + first_nan * inner_size
            + inner_offsets,
            mask=out_mask,
            other=0.0,
        )
        median_values = tl.where(has_nan, nan_values, median_values)
        first_match = tl.where(has_nan, first_nan, first_match)

    tl.store(values + out_offsets, median_values, mask=out_mask)
    tl.store(indices + out_offsets, first_match.to(tl.int64), mask=out_mask)


@libentry()
@triton.jit
def median_small_flat_kernel(
    inp,
    value,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    valid = offsets < WIDTH

    if inp.dtype.element_ty.is_floating():
        high = _get_finfo_val(inp.dtype.element_ty, return_max=True)
    else:
        high = _get_iinfo_val(inp.dtype.element_ty, return_max=True)

    data = tl.load(inp + offsets, mask=valid, other=high)
    sortable = data

    if inp.dtype.element_ty.is_floating():
        nan_mask = valid & (data != data)
        sortable = tl.where(nan_mask, high, data)

    ordered = tl.sort(sortable)
    rank = (WIDTH - 1) // 2
    median_value = tl.sum(
        tl.where(offsets == rank, ordered, tl.zeros_like(ordered)), axis=0
    )

    if inp.dtype.element_ty.is_floating():
        nan_i32 = nan_mask.to(tl.int32)
        has_nan = tl.max(nan_i32, axis=0) != 0
        first_nan = tl.argmax(nan_i32, axis=0)
        nan_value = tl.load(inp + first_nan, mask=has_nan, other=0.0)
        median_value = tl.where(has_nan, nan_value, median_value)

    tl.store(value, median_value)


@libentry()
@triton.jit
def median_nan_fix_kernel(
    row_data,
    values,
    indices,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    valid = lane < WIDTH
    base = row_data + row * WIDTH
    data = tl.load(base + lane, mask=valid, other=0.0)
    is_nan = valid & (data != data)
    is_nan_i32 = is_nan.to(tl.int32)
    row_has_nan = tl.max(is_nan_i32, axis=0) != 0
    first_nan = tl.argmax(is_nan_i32, axis=0)
    tl.store(values + row, tl.load(base + first_nan), mask=row_has_nan)
    tl.store(indices + row, first_nan.to(tl.int64), mask=row_has_nan)


@libentry()
@triton.jit
def median_lastdim_sort_kernel(
    row_data,
    values,
    indices,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < WIDTH
    base = row_data + row * WIDTH
    data = tl.load(base + cols, mask=valid, other=float("inf"))

    nan_mask = valid & (data != data)
    sortable = tl.where(nan_mask, float("inf"), data)
    ordered = tl.sort(sortable)
    rank = (WIDTH - 1) // 2
    median_value = tl.sum(
        tl.where(cols == rank, ordered, tl.zeros_like(ordered)), axis=0
    )

    first_match = tl.argmax((valid & (data == median_value)).to(tl.int32), axis=0)
    nan_i32 = nan_mask.to(tl.int32)
    has_nan = tl.max(nan_i32, axis=0) != 0
    first_nan = tl.argmax(nan_i32, axis=0)
    nan_value = tl.load(base + first_nan, mask=has_nan, other=0.0)
    median_value = tl.where(has_nan, nan_value, median_value)
    first_match = tl.where(has_nan, first_nan, first_match)

    tl.store(values + row, median_value)
    tl.store(indices + row, first_match.to(tl.int64))


@libentry()
@triton.jit
def median_int_lastdim_select_kernel(
    row_data,
    values,
    indices,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < WIDTH
    base = row_data + row * WIDTH
    data = tl.load(base + cols, mask=valid, other=0)

    dtype = row_data.dtype.element_ty
    high = _get_iinfo_val(dtype, return_max=True)
    low = _get_iinfo_val(dtype, return_max=False)
    row_min = tl.min(tl.where(valid, data, high), axis=0).to(tl.int64)
    row_max = tl.max(tl.where(valid, data, low), axis=0).to(tl.int64)

    lo = row_min
    hi = row_max
    rank = (WIDTH - 1) // 2
    for _ in tl.static_range(0, SEARCH_STEPS):
        mid = lo + ((hi - lo) // 2)
        le_count = tl.sum((valid & (data <= mid.to(dtype))).to(tl.int32), axis=0)
        take_left = le_count > rank
        hi = tl.where(take_left, mid, hi)
        lo = tl.where(take_left, lo, mid + 1)

    median_value = lo.to(dtype)
    first_match = tl.argmax((valid & (data == median_value)).to(tl.int32), axis=0)
    tl.store(values + row, median_value)
    tl.store(indices + row, first_match.to(tl.int64))


@triton.jit
def _fp32_order_key(x):
    bits = x.to(tl.uint32, bitcast=True)
    signed = x.to(tl.int32, bitcast=True)
    sign = signed >> 31
    sign_mask = tl.full((), 0x80000000, dtype=tl.uint32)
    mask = sign_mask | sign.to(tl.uint32, bitcast=True)
    return bits ^ mask


@triton.jit
def _f16_order_key(x):
    bits = x.to(tl.uint16, bitcast=True)
    signed = x.to(tl.int16, bitcast=True)
    sign = signed >> 15
    sign_mask = tl.full((), 0x8000, dtype=tl.uint16)
    mask = sign_mask | sign.to(tl.uint16, bitcast=True)
    return bits ^ mask


@libentry()
@triton.jit
def median_f16_key_select_kernel(
    row_data,
    values,
    indices,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < WIDTH
    base = row_data + row * WIDTH
    data = tl.load(base + cols, mask=valid, other=0.0)

    nan_mask = valid & (data != data)
    nan_i32 = nan_mask.to(tl.int32)
    has_nan = tl.max(nan_i32, axis=0) != 0
    first_nan = tl.argmax(nan_i32, axis=0)
    nan_value = tl.load(base + first_nan, mask=has_nan, other=0.0)

    keys = _f16_order_key(data)
    finite = valid & ~nan_mask
    key_min_fill = tl.full((), 0xFFFF, dtype=tl.uint16)
    key_max_fill = tl.full((), 0, dtype=tl.uint16)
    row_min = tl.min(tl.where(finite, keys, key_min_fill), axis=0)
    row_max = tl.max(tl.where(finite, keys, key_max_fill), axis=0)

    lo = row_min
    hi = row_max
    rank = (WIDTH - 1) // 2
    for _ in tl.static_range(0, 16):
        mid = lo + ((hi - lo) >> 1)
        le_count = tl.sum((finite & (keys <= mid)).to(tl.int32), axis=0)
        take_left = le_count > rank
        hi = tl.where(take_left, mid, hi)
        lo = tl.where(take_left, lo, mid + 1)

    selected_key = lo
    key_match = finite & (keys == selected_key)
    selected_key_first = tl.argmax(key_match.to(tl.int32), axis=0)
    selected_value = tl.load(base + selected_key_first)

    selected_value = tl.where(has_nan, nan_value, selected_value)
    first_match = tl.where(has_nan, first_nan, selected_key_first)
    tl.store(values + row, selected_value)
    tl.store(indices + row, first_match.to(tl.int64))


@libentry()
@triton.jit
def median_fp32_key_select_kernel(
    row_data,
    values,
    indices,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < WIDTH
    base = row_data + row * WIDTH
    data = tl.load(base + cols, mask=valid, other=0.0)

    nan_mask = valid & (data != data)
    nan_i32 = nan_mask.to(tl.int32)
    has_nan = tl.max(nan_i32, axis=0) != 0
    first_nan = tl.argmax(nan_i32, axis=0)
    nan_value = tl.load(base + first_nan, mask=has_nan, other=0.0)

    keys = _fp32_order_key(data)
    finite = valid & ~nan_mask
    key_min_fill = tl.full((), 0xFFFFFFFF, dtype=tl.uint32)
    key_max_fill = tl.full((), 0, dtype=tl.uint32)
    row_min = tl.min(tl.where(finite, keys, key_min_fill), axis=0)
    row_max = tl.max(tl.where(finite, keys, key_max_fill), axis=0)

    lo = row_min
    hi = row_max
    rank = (WIDTH - 1) // 2
    for _ in tl.static_range(0, 32):
        mid = lo + ((hi - lo) >> 1)
        le_count = tl.sum((finite & (keys <= mid)).to(tl.int32), axis=0)
        take_left = le_count > rank
        hi = tl.where(take_left, mid, hi)
        lo = tl.where(take_left, lo, mid + 1)

    selected_key = lo
    key_match = finite & (keys == selected_key)
    selected_key_first = tl.argmax(key_match.to(tl.int32), axis=0)
    selected_value = tl.load(base + selected_key_first)

    selected_value = tl.where(has_nan, nan_value, selected_value)
    first_match = tl.where(has_nan, first_nan, selected_key_first)
    tl.store(values + row, selected_value)
    tl.store(indices + row, first_match.to(tl.int64))


def _has_names(inp):
    return any(name is not None for name in inp.names)


def _anonymous(inp):
    return inp.rename(None) if _has_names(inp) else inp


def _canonical_dim(ndim, dim):
    lower = -1 if ndim == 0 else -ndim
    upper = 0 if ndim == 0 else ndim - 1
    if dim < lower or dim > upper:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{lower}, {upper}], but got {dim})"
        )
    return 0 if ndim == 0 else dim % ndim


def _name_to_dim(inp, dim):
    if dim not in inp.names:
        raise RuntimeError(f"Name '{dim}' not found in Tensor{inp.names}.")
    return inp.names.index(dim)


def _kept_names(names, dim, keepdim):
    if names is None:
        return None
    if keepdim:
        return names
    return names[:dim] + names[dim + 1 :]


def _empty_result_value(inp):
    if inp.dtype.is_complex:
        out = torch.empty((), dtype=inp.dtype, device=inp.device)
        out.real.fill_(float("nan"))
        out.imag.zero_()
        return out
    if inp.dtype.is_floating_point:
        return torch.full((), float("nan"), dtype=inp.dtype, device=inp.device)
    if inp.dtype == torch.bool:
        return torch.ones((), dtype=inp.dtype, device=inp.device)
    if inp.dtype in (torch.int32, torch.int64):
        return torch.full(
            (), torch.iinfo(inp.dtype).min, dtype=inp.dtype, device=inp.device
        )
    return torch.zeros((), dtype=inp.dtype, device=inp.device)


def _raise_dim_dtype(dtype):
    dtype_names = {
        torch.bool: "Bool",
        torch.complex64: "ComplexFloat",
        torch.complex128: "ComplexDouble",
    }
    dtype_name = dtype_names.get(dtype, str(dtype).removeprefix("torch."))
    raise NotImplementedError(f'"median_out_impl" not implemented for {dtype_name!r}')


def _nan_override(row_data, row_values, row_indices):
    if not row_data.dtype.is_floating_point:
        return row_values, row_indices

    width = row_data.shape[-1]
    rows = row_data.numel() // width
    block = triton.next_power_of_2(width)
    num_warps = 1 if block <= 1024 else min(8, max(4, block // 512))
    with torch_device_fn.device(row_data.device):
        median_nan_fix_kernel[(rows,)](
            row_data.reshape(rows, width),
            row_values.reshape(rows),
            row_indices.reshape(rows),
            WIDTH=width,
            BLOCK=block,
            num_warps=num_warps,
        )
    return row_values, row_indices


def _median_from_rows(row_data):
    kth = (row_data.shape[-1] + 1) // 2
    result = torch.kthvalue(row_data, kth, dim=-1)
    return _nan_override(row_data, result.values, result.indices)


def _median_small_flat(inp):
    value = torch.empty((), dtype=inp.dtype, device=inp.device)
    block = triton.next_power_of_2(inp.numel())
    with torch_device_fn.device(inp.device):
        median_small_flat_kernel[(1,)](
            inp.reshape(-1),
            value,
            WIDTH=inp.numel(),
            BLOCK=block,
            num_warps=min(8, max(4, block // 32)),
        )
    return value


def _median_lastdim_sort(row_data, output_shape):
    width = row_data.shape[-1]
    rows = row_data.numel() // width
    values = torch.empty(output_shape, dtype=row_data.dtype, device=row_data.device)
    indices = torch.empty(output_shape, dtype=torch.int64, device=row_data.device)
    block = triton.next_power_of_2(width)
    num_warps = 8 if rows == 1 and block >= 1024 else min(8, max(4, block // 512))
    with torch_device_fn.device(row_data.device):
        median_lastdim_sort_kernel[(rows,)](
            row_data.reshape(rows, width),
            values.reshape(rows),
            indices.reshape(rows),
            WIDTH=width,
            BLOCK=block,
            num_warps=num_warps,
        )
    return values, indices


def _use_lastdim_sort(dtype, width):
    if dtype == torch.bfloat16:
        return width <= _BF16_LASTDIM_SORT_LIMIT
    if dtype == torch.float16:
        return width <= _LASTDIM_SORT_LIMIT
    return False


def _use_f16_key_select(dtype, width):
    return (
        dtype in _F16_KEY_SELECT_DTYPES
        and _F16_KEY_SELECT_MIN <= width <= _F16_KEY_SELECT_LIMIT
    )


def _use_fp32_key_select(dtype, width):
    return (
        dtype == torch.float32
        and _FP32_KEY_SELECT_MIN <= width <= _FP32_KEY_SELECT_LIMIT
    )


def _median_int_lastdim_select(row_data, output_shape):
    width = row_data.shape[-1]
    rows = row_data.numel() // width
    values = torch.empty(output_shape, dtype=row_data.dtype, device=row_data.device)
    indices = torch.empty(output_shape, dtype=torch.int64, device=row_data.device)
    block = triton.next_power_of_2(width)
    search_steps = 16 if row_data.dtype == torch.int16 else 32
    with torch_device_fn.device(row_data.device):
        median_int_lastdim_select_kernel[(rows,)](
            row_data.reshape(rows, width),
            values.reshape(rows),
            indices.reshape(rows),
            WIDTH=width,
            BLOCK=block,
            SEARCH_STEPS=search_steps,
            num_warps=min(8, max(4, block // 512)),
        )
    return values, indices


def _median_f16_key_select(row_data, output_shape):
    width = row_data.shape[-1]
    rows = row_data.numel() // width
    values = torch.empty(output_shape, dtype=row_data.dtype, device=row_data.device)
    indices = torch.empty(output_shape, dtype=torch.int64, device=row_data.device)
    block = triton.next_power_of_2(width)
    num_warps = 1 if block <= 1024 else 2 if block <= 2048 else 4
    with torch_device_fn.device(row_data.device):
        median_f16_key_select_kernel[(rows,)](
            row_data.reshape(rows, width),
            values.reshape(rows),
            indices.reshape(rows),
            WIDTH=width,
            BLOCK=block,
            num_warps=num_warps,
        )
    return values, indices


def _median_fp32_key_select(row_data, output_shape):
    width = row_data.shape[-1]
    rows = row_data.numel() // width
    values = torch.empty(output_shape, dtype=row_data.dtype, device=row_data.device)
    indices = torch.empty(output_shape, dtype=torch.int64, device=row_data.device)
    block = triton.next_power_of_2(width)
    num_warps = 2 if block <= 1024 else 8
    with torch_device_fn.device(row_data.device):
        median_fp32_key_select_kernel[(rows,)](
            row_data.reshape(rows, width),
            values.reshape(rows),
            indices.reshape(rows),
            WIDTH=width,
            BLOCK=block,
            num_warps=num_warps,
        )
    return values, indices


def _median_direct_dim(inp, dim, output_shape):
    reduction_size = inp.shape[dim]
    inner_size = math.prod(inp.shape[dim + 1 :])
    total_outputs = math.prod(output_shape)
    values = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)
    indices = torch.empty(output_shape, dtype=torch.int64, device=inp.device)
    block_n = triton.next_power_of_2(reduction_size)
    block_out = 2 if block_n >= 128 else 16
    if block_n >= 128:
        num_warps = 8 if inp.dtype in (torch.int32, torch.int64) else 4
    else:
        num_warps = 1
    with torch_device_fn.device(inp.device):
        median_small_dim_kernel[(triton.cdiv(total_outputs, block_out),)](
            inp,
            values.reshape(-1),
            indices.reshape(-1),
            total_outputs,
            reduction_size,
            inner_size,
            BLOCK_N=block_n,
            BLOCK_OUT=block_out,
            num_warps=num_warps,
        )
    return values, indices


def _copy_out(src, out, name):
    if out.device != src.device:
        raise RuntimeError(
            f"Expected {name} tensor to have device {src.device}, "
            f"but got {out.device} instead"
        )
    if out.dtype != src.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {src.dtype}, but got {out.dtype}"
        )
    out.resize_as_(src)
    out.copy_(src)
    return out


def median(inp):
    logger.debug("GEMS MEDIAN")

    inp = _anonymous(inp)
    if inp.numel() == 0:
        return _empty_result_value(inp)
    if inp.dtype == torch.bool:
        false_count = torch.count_nonzero(torch.logical_not(inp))
        return (false_count <= (inp.numel() - 1) // 2).to(torch.bool)
    if inp.dtype.is_complex:
        raise RuntimeError("Sort does not support complex dtypes on CPU")
    if inp.numel() == 1:
        return inp.reshape(()).clone()

    flat = inp.contiguous().reshape(-1)
    if inp.dtype in _DIRECT_REDUCTION_DTYPES and inp.numel() <= _DIRECT_FLAT_LIMIT:
        return _median_small_flat(flat)

    row_data = flat.reshape(1, inp.numel())
    if inp.dtype in _FLAT_SORT_DTYPES and inp.numel() <= _FLAT_SORT_LIMIT:
        values, _ = _median_lastdim_sort(row_data, ())
    else:
        values, _ = _median_from_rows(row_data)
    return values.reshape(())


def median_out(inp, *, out):
    logger.debug("GEMS MEDIAN.OUT")
    return _copy_out(median(inp), out, "out")


def median_dim(inp, dim=0, keepdim=False):
    logger.debug("GEMS MEDIAN.DIM")

    if isinstance(dim, str):
        dim = _name_to_dim(inp, dim)
    dim = _canonical_dim(inp.ndim, dim)
    names = inp.names if _has_names(inp) else None
    work = _anonymous(inp)

    if work.ndim == 0:
        if work.dtype == torch.bool or work.dtype.is_complex:
            _raise_dim_dtype(work.dtype)
        return MedianResult(
            values=work.clone(),
            indices=torch.zeros((), dtype=torch.int64, device=work.device),
        )

    if work.shape[dim] == 0:
        raise IndexError(
            f"median(): Expected reduction dim {dim} to have non-zero size."
        )

    output_shape = list(work.shape)
    if keepdim:
        output_shape[dim] = 1
    else:
        del output_shape[dim]
    output_names = _kept_names(names, dim, keepdim)

    if work.numel() == 0:
        values = torch.empty(output_shape, dtype=work.dtype, device=work.device)
        indices = torch.empty(output_shape, dtype=torch.int64, device=work.device)
    else:
        if work.dtype == torch.bool or work.dtype.is_complex:
            _raise_dim_dtype(work.dtype)
        if (
            work.shape[dim] <= _DIRECT_REDUCTION_LIMIT
            and work.dtype in _DIRECT_REDUCTION_DTYPES
        ):
            values, indices = _median_direct_dim(work.contiguous(), dim, output_shape)
        elif dim == work.ndim - 1 and _use_f16_key_select(work.dtype, work.shape[dim]):
            values, indices = _median_f16_key_select(work.contiguous(), output_shape)
        elif dim == work.ndim - 1 and _use_lastdim_sort(work.dtype, work.shape[dim]):
            values, indices = _median_lastdim_sort(work.contiguous(), output_shape)
        elif dim == work.ndim - 1 and _use_fp32_key_select(work.dtype, work.shape[dim]):
            values, indices = _median_fp32_key_select(work.contiguous(), output_shape)
        elif (
            dim == work.ndim - 1
            and work.shape[dim] <= _INT_LASTDIM_SELECT_LIMIT
            and work.dtype in _INT_LASTDIM_SELECT_DTYPES
        ):
            values, indices = _median_int_lastdim_select(
                work.contiguous(), output_shape
            )
        else:
            rows = torch.movedim(work, dim, -1).contiguous()
            values, indices = _median_from_rows(rows)
            if keepdim:
                values = torch.movedim(values.unsqueeze(-1), -1, dim)
                indices = torch.movedim(indices.unsqueeze(-1), -1, dim)

    if output_names is not None:
        values = values.refine_names(*output_names)
        indices = indices.refine_names(*output_names)

    return MedianResult(values=values, indices=indices)


def median_dim_values(inp, dim=0, keepdim=False, *, values, indices):
    logger.debug("GEMS MEDIAN.DIM_VALUES")
    result = median_dim(inp, dim=dim, keepdim=keepdim)
    _copy_out(result.values, values, "values")
    _copy_out(result.indices, indices, "indices")
    return MedianResult(values=values, indices=indices)
