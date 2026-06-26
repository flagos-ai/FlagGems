import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 1024
_NPU_BLOCK_SIZE = 256
_UNIFORM_FAST_PATH_MIN_NUMEL = 1 << 20
_UNIFORM_KERNEL_MAX_SEGMENT_LENGTH = 1024
_UNIFORM_LENGTHS_CACHE = {}
_SUPPORTED_REDUCES = ("sum", "mean", "max", "min", "prod")
_SUPPORTED_DATA_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)
_SUPPORTED_INDEX_DTYPES = (torch.int32, torch.int64)


def _prod(shape):
    return math.prod(shape) if shape else 1


def _get_block_size(device):
    return _NPU_BLOCK_SIZE if device.type == "npu" else _BLOCK_SIZE


def _get_uniform_kernel_config(device, inner_size):
    if device.type == "npu":
        return 4, 16 if inner_size > 1 else 1
    if inner_size == 1:
        return 16, 1
    return 4, 64


def _get_uniform_backward_tile_config(device, inner_size, reduce, dtype):
    if device.type == "npu":
        return 1, 16 if inner_size > 1 else 1
    if reduce == "prod" and inner_size > 1 and dtype in (torch.float16, torch.bfloat16):
        return 4, 256
    return 4, 64 if inner_size > 1 else 1


@triton.jit
def _mul_combine(a, b):
    return a * b


def _all_lengths_equal(lengths, value):
    cache_key = (
        lengths.device.type,
        lengths.data_ptr(),
        tuple(lengths.shape),
        getattr(lengths, "_version", None),
        value,
    )
    is_equal = _UNIFORM_LENGTHS_CACHE.get(cache_key)
    if is_equal is None:
        is_equal = torch.all(lengths.detach().cpu() == value).item()
        if len(_UNIFORM_LENGTHS_CACHE) > 128:
            _UNIFORM_LENGTHS_CACHE.clear()
        _UNIFORM_LENGTHS_CACHE[cache_key] = is_equal
    return is_equal


def _wrap_axis(axis, ndim):
    if ndim == 0:
        raise IndexError(
            "segment_reduce(): input tensor must have at least one dimension."
        )
    if axis < -ndim or axis >= ndim:
        raise IndexError(
            f"segment_reduce(): axis {axis} is out of bounds for tensor of dimension {ndim}."
        )
    return axis % ndim


def _check_reduce_and_dtype(data, reduce):
    if reduce not in _SUPPORTED_REDUCES:
        raise RuntimeError(
            "segment_reduce(): reduce must be one of 'sum', 'mean', 'max', 'min', or 'prod'."
        )
    if data.dtype not in _SUPPORTED_DATA_DTYPES:
        raise NotImplementedError(f'"segment_reduce" not implemented for {data.dtype}.')


def _check_index_tensor(data, index_tensor, name, axis):
    if index_tensor.dtype not in _SUPPORTED_INDEX_DTYPES:
        raise NotImplementedError(f"segment_reduce(): {name} must be int32 or int64.")
    if index_tensor.device != data.device:
        raise RuntimeError(
            f"segment_reduce(): Expected data and {name} on the same device."
        )
    if data.dim() < index_tensor.dim():
        raise RuntimeError(
            f"segment_reduce(): Expected data.dim() >= {name}.dim(), got "
            f"{data.dim()} and {index_tensor.dim()}."
        )
    if axis != index_tensor.dim() - 1:
        raise RuntimeError(
            f"segment_reduce(): Expected axis to be the last dimension of {name} "
            f"but got {axis}."
        )


def _validate_lengths(data, lengths, axis, unsafe):
    _check_index_tensor(data, lengths, "lengths", axis)
    if unsafe:
        return
    lengths_cpu = lengths.detach().cpu()
    if torch.any(lengths_cpu < 0).item():
        raise RuntimeError("lengths contains negative value!")
    valid_lengths = torch.all(lengths_cpu.sum(dim=-1) == data.size(axis)).item()
    if not valid_lengths:
        raise RuntimeError(
            "segment_reduce(): Expected all rows of lengths along axis to sum to "
            "data.size(lengths.dim()-1) when !unsafe."
        )


def _make_initial(reduce, initial):
    if initial is not None:
        return True, initial
    if reduce == "max":
        return False, float("-inf")
    if reduce == "min":
        return False, float("inf")
    if reduce == "prod":
        return False, 1.0
    return False, 0.0


def _get_uniform_segment_length(data, lengths, axis):
    if data.numel() < _UNIFORM_FAST_PATH_MIN_NUMEL:
        return None
    if tuple(lengths.shape[:-1]) != tuple(data.shape[:axis]):
        return None
    segment_count = lengths.shape[-1]
    if segment_count <= 0:
        return None
    data_size_axis = data.shape[axis]
    if data_size_axis % segment_count != 0:
        return None
    segment_length = data_size_axis // segment_count
    if segment_length <= 0:
        return None

    if _all_lengths_equal(lengths, segment_length):
        return segment_length
    return None


def _is_unit_lengths(data, lengths, axis):
    if tuple(lengths.shape[:-1]) != tuple(data.shape[:axis]):
        return False
    if lengths.shape[-1] != data.shape[axis]:
        return False
    return _all_lengths_equal(lengths, 1)


@libentry()
@triton.jit
def _segment_reduce_uniform_other_backward_kernel(
    grad,
    output,
    data,
    grad_input,
    total_rows,
    segment_count,
    segment_length,
    inner_size,
    data_size_axis,
    IS_MAX: tl.constexpr,
    IS_MIN: tl.constexpr,
    IS_PROD: tl.constexpr,
    INITIAL_PROD_VALUE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None, None]
    seg_offsets = tl.arange(0, BLOCK_N)[None, :, None]
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)[None, None, :]
    output_rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    output_k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
    row_mask = rows < total_rows
    seg_mask = seg_offsets < segment_length
    k_mask = k_offsets < inner_size
    mask = row_mask & seg_mask & k_mask

    outer_idx = rows // segment_count
    dim_idx = rows - outer_idx * segment_count
    data_offsets = (
        outer_idx * data_size_axis * inner_size
        + (dim_idx * segment_length + seg_offsets) * inner_size
        + k_offsets
    )
    output_offsets = output_rows * inner_size + output_k_offsets
    output_mask = (output_rows < total_rows) & (output_k_offsets < inner_size)

    values = tl.load(data + data_offsets, mask=mask, other=0.0).to(compute_dtype)
    grad_value = tl.load(grad + output_offsets, mask=output_mask, other=0.0).to(
        compute_dtype
    )
    output_value = tl.load(output + output_offsets, mask=output_mask, other=0.0).to(
        compute_dtype
    )

    if IS_MAX or IS_MIN:
        match = ((values != values) | (values == output_value[:, None, :])) & mask
        counter = tl.sum(match.to(tl.int64), axis=1)
        store_value = tl.where(
            (counter >= 2) & (grad_value > 0),
            grad_value / counter,
            grad_value,
        )
        tl.store(grad_input + data_offsets, store_value[:, None, :], mask=match)
    elif IS_PROD:
        nan_mask = (values != values) & mask
        zero_mask = (values == 0) & mask & ~nan_mask
        zero_count = tl.sum(zero_mask.to(tl.int64), axis=1)
        nan_count = tl.sum(nan_mask.to(tl.int64), axis=1)
        product_values = tl.where(nan_mask | zero_mask | ~mask, 1.0, values)
        product = tl.reduce(product_values, axis=1, combine_fn=_mul_combine)
        product *= INITIAL_PROD_VALUE

        zero_scalar = tl.full((BLOCK_M, BLOCK_K), 0.0, dtype=compute_dtype)
        nan_scalar = zero_scalar / zero_scalar
        normal_prefix = grad_value * output_value
        normal_grad = normal_prefix[:, None, :] / values
        zero_exclusive = tl.where(
            nan_count > 0,
            nan_scalar,
            tl.where(zero_count > 1, zero_scalar, product),
        )
        nan_exclusive = tl.where(
            nan_count > 1,
            nan_scalar,
            tl.where(zero_count > 0, zero_scalar, product),
        )
        exclusive = tl.where(
            nan_mask, nan_exclusive[:, None, :], zero_exclusive[:, None, :]
        )
        grad_result = tl.where(
            nan_mask | zero_mask,
            grad_value[:, None, :] * exclusive,
            normal_grad,
        )
        tl.store(grad_input + data_offsets, grad_result, mask=mask)


@libentry()
@triton.jit
def _segment_reduce_uniform_sum_mean_backward_kernel(
    grad,
    grad_input,
    total_numel,
    segment_count,
    segment_length,
    inner_size,
    data_size_axis,
    IS_MEAN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_numel

    inner_idx = offsets % inner_size
    axis_idx = (offsets // inner_size) % data_size_axis
    outer_idx = offsets // (data_size_axis * inner_size)
    segment_idx = axis_idx // segment_length
    grad_offsets = (outer_idx * segment_count + segment_idx) * inner_size + inner_idx

    grad_value = tl.load(grad + grad_offsets, mask=mask, other=0.0)
    if IS_MEAN:
        grad_value = grad_value / segment_length
    tl.store(grad_input + offsets, grad_value, mask=mask)


@libentry()
@triton.jit
def _segment_reduce_uniform_inner1_forward_kernel(
    data,
    output,
    total_rows,
    segment_count,
    segment_length,
    data_size_axis,
    IS_SUM: tl.constexpr,
    IS_MEAN: tl.constexpr,
    IS_MAX: tl.constexpr,
    IS_MIN: tl.constexpr,
    IS_PROD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32

    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    row_mask = rows < total_rows
    col_mask = cols < segment_length
    mask = row_mask & col_mask

    outer_idx = rows // segment_count
    dim_idx = rows - outer_idx * segment_count
    data_offsets = outer_idx * data_size_axis + dim_idx * segment_length + cols

    if IS_SUM or IS_MEAN:
        values = tl.load(data + data_offsets, mask=mask, other=0.0).to(compute_dtype)
        result = tl.sum(values, axis=1)
        if IS_MEAN:
            result = result / segment_length
    elif IS_PROD:
        values = tl.load(data + data_offsets, mask=mask, other=1.0).to(compute_dtype)
        result = tl.reduce(values, axis=1, combine_fn=_mul_combine)
    elif IS_MAX:
        values = tl.load(data + data_offsets, mask=mask, other=float("-inf")).to(
            compute_dtype
        )
        nan_mask = (values != values) & mask
        has_nan = tl.sum(nan_mask.to(tl.int32), axis=1) > 0
        nan_value = tl.sum(tl.where(nan_mask, values, 0.0), axis=1)
        result = tl.max(tl.where(mask & ~nan_mask, values, float("-inf")), axis=1)
        result = tl.where(has_nan, nan_value, result)
    elif IS_MIN:
        values = tl.load(data + data_offsets, mask=mask, other=float("inf")).to(
            compute_dtype
        )
        nan_mask = (values != values) & mask
        has_nan = tl.sum(nan_mask.to(tl.int32), axis=1) > 0
        nan_value = tl.sum(tl.where(nan_mask, values, 0.0), axis=1)
        result = tl.min(tl.where(mask & ~nan_mask, values, float("inf")), axis=1)
        result = tl.where(has_nan, nan_value, result)

    output_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(output + output_offsets, result, mask=output_offsets < total_rows)


@libentry()
@triton.jit
def _segment_reduce_uniform_forward_kernel(
    data,
    output,
    total_rows,
    segment_count,
    segment_length,
    inner_size,
    data_size_axis,
    IS_SUM: tl.constexpr,
    IS_MEAN: tl.constexpr,
    IS_MAX: tl.constexpr,
    IS_MIN: tl.constexpr,
    IS_PROD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
    row_mask = rows < total_rows
    k_mask = k_offsets < inner_size
    mask = row_mask & k_mask

    outer_idx = rows // segment_count
    dim_idx = rows - outer_idx * segment_count
    segment_start = dim_idx * segment_length
    base_offsets = (
        outer_idx * data_size_axis * inner_size + segment_start * inner_size + k_offsets
    )

    if IS_MAX:
        acc = tl.full((BLOCK_M, BLOCK_K), float("-inf"), dtype=compute_dtype)
    elif IS_MIN:
        acc = tl.full((BLOCK_M, BLOCK_K), float("inf"), dtype=compute_dtype)
    elif IS_PROD:
        acc = tl.full((BLOCK_M, BLOCK_K), 1.0, dtype=compute_dtype)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=compute_dtype)

    has_nan = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.int1)
    nan_value = tl.zeros((BLOCK_M, BLOCK_K), dtype=compute_dtype)

    pos = 0
    while pos < segment_length:
        data_offsets = base_offsets + pos * inner_size
        if IS_SUM or IS_MEAN:
            values = tl.load(data + data_offsets, mask=mask, other=0.0).to(
                compute_dtype
            )
            acc += values
        elif IS_PROD:
            values = tl.load(data + data_offsets, mask=mask, other=1.0).to(
                compute_dtype
            )
            acc *= values
        elif IS_MAX:
            values = tl.load(data + data_offsets, mask=mask, other=float("-inf")).to(
                compute_dtype
            )
            nan_mask = (values != values) & mask
            has_nan |= nan_mask
            nan_value = tl.where(nan_mask, values, nan_value)
            acc = tl.maximum(acc, tl.where(mask & ~nan_mask, values, float("-inf")))
        elif IS_MIN:
            values = tl.load(data + data_offsets, mask=mask, other=float("inf")).to(
                compute_dtype
            )
            nan_mask = (values != values) & mask
            has_nan |= nan_mask
            nan_value = tl.where(nan_mask, values, nan_value)
            acc = tl.minimum(acc, tl.where(mask & ~nan_mask, values, float("inf")))
        pos += 1

    if IS_MEAN:
        acc = acc / segment_length
    if IS_MAX or IS_MIN:
        acc = tl.where(has_nan, nan_value, acc)

    output_offsets = rows * inner_size + k_offsets
    tl.store(output + output_offsets, acc, mask=mask)


def _segment_reduce_uniform_lengths(data, reduce, lengths, axis):
    segment_count = lengths.shape[-1]
    segment_length = _get_uniform_segment_length(data, lengths, axis)
    if segment_length is None:
        return None

    output_shape = lengths.shape + data.shape[axis + 1 :]
    inner_size = _prod(data.shape[axis + 1 :])
    if segment_length <= _UNIFORM_KERNEL_MAX_SEGMENT_LENGTH:
        output = torch.empty(output_shape, dtype=data.dtype, device=data.device)
        if output.numel() == 0:
            return output
        total_rows = _prod(lengths.shape)
        if inner_size == 1:
            block_m = 4 if data.device.type == "npu" else 32
            block_n = min(
                _get_block_size(data.device),
                triton.next_power_of_2(segment_length),
            )
            grid = (triton.cdiv(total_rows, block_m),)
            with torch_device_fn.device(data.device):
                _segment_reduce_uniform_inner1_forward_kernel[grid](
                    data,
                    output,
                    total_rows,
                    segment_count,
                    segment_length,
                    data.shape[axis],
                    reduce == "sum",
                    reduce == "mean",
                    reduce == "max",
                    reduce == "min",
                    reduce == "prod",
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                )
            return output

        block_m, block_k = _get_uniform_kernel_config(data.device, inner_size)
        grid = (triton.cdiv(total_rows, block_m), triton.cdiv(inner_size, block_k))
        with torch_device_fn.device(data.device):
            _segment_reduce_uniform_forward_kernel[grid](
                data,
                output,
                total_rows,
                segment_count,
                segment_length,
                inner_size,
                data.shape[axis],
                reduce == "sum",
                reduce == "mean",
                reduce == "max",
                reduce == "min",
                reduce == "prod",
                BLOCK_M=block_m,
                BLOCK_K=block_k,
            )
        return output

    if data.device.type == "npu":
        return None

    view_shape = (
        data.shape[:axis] + (segment_count, segment_length) + data.shape[axis + 1 :]
    )
    reshaped = data.reshape(view_shape)
    reduce_dim = axis + 1

    if segment_length == 1:
        return torch.squeeze(reshaped, dim=reduce_dim)
    if reduce == "sum":
        return torch.sum(reshaped, dim=reduce_dim)
    if reduce == "mean":
        return torch.mean(reshaped, dim=reduce_dim)
    if reduce == "max":
        return torch.amax(reshaped, dim=reduce_dim)
    if reduce == "min":
        return torch.amin(reshaped, dim=reduce_dim)
    return torch.prod(reshaped, dim=reduce_dim)


def _segment_reduce_uniform_sum_mean_backward(data, grad, reduce, lengths, axis):
    segment_count = lengths.shape[-1]
    segment_length = _get_uniform_segment_length(data, lengths, axis)
    if segment_length is None:
        return None

    grad_input = torch.empty_like(data, dtype=grad.dtype)
    if grad_input.numel() == 0:
        return grad_input

    inner_size = _prod(data.shape[axis + 1 :])
    block_size = _get_block_size(data.device)
    grid = (triton.cdiv(data.numel(), block_size),)
    with torch_device_fn.device(data.device):
        _segment_reduce_uniform_sum_mean_backward_kernel[grid](
            grad,
            grad_input,
            data.numel(),
            segment_count,
            segment_length,
            inner_size,
            data.shape[axis],
            reduce == "mean",
            BLOCK_SIZE=block_size,
        )
    return grad_input


def _segment_reduce_uniform_other_backward(
    data, output, grad, reduce, lengths, axis, initial
):
    segment_count = lengths.shape[-1]
    segment_length = _get_uniform_segment_length(data, lengths, axis)
    if segment_length is None or segment_length > _UNIFORM_KERNEL_MAX_SEGMENT_LENGTH:
        return None

    if reduce in ("max", "min"):
        grad_input = torch.zeros_like(data, dtype=grad.dtype)
    else:
        grad_input = torch.empty_like(data, dtype=grad.dtype)
    if grad_input.numel() == 0:
        return grad_input

    inner_size = _prod(data.shape[axis + 1 :])
    total_rows = _prod(lengths.shape)
    block_m, block_k = _get_uniform_backward_tile_config(
        data.device, inner_size, reduce, data.dtype
    )
    block_n = min(_get_block_size(data.device), triton.next_power_of_2(segment_length))
    _, initial_prod_value = _make_initial("prod", initial)
    grid = (triton.cdiv(total_rows, block_m), triton.cdiv(inner_size, block_k))
    with torch_device_fn.device(data.device):
        _segment_reduce_uniform_other_backward_kernel[grid](
            grad,
            output,
            data,
            grad_input,
            total_rows,
            segment_count,
            segment_length,
            inner_size,
            data.shape[axis],
            reduce == "max",
            reduce == "min",
            reduce == "prod",
            initial_prod_value,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )
    return grad_input


@libentry()
@triton.jit
def _lengths_to_offsets_kernel(
    lengths,
    offsets,
    outer_count,
    segment_count,
):
    pid = tle.program_id(0)
    acc = tl.full((), 0, dtype=tl.int64)
    base_lengths = pid * segment_count
    base_offsets = pid * (segment_count + 1)
    tl.store(offsets + base_offsets, acc)

    idx = 0
    while idx < segment_count:
        length = tl.load(lengths + base_lengths + idx)
        acc += length
        tl.store(offsets + base_offsets + idx + 1, acc)
        idx += 1


@libentry()
@triton.jit
def _segment_reduce_forward_kernel(
    data,
    offsets,
    output,
    segment_count,
    inner_size,
    data_size_axis,
    IS_SUM: tl.constexpr,
    IS_MEAN: tl.constexpr,
    IS_MAX: tl.constexpr,
    IS_MIN: tl.constexpr,
    IS_PROD: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    INITIAL_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32

    inner_idx = pid % inner_size
    row_idx = pid // inner_size
    dim_idx = row_idx % segment_count
    outer_idx = row_idx // segment_count

    offsets_base = outer_idx * (segment_count + 1) + dim_idx
    segment_start = tl.load(offsets + offsets_base)
    segment_end = tl.load(offsets + offsets_base + 1)
    segment_length = segment_end - segment_start

    acc = tl.full((), INITIAL_VALUE, dtype=compute_dtype)
    if IS_PROD:
        pos = segment_start
        while pos < segment_end:
            data_offset = (
                outer_idx * data_size_axis * inner_size + pos * inner_size + inner_idx
            )
            value = tl.load(data + data_offset).to(compute_dtype)
            acc *= value
            pos += 1
    else:
        pos = segment_start
        while pos < segment_end:
            segment_offsets = pos + tl.arange(0, BLOCK_SIZE)
            mask = segment_offsets < segment_end
            data_offsets = (
                outer_idx * data_size_axis * inner_size
                + segment_offsets * inner_size
                + inner_idx
            )

            if IS_SUM or IS_MEAN:
                values = tl.load(data + data_offsets, mask=mask, other=0.0).to(
                    compute_dtype
                )
                acc += tl.sum(tl.where(mask, values, 0.0), axis=0)
            elif IS_MAX:
                values = tl.load(
                    data + data_offsets, mask=mask, other=float("-inf")
                ).to(compute_dtype)
                nan_mask = (values != values) & mask
                has_nan = tl.sum(nan_mask.to(tl.int32), axis=0) > 0
                nan_value = tl.sum(tl.where(nan_mask, values, 0.0), axis=0)
                chunk = tl.max(
                    tl.where(mask & ~nan_mask, values, float("-inf")), axis=0
                )
                chunk = tl.where(has_nan, nan_value, chunk)
                acc = tl.where(has_nan, chunk, tl.maximum(acc, chunk))
            elif IS_MIN:
                values = tl.load(data + data_offsets, mask=mask, other=float("inf")).to(
                    compute_dtype
                )
                nan_mask = (values != values) & mask
                has_nan = tl.sum(nan_mask.to(tl.int32), axis=0) > 0
                nan_value = tl.sum(tl.where(nan_mask, values, 0.0), axis=0)
                chunk = tl.min(tl.where(mask & ~nan_mask, values, float("inf")), axis=0)
                chunk = tl.where(has_nan, nan_value, chunk)
                acc = tl.where(has_nan, chunk, tl.minimum(acc, chunk))
            pos += BLOCK_SIZE

    if IS_MEAN:
        acc_is_nan = acc != acc
        nan_value = acc / acc
        if not HAS_INITIAL:
            acc = tl.where(segment_length == 0, nan_value, acc)
        acc = tl.where((segment_length > 0) & ~acc_is_nan, acc / segment_length, acc)

    tl.store(output + pid, acc)


@libentry()
@triton.jit
def _segment_reduce_backward_kernel(
    grad,
    output,
    data,
    offsets,
    grad_input,
    segment_count,
    inner_size,
    data_size_axis,
    IS_SUM: tl.constexpr,
    IS_MEAN: tl.constexpr,
    IS_MAX: tl.constexpr,
    IS_MIN: tl.constexpr,
    IS_PROD: tl.constexpr,
    INITIAL_PROD_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32

    inner_idx = pid % inner_size
    row_idx = pid // inner_size
    dim_idx = row_idx % segment_count
    outer_idx = row_idx // segment_count

    offsets_base = outer_idx * (segment_count + 1) + dim_idx
    segment_start = tl.load(offsets + offsets_base)
    segment_end = tl.load(offsets + offsets_base + 1)
    segment_length = segment_end - segment_start

    if segment_length > 0:
        grad_value = tl.load(grad + pid).to(compute_dtype)
        output_value = tl.load(output + pid).to(compute_dtype)

        if IS_SUM or IS_MEAN:
            if IS_MEAN:
                grad_value = grad_value / segment_length
            pos = segment_start
            while pos < segment_end:
                segment_offsets = pos + tl.arange(0, BLOCK_SIZE)
                mask = segment_offsets < segment_end
                data_offsets = (
                    outer_idx * data_size_axis * inner_size
                    + segment_offsets * inner_size
                    + inner_idx
                )
                tl.store(grad_input + data_offsets, grad_value, mask=mask)
                pos += BLOCK_SIZE
        elif IS_MAX or IS_MIN:
            counter = tl.full((), 0, dtype=tl.int64)
            pos = segment_start
            while pos < segment_end:
                segment_offsets = pos + tl.arange(0, BLOCK_SIZE)
                mask = segment_offsets < segment_end
                data_offsets = (
                    outer_idx * data_size_axis * inner_size
                    + segment_offsets * inner_size
                    + inner_idx
                )
                values = tl.load(data + data_offsets, mask=mask, other=0.0).to(
                    compute_dtype
                )
                match = ((values != values) | (values == output_value)) & mask
                counter += tl.sum(match.to(tl.int64), axis=0)
                pos += BLOCK_SIZE

            store_value = tl.where(
                (counter >= 2) & (grad_value > 0),
                grad_value / counter,
                grad_value,
            )
            pos = segment_start
            while pos < segment_end:
                segment_offsets = pos + tl.arange(0, BLOCK_SIZE)
                mask = segment_offsets < segment_end
                data_offsets = (
                    outer_idx * data_size_axis * inner_size
                    + segment_offsets * inner_size
                    + inner_idx
                )
                values = tl.load(data + data_offsets, mask=mask, other=0.0).to(
                    compute_dtype
                )
                match = ((values != values) | (values == output_value)) & mask
                tl.store(grad_input + data_offsets, store_value, mask=match)
                pos += BLOCK_SIZE
        elif IS_PROD:
            zero_count = tl.full((), 0, dtype=tl.int64)
            nan_count = tl.full((), 0, dtype=tl.int64)
            product = tl.full((), INITIAL_PROD_VALUE, dtype=compute_dtype)
            pos = segment_start
            while pos < segment_end:
                data_offset = (
                    outer_idx * data_size_axis * inner_size
                    + pos * inner_size
                    + inner_idx
                )
                value = tl.load(data + data_offset).to(compute_dtype)
                if value != value:
                    nan_count += 1
                elif value == 0:
                    zero_count += 1
                else:
                    product *= value
                pos += 1

            zero_scalar = tl.full((), 0.0, dtype=compute_dtype)
            nan_scalar = zero_scalar / zero_scalar
            normal_prefix = grad_value * output_value
            pos = segment_start
            while pos < segment_end:
                segment_offsets = pos + tl.arange(0, BLOCK_SIZE)
                mask = segment_offsets < segment_end
                data_offsets = (
                    outer_idx * data_size_axis * inner_size
                    + segment_offsets * inner_size
                    + inner_idx
                )
                values = tl.load(data + data_offsets, mask=mask, other=1.0).to(
                    compute_dtype
                )
                nan_mask = (values != values) & mask
                zero_mask = (values == 0) & mask & ~nan_mask
                normal_grad = normal_prefix / values
                zero_exclusive = tl.where(
                    nan_count > 0,
                    nan_scalar,
                    tl.where(zero_count > 1, zero_scalar, product),
                )
                nan_exclusive = tl.where(
                    nan_count > 1,
                    nan_scalar,
                    tl.where(zero_count > 0, zero_scalar, product),
                )
                exclusive = tl.where(nan_mask, nan_exclusive, zero_exclusive)
                grad_result = tl.where(
                    nan_mask | zero_mask,
                    grad_value * exclusive,
                    normal_grad,
                )
                tl.store(grad_input + data_offsets, grad_result, mask=mask)
                pos += BLOCK_SIZE


def _lengths_to_offsets(lengths):
    segment_count = lengths.shape[-1]
    offsets_shape = lengths.shape[:-1] + (segment_count + 1,)
    offsets = torch.empty(offsets_shape, dtype=lengths.dtype, device=lengths.device)
    outer_count = _prod(lengths.shape[:-1])
    if offsets.numel() > 0:
        with torch_device_fn.device(lengths.device):
            _lengths_to_offsets_kernel[(outer_count,)](
                lengths,
                offsets,
                outer_count,
                segment_count,
            )
    return offsets


def _prepare_common(data, reduce, lengths, offsets, indices, axis, unsafe):
    _check_reduce_and_dtype(data, reduce)
    axis = _wrap_axis(axis, data.dim())
    if indices is not None:
        raise RuntimeError(
            "segment_reduce(): indices based reduction is not supported yet."
        )

    if offsets is not None:
        _check_index_tensor(data, offsets, "offsets", axis)
        offsets_contig = offsets.contiguous()
        segment_count = offsets_contig.shape[-1] - 1
        output_shape = (
            offsets_contig.shape[:-1] + (segment_count,) + data.shape[axis + 1 :]
        )
        return axis, offsets_contig, output_shape, True

    if lengths is None:
        raise RuntimeError(
            "segment_reduce(): Either lengths or offsets must be defined."
        )

    _validate_lengths(data, lengths, axis, unsafe)
    lengths_contig = lengths.contiguous()
    offsets_contig = _lengths_to_offsets(lengths_contig)
    output_shape = lengths_contig.shape + data.shape[axis + 1 :]
    return axis, offsets_contig, output_shape, False


def segment_reduce(
    data,
    reduce,
    *,
    lengths=None,
    indices=None,
    offsets=None,
    axis=0,
    unsafe=False,
    initial=None,
):
    logger.debug("GEMS SEGMENT_REDUCE")
    _check_reduce_and_dtype(data, reduce)
    axis = _wrap_axis(axis, data.dim())
    if indices is not None:
        raise RuntimeError(
            "segment_reduce(): indices based reduction is not supported yet."
        )

    if initial is None and lengths is not None and offsets is None:
        _check_index_tensor(data, lengths, "lengths", axis)
        if _is_unit_lengths(data, lengths, axis):
            return data.contiguous()

        data_contig = data.contiguous()
        uniform_result = _segment_reduce_uniform_lengths(
            data_contig, reduce, lengths, axis
        )
        if uniform_result is not None:
            return uniform_result

    axis, offsets_contig, output_shape, _ = _prepare_common(
        data, reduce, lengths, offsets, indices, axis, unsafe
    )

    data_contig = data.contiguous()
    output = torch.empty(output_shape, dtype=data.dtype, device=data.device)
    if output.numel() == 0:
        return output

    segment_count = output_shape[axis]
    inner_size = _prod(data_contig.shape[axis + 1 :])
    data_size_axis = data_contig.shape[axis]
    has_initial, initial_value = _make_initial(reduce, initial)
    grid = (output.numel(),)

    with torch_device_fn.device(data.device):
        _segment_reduce_forward_kernel[grid](
            data_contig,
            offsets_contig,
            output,
            segment_count,
            inner_size,
            data_size_axis,
            reduce == "sum",
            reduce == "mean",
            reduce == "max",
            reduce == "min",
            reduce == "prod",
            has_initial,
            initial_value,
            BLOCK_SIZE=_get_block_size(data.device),
        )
    return output


def segment_reduce_out(
    data,
    reduce,
    *,
    lengths=None,
    indices=None,
    offsets=None,
    axis=0,
    unsafe=False,
    initial=None,
    out,
):
    logger.debug("GEMS SEGMENT_REDUCE_OUT")
    result = segment_reduce(
        data,
        reduce,
        lengths=lengths,
        indices=indices,
        offsets=offsets,
        axis=axis,
        unsafe=unsafe,
        initial=initial,
    )
    if out.shape != result.shape:
        out.resize_(result.shape)
    out.copy_(result)
    return out


def _segment_reduce_backward(
    grad,
    output,
    data,
    reduce,
    *,
    lengths=None,
    offsets=None,
    axis=0,
    initial=None,
):
    logger.debug("GEMS _SEGMENT_REDUCE_BACKWARD")
    if (
        initial is None
        and lengths is not None
        and offsets is None
        and reduce in _SUPPORTED_REDUCES
    ):
        _check_reduce_and_dtype(data, reduce)
        axis = _wrap_axis(axis, data.dim())
        _check_index_tensor(data, lengths, "lengths", axis)
        if _is_unit_lengths(data, lengths, axis):
            return grad.contiguous()

    if lengths is not None and offsets is None and reduce in ("sum", "mean"):
        _check_reduce_and_dtype(data, reduce)
        axis = _wrap_axis(axis, data.dim())
        _check_index_tensor(data, lengths, "lengths", axis)
        data_contig = data.contiguous()
        grad_contig = grad.contiguous()
        uniform_result = _segment_reduce_uniform_sum_mean_backward(
            data_contig, grad_contig, reduce, lengths, axis
        )
        if uniform_result is not None:
            return uniform_result
    if lengths is not None and offsets is None and reduce in ("max", "min", "prod"):
        _check_reduce_and_dtype(data, reduce)
        axis = _wrap_axis(axis, data.dim())
        _check_index_tensor(data, lengths, "lengths", axis)
        data_contig = data.contiguous()
        grad_contig = grad.contiguous()
        output_contig = output.contiguous()
        uniform_result = _segment_reduce_uniform_other_backward(
            data_contig, output_contig, grad_contig, reduce, lengths, axis, initial
        )
        if uniform_result is not None:
            return uniform_result

    axis, offsets_contig, output_shape, _ = _prepare_common(
        data, reduce, lengths, offsets, None, axis, True
    )
    data_contig = data.contiguous()
    grad_contig = grad.contiguous()
    output_contig = output.contiguous()
    grad_input = torch.zeros(data_contig.shape, dtype=grad.dtype, device=grad.device)

    if output_contig.numel() == 0:
        return grad_input

    segment_count = output_shape[axis]
    inner_size = _prod(data_contig.shape[axis + 1 :])
    data_size_axis = data_contig.shape[axis]
    _, initial_prod_value = _make_initial("prod", initial)
    grid = (output_contig.numel(),)

    with torch_device_fn.device(data.device):
        _segment_reduce_backward_kernel[grid](
            grad_contig,
            output_contig,
            data_contig,
            offsets_contig,
            grad_input,
            segment_count,
            inner_size,
            data_size_axis,
            reduce == "sum",
            reduce == "mean",
            reduce == "max",
            reduce == "min",
            reduce == "prod",
            initial_prod_value,
            BLOCK_SIZE=_get_block_size(data.device),
        )
    return grad_input


def _segment_reduce_backward_out(
    grad,
    output,
    data,
    reduce,
    *,
    lengths=None,
    offsets=None,
    axis=0,
    initial=None,
    out,
):
    logger.debug("GEMS _SEGMENT_REDUCE_BACKWARD_OUT")
    result = _segment_reduce_backward(
        grad,
        output,
        data,
        reduce,
        lengths=lengths,
        offsets=offsets,
        axis=axis,
        initial=initial,
    )
    if out.shape != result.shape:
        out.resize_(result.shape)
    out.copy_(result)
    return out
