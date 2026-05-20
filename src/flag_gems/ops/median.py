import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_max

logger = logging.getLogger(__name__)

MedianResult = namedtuple("median", ["values", "indices"])

_SORT_MAX_BLOCK = 4096
_CUDA_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)

_MAX_FP16 = tl.constexpr(torch.finfo(torch.float16).max)
_MAX_BF16 = tl.constexpr(torch.finfo(torch.bfloat16).max)
_MAX_FP32 = tl.constexpr(torch.finfo(torch.float32).max)
_MAX_FP64 = tl.constexpr(torch.finfo(torch.float64).max)


@triton.jit
def _finite_max_scalar(dtype: tl.constexpr):
    dtype_ = dtype.value
    if dtype_ == tl.float16:
        return tl.full((), _MAX_FP16, dtype=tl.float16)
    if dtype_ == tl.bfloat16:
        return tl.full((), _MAX_BF16, dtype=tl.bfloat16)
    if dtype_ == tl.float32:
        return tl.full((), _MAX_FP32, dtype=tl.float32)
    if dtype_ == tl.float64:
        return tl.full((), _MAX_FP64, dtype=tl.float64)
    return tl.full((), get_dtype_max(dtype), dtype=dtype)


@libentry()
@triton.jit
def _median_sort_kernel(
    inp_ptr,
    val_ptr,
    idx_ptr,
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    cols = tl.arange(0, BLOCK_N)
    valid = cols < N
    base = inp_ptr + row * N

    dtype = inp_ptr.dtype.element_ty
    is_float = dtype.is_floating()
    pad = _finite_max_scalar(dtype)

    data = tl.load(base + cols, mask=valid, other=pad)

    if is_float:
        nan_lane = valid & (data != data)
    else:
        nan_lane = valid & False
    sortable = tl.where(nan_lane, pad, data)

    ordered = tl.sort(sortable, dim=0, descending=False)
    rank = (N - 1) // 2
    rank_mask = cols == rank
    median_val = tl.sum(tl.where(rank_mask, ordered, tl.zeros_like(ordered)), axis=0)

    match = valid & (data == median_val)
    first_match = tl.argmax(match.to(tl.int32), axis=0)

    if is_float:
        nan_i32 = nan_lane.to(tl.int32)
        any_nan = tl.max(nan_i32, axis=0) != 0
        first_nan = tl.argmax(nan_i32, axis=0)
        nan_val = tl.load(base + first_nan, mask=any_nan, other=median_val)
        median_val = tl.where(any_nan, nan_val, median_val)
        first_match = tl.where(any_nan, first_nan, first_match)

    tl.store(val_ptr + row, median_val)
    tl.store(idx_ptr + row, first_match.to(tl.int64))


@libentry()
@triton.jit
def _median_multi_row_sort_kernel(
    inp_ptr,
    val_ptr,
    idx_ptr,
    M,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < M

    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    valid = row_mask[:, None] & col_mask[None, :]

    dtype = inp_ptr.dtype.element_ty
    is_float = dtype.is_floating()
    pad = _finite_max_scalar(dtype)

    base = inp_ptr + row_offsets[:, None] * N + cols[None, :]
    data = tl.load(base, mask=valid, other=pad)

    if is_float:
        nan_lane = valid & (data != data)
    else:
        nan_lane = valid & False
    sortable = tl.where(nan_lane, pad, data)

    ordered = tl.sort(sortable, dim=1, descending=False)
    rank = (N - 1) // 2
    rank_mask = cols[None, :] == rank
    median_val = tl.sum(tl.where(rank_mask, ordered, tl.zeros_like(ordered)), axis=1)

    match = valid & (data == median_val[:, None])
    first_match = tl.argmax(match.to(tl.int32), axis=1)

    if is_float:
        nan_i32 = nan_lane.to(tl.int32)
        any_nan = tl.max(nan_i32, axis=1) != 0
        first_nan = tl.argmax(nan_i32, axis=1)
        nan_row_base = inp_ptr + row_offsets * N + first_nan
        zero_pad = tl.zeros([BLOCK_M], dtype=median_val.dtype)
        nan_val = tl.load(nan_row_base, mask=row_mask & any_nan, other=zero_pad)
        median_val = tl.where(any_nan, nan_val, median_val)
        first_match = tl.where(any_nan, first_nan, first_match)

    tl.store(val_ptr + row_offsets, median_val, mask=row_mask)
    tl.store(idx_ptr + row_offsets, first_match.to(tl.int64), mask=row_mask)


_SORT_UPCAST = {
    torch.bfloat16: torch.float32,
    torch.int8: torch.int32,
    torch.uint8: torch.int32,
    torch.int16: torch.int32,
}


def _is_sort_supported(dtype):
    if dtype in (torch.float16, torch.float32):
        return True
    if dtype in _SORT_UPCAST:
        return True
    return dtype in (torch.int32, torch.int64)


def _launch_sort_kernel(inp_2d, values_flat, indices_flat):
    M, N = inp_2d.shape
    block_n = triton.next_power_of_2(N)

    if M >= 4 and block_n <= 1024:
        if block_n <= 64:
            block_m = 16
        elif block_n <= 256:
            block_m = 8
        elif block_n <= 512:
            block_m = 4
        else:
            block_m = 2
        if block_m * block_n <= 8192 and M >= block_m:
            num_warps = 4 if block_n <= 256 else 8
            grid = (triton.cdiv(M, block_m),)
            with torch_device_fn.device(inp_2d.device):
                _median_multi_row_sort_kernel[grid](
                    inp_2d,
                    values_flat,
                    indices_flat,
                    M,
                    N=N,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    num_warps=num_warps,
                )
            return

    num_warps = 4 if block_n <= 1024 else 8
    with torch_device_fn.device(inp_2d.device):
        _median_sort_kernel[(M,)](
            inp_2d,
            values_flat,
            indices_flat,
            M,
            N=N,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )


def _launch_sort(inp_2d, values_flat, indices_flat):
    upcast = _SORT_UPCAST.get(inp_2d.dtype)
    if upcast is not None:
        inp_up = inp_2d.to(upcast)
        tmp_val = torch.empty(values_flat.shape, dtype=upcast, device=inp_2d.device)
        _launch_sort_kernel(inp_up, tmp_val, indices_flat)
        values_flat.copy_(tmp_val.to(values_flat.dtype))
    else:
        _launch_sort_kernel(inp_2d, values_flat, indices_flat)


def _median_rows(inp_2d, out_values, out_indices):
    M, N = inp_2d.shape
    if N <= _SORT_MAX_BLOCK and _is_sort_supported(inp_2d.dtype):
        _launch_sort(inp_2d, out_values, out_indices)
        return

    inp = inp_2d
    sorted_vals, sorted_idx = torch.ops.aten.sort.default.redispatch(
        _CUDA_KEYSET, inp, -1, False
    )
    rank = (N - 1) // 2
    median_val = sorted_vals[..., rank]
    median_idx = sorted_idx[..., rank].to(torch.int64)

    if inp.dtype.is_floating_point:
        any_nan_overall = torch.isnan(inp).any()
        if bool(any_nan_overall):
            nan_mask = torch.isnan(inp)
            any_nan = nan_mask.any(dim=-1)
            first_nan_idx = nan_mask.to(torch.int64).argmax(dim=-1)
            nan_vals = torch.gather(inp, -1, first_nan_idx.unsqueeze(-1)).squeeze(-1)
            median_val = torch.where(any_nan, nan_vals, median_val)
            median_idx = torch.where(any_nan, first_nan_idx, median_idx)

    out_values.copy_(median_val)
    out_indices.copy_(median_idx)


def _canonical_dim(ndim, dim):
    if ndim == 0:
        if dim not in (0, -1):
            raise IndexError(
                f"Dimension out of range (expected to be in range of [-1, 0], "
                f"but got {dim})"
            )
        return 0
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-ndim}, {ndim - 1}], but got {dim})"
        )
    return dim % ndim


def _empty_whole_result(inp):
    if inp.dtype.is_floating_point:
        return torch.full((), float("nan"), dtype=inp.dtype, device=inp.device)
    if inp.dtype == torch.bool:
        return torch.ones((), dtype=inp.dtype, device=inp.device)
    return torch.empty((), dtype=inp.dtype, device=inp.device)


def _check_dtype(inp):
    if inp.dtype == torch.bool:
        raise RuntimeError("median(): not implemented for dtype torch.bool")
    if inp.dtype.is_complex:
        raise RuntimeError("median(): not implemented for complex dtypes")


def median(inp):
    logger.debug("GEMS MEDIAN")
    _check_dtype(inp)
    if inp.numel() == 0:
        return _empty_whole_result(inp)
    if inp.numel() == 1:
        return inp.reshape(()).clone()

    N = inp.numel()
    flat = inp.contiguous().reshape(1, N)
    values = torch.empty((), dtype=inp.dtype, device=inp.device)
    indices = torch.empty((), dtype=torch.int64, device=inp.device)
    _median_rows(flat, values.reshape(1), indices.reshape(1))
    return values


def median_out(inp, *, out):
    logger.debug("GEMS MEDIAN.OUT")
    result = median(inp)
    if out.shape != result.shape:
        out.resize_as_(result)
    out.copy_(result)
    return out


def median_dim(inp, dim=-1, keepdim=False):
    logger.debug("GEMS MEDIAN.DIM")
    _check_dtype(inp)

    if inp.ndim == 0:
        _canonical_dim(0, dim)
        return MedianResult(
            values=inp.clone(),
            indices=torch.zeros((), dtype=torch.int64, device=inp.device),
        )

    dim = _canonical_dim(inp.ndim, dim)
    if inp.shape[dim] == 0:
        if inp.numel() == 0:
            out_shape = list(inp.shape)
            if keepdim:
                out_shape[dim] = 1
            else:
                del out_shape[dim]
            return MedianResult(
                values=torch.empty(out_shape, dtype=inp.dtype, device=inp.device),
                indices=torch.empty(out_shape, dtype=torch.int64, device=inp.device),
            )
        raise IndexError(
            f"median(): Expected reduction dim {dim} to have non-zero size."
        )

    out_shape = list(inp.shape)
    if keepdim:
        out_shape[dim] = 1
    else:
        del out_shape[dim]

    if dim == inp.ndim - 1 and inp.is_contiguous():
        work = inp
        target_shape = out_shape if keepdim else inp.shape[:-1]
    else:
        work = torch.movedim(inp, dim, -1).contiguous()
        target_shape = work.shape[:-1]

    N = work.shape[-1]
    M = work.numel() // N
    values = torch.empty(target_shape, dtype=inp.dtype, device=inp.device)
    indices = torch.empty(target_shape, dtype=torch.int64, device=inp.device)
    _median_rows(work.reshape(M, N), values.reshape(-1), indices.reshape(-1))

    if dim != inp.ndim - 1 or not inp.is_contiguous():
        if keepdim:
            values = torch.movedim(values.unsqueeze(-1), -1, dim)
            indices = torch.movedim(indices.unsqueeze(-1), -1, dim)
    return MedianResult(values=values, indices=indices)


def median_dim_values(inp, dim=-1, keepdim=False, *, values, indices):
    logger.debug("GEMS MEDIAN.DIM_VALUES")
    result = median_dim(inp, dim=dim, keepdim=keepdim)
    if values.shape != result.values.shape:
        values.resize_as_(result.values)
    if indices.shape != result.indices.shape:
        indices.resize_as_(result.indices)
    values.copy_(result.values)
    indices.copy_(result.indices)
    return MedianResult(values=values, indices=indices)
