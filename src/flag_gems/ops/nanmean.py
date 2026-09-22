import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.copy import copy_
from flag_gems.ops.sum import sum_dim
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, pointwise_dynamic
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def nanmean_kernel_1(
    inp,
    mid_sum,
    mid_cnt,
    out,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(out.dtype.element_ty == tl.float64):
        value_dtype = tl.float64
        acc_dtype = tl.float64
    elif tl.constexpr(out.dtype.element_ty == tl.float16):
        value_dtype = tl.float16
        acc_dtype = tl.float32
    elif tl.constexpr(out.dtype.element_ty == tl.bfloat16):
        value_dtype = tl.bfloat16
        acc_dtype = tl.float32
    else:
        value_dtype = tl.float32
        acc_dtype = tl.float32

    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M

    x = tl.load(inp_ptrs, mask=mask, other=0.0).to(value_dtype)
    is_nan = x != x
    valid = mask & (~is_nan)
    x = tl.where(valid, x, 0.0)

    sum_val = tl.sum(x, axis=0, dtype=acc_dtype)
    cnt_val = tl.sum(valid.to(acc_dtype), axis=0)
    tl.store(mid_sum + pid, sum_val)
    tl.store(mid_cnt + pid, cnt_val)


@libentry()
@triton.jit
def nanmean_kernel_2(
    mid_sum,
    mid_cnt,
    out,
    mid_size,
    BLOCK_SIZE: tl.constexpr,
):
    acc_dtype = (
        tl.float64
        if tl.constexpr(mid_sum.dtype.element_ty == tl.float64)
        else tl.float32
    )

    _sum = tl.zeros((), dtype=acc_dtype)
    _cnt = tl.zeros((), dtype=acc_dtype)

    for start in range(0, mid_size, BLOCK_SIZE):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < mid_size
        sv = tl.load(mid_sum + idx, mask=mask, other=0.0).to(acc_dtype)
        cv = tl.load(mid_cnt + idx, mask=mask, other=0.0).to(acc_dtype)
        _sum += tl.sum(sv, axis=0, dtype=acc_dtype)
        _cnt += tl.sum(cv, axis=0)

    tl.store(out, _sum / _cnt)


@libentry()
@triton.jit
def nanmean_global_single_kernel(
    inp,
    out,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(out.dtype.element_ty == tl.float64):
        value_dtype = tl.float64
        acc_dtype = tl.float64
    elif tl.constexpr(out.dtype.element_ty == tl.float16):
        value_dtype = tl.float16
        acc_dtype = tl.float32
    elif tl.constexpr(out.dtype.element_ty == tl.bfloat16):
        value_dtype = tl.bfloat16
        acc_dtype = tl.float32
    else:
        value_dtype = tl.float32
        acc_dtype = tl.float32

    _sum = tl.zeros((), dtype=acc_dtype)
    _cnt = tl.zeros((), dtype=acc_dtype)

    for start in range(0, M, BLOCK_SIZE):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < M
        x = tl.load(inp + idx, mask=mask, other=0.0).to(value_dtype)
        is_nan = x != x
        valid = mask & (~is_nan)
        x = tl.where(valid, x, 0.0)
        _sum += tl.sum(x, axis=0, dtype=acc_dtype)
        _cnt += tl.sum(valid.to(acc_dtype), axis=0)

    tl.store(out, _sum / _cnt)


def _compute_dtype(dtype):
    return torch.float64 if dtype == torch.float64 else torch.float32


def _nanmean_global(inp, *, dtype=None):
    if dtype is None:
        dtype = inp.dtype

    if not inp.is_contiguous():
        inp = inp.contiguous()
    M = inp.numel()

    out = torch.empty([], dtype=dtype, device=inp.device)

    if M == 0:
        out.fill_(float("nan"))
        return out

    if M <= 32768:
        with torch_device_fn.device(inp.device):
            nanmean_global_single_kernel[(1,)](inp, out, M, BLOCK_SIZE=4096)
        return out

    compute_dtype = _compute_dtype(dtype)
    block_size = max(triton.next_power_of_2(math.ceil(math.sqrt(M))), 4096)
    mid_size = triton.cdiv(M, block_size)
    mid_sum = torch.empty(mid_size, dtype=compute_dtype, device=inp.device)
    mid_cnt = torch.empty(mid_size, dtype=compute_dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        nanmean_kernel_1[(mid_size, 1, 1)](inp, mid_sum, mid_cnt, out, M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        nanmean_kernel_2[(1,)](mid_sum, mid_cnt, out, mid_size, BLOCK_SIZE=block_mid)

    return out


def _dim_block_n(args):
    tile_budget = 4096
    tile_n = min(args["N"], tile_budget // args["BLOCK_K"])
    return max(1, triton.next_power_of_2(tile_n))


def _dim_block_k(M, K):
    num_sms = torch_device_fn.get_device_properties(
        torch_device_fn.current_device()
    ).multi_processor_count
    target_waves = 4 if M <= 4 else 2
    target_blocks = target_waves * num_sms
    ideal = max(1, triton.cdiv(M * K, target_blocks))
    return min(8192, triton.next_power_of_2(K), triton.next_power_of_2(ideal))


@libentry()
@triton.heuristics(
    values={
        "BLOCK_N": lambda args: max(
            1, triton.next_power_of_2(min(args["N"], 4096 // args["BLOCK_K"]))
        )
    }
)
@triton.jit
def nanmean_dim_non_inner_kernel(
    inp,
    out,
    M,
    N,
    K,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if tl.constexpr(out.dtype.element_ty == tl.float64):
        value_dtype = tl.float64
        acc_dtype = tl.float64
    elif tl.constexpr(out.dtype.element_ty == tl.float16):
        value_dtype = tl.float16
        acc_dtype = tl.float32
    elif tl.constexpr(out.dtype.element_ty == tl.bfloat16):
        value_dtype = tl.bfloat16
        acc_dtype = tl.float32
    else:
        value_dtype = tl.float32
        acc_dtype = tl.float32

    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
    k_mask = k < K

    sum_acc = tl.zeros([BLOCK_N, BLOCK_K], dtype=acc_dtype)
    count_acc = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.int32)

    for start_n in range(0, N, BLOCK_N):
        n = start_n + tl.arange(0, BLOCK_N)[:, None]
        mask = (n < N) & k_mask
        offsets = pid_m * N * K + n * K + k
        val = tl.load(inp + offsets, mask=mask, other=0.0).to(value_dtype)
        valid = mask & (val == val)
        val = tl.where(valid, val, 0.0)
        sum_acc += val
        count_acc += valid.to(tl.int32)

    result = tl.sum(sum_acc, axis=0, dtype=acc_dtype) / tl.sum(count_acc, axis=0)
    tl.store(out + pid_m * K + k, result[None, :], mask=k_mask)


@libentry()
@triton.jit
def nanmean_dim_inner_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if tl.constexpr(out.dtype.element_ty == tl.float64):
        value_dtype = tl.float64
        acc_dtype = tl.float64
    elif tl.constexpr(out.dtype.element_ty == tl.float16):
        value_dtype = tl.float16
        acc_dtype = tl.float32
    elif tl.constexpr(out.dtype.element_ty == tl.bfloat16):
        value_dtype = tl.bfloat16
        acc_dtype = tl.float32
    else:
        value_dtype = tl.float32
        acc_dtype = tl.float32

    rows = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = rows < M
    sum_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=acc_dtype)
    count_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)

    for start_n in range(0, N, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask & (cols < N)
        val = tl.load(inp + rows * N + cols, mask=mask, other=0.0).to(value_dtype)
        valid = mask & (val == val)
        sum_acc += tl.where(valid, val, 0.0)
        count_acc += valid.to(tl.int32)

    result = tl.sum(sum_acc, axis=1, dtype=acc_dtype) / tl.sum(count_acc, axis=1)
    tl.store(out + rows, result[:, None], mask=row_mask)


def _normalize_dims(dim, ndim):
    if isinstance(dim, (list, tuple)) and len(dim) == 0:
        return []
    if dim is None:
        return list(range(ndim))
    if isinstance(dim, int):
        dim = [dim]
    dims = []
    for d in dim:
        if ndim == 0:
            if d not in (-1, 0):
                raise IndexError(
                    f"Dimension out of range (expected to be in range of [-1, 0], but got {d})"
                )
            wrapped = 0
        else:
            if d < -ndim or d >= ndim:
                raise IndexError(
                    "Dimension out of range (expected to be in range of "
                    f"[-{ndim}, {ndim - 1}], but got {d})"
                )
            wrapped = d % ndim
        if wrapped in dims:
            raise RuntimeError(
                f"dim {wrapped} appears multiple times in the list of dims"
            )
        dims.append(wrapped)
    return sorted(dims, reverse=True)


def _squeeze_dims(result, dims):
    for d in sorted(dims, reverse=True):
        result = result.squeeze(dim=d)
    return result


def nanmean_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS NANMEAN DIM")
    if dtype is None:
        dtype = inp.dtype

    dims = _normalize_dims(dim, inp.ndim)

    if inp.ndim == 0:
        return _nanmean_global(inp, dtype=dtype)

    # dim=[] -> reduce all
    if len(dims) == 0:
        result = _nanmean_global(inp, dtype=dtype)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result

    # full-dimensional reduction -> delegate to global
    if len(dims) == inp.ndim:
        result = _nanmean_global(inp, dtype=dtype)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result

    shape = list(inp.shape)
    N = 1
    for d in dims:
        N *= shape[d]
        shape[d] = 1

    if N == 0:
        out = torch.full(shape, float("nan"), dtype=dtype, device=inp.device)
        return out if keepdim else _squeeze_dims(out, dims)

    if math.prod(shape) == 0:
        out = torch.empty(shape, dtype=dtype, device=inp.device)
        return out if keepdim else _squeeze_dims(out, dims)

    if len(dims) == 1:
        dim = dims[0]
        if not inp.is_contiguous():
            inp = inp.contiguous()
        M = math.prod(shape[:dim])
        K = inp.numel() // (M * N)
    else:
        inp = dim_compress(inp, dims)
        M = inp.numel() // N
        K = 1

    out = torch.empty(M * K, dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        if K > 1:
            block_k = _dim_block_k(M, K)
            grid = (M, triton.cdiv(K, block_k))
            nanmean_dim_non_inner_kernel[grid](inp, out, M, N, K, BLOCK_K=block_k)
        else:
            block_m = min(8, triton.next_power_of_2(M))
            block_n = max(1, min(1024, triton.next_power_of_2(N)))
            grid = (triton.cdiv(M, block_m),)
            nanmean_dim_inner_kernel[grid](
                inp, out, M, N, BLOCK_M=block_m, BLOCK_N=block_n
            )

    out = out.reshape(shape)
    return out if keepdim else _squeeze_dims(out, dims)


@pointwise_dynamic(
    is_tensor=[True, True, False, False, False],
    num_outputs=3,
    promotion_methods=[(0, 1, "DEFAULT")] * 3,
)
@triton.jit
def _masked_parts(
    real,
    imag,
    real_output: tl.constexpr,
    conjugated: tl.constexpr,
    complex_input: tl.constexpr,
):
    valid = real == real
    real_valid = valid
    if complex_input:
        valid = valid & (imag == imag)
    if not real_output:
        real_valid = valid
    if conjugated:
        imag = -imag
    if not complex_input:
        imag = tl.full(real.shape, 0, real.dtype)
    return tl.where(real_valid, real, 0), tl.where(valid, imag, 0), valid.to(real.dtype)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _mean_divide(total, count):
    return total / count


@pointwise_dynamic(promotion_methods=[(0, 1, 2, "DEFAULT")])
@triton.jit
def _mean_backward(grad, valid, count):
    # Multiplication follows division: an all-NaN slice must yield NaN gradients.
    return (grad / count) * valid


def _assemble_complex(real, imag, dtype):
    result = torch.empty(real.shape, device=real.device, dtype=dtype)
    parts = torch.view_as_real(result)
    copy_(parts[..., 0], real)
    copy_(parts[..., 1], imag)
    return result


def _masked_input_parts(inp, dtype):
    real_dtype = (
        torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32
    )
    real = torch.empty(inp.shape, dtype=real_dtype, device=inp.device)
    imag = torch.empty_like(real)
    valid = torch.empty_like(real)
    if inp.is_complex():
        physical = inp.conj() if inp.is_conj() else inp
        parts = torch.view_as_real(physical)
        xr, xi = parts[..., 0], parts[..., 1]
    else:
        xr, xi = inp, inp
    _masked_parts(
        xr,
        xi,
        not dtype.is_complex,
        inp.is_conj(),
        inp.is_complex(),
        out0=real,
        out1=imag,
        out2=valid,
    )
    return real, imag, valid


def _complex_mean(inp, dim, keepdim, dtype):
    dim = _normalize_dims(dim, inp.ndim)
    real, imag, valid = _masked_input_parts(inp, dtype)
    count = sum_dim(valid, dim=dim, keepdim=keepdim)
    real = _mean_divide(sum_dim(real, dim=dim, keepdim=keepdim), count)
    if not dtype.is_complex:
        result = torch.empty_like(real, dtype=dtype)
        copy_(result, real)
        return result
    imag = _mean_divide(sum_dim(imag, dim=dim, keepdim=keepdim), count)
    return _assemble_complex(real, imag, dtype)


class _NanmeanAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, dim, keepdim, dtype):
        dtype = dtype or inp.dtype
        dims = _normalize_dims(dim, inp.ndim)
        if not dims:
            dims = list(range(inp.ndim))
        ctx.dims = dims
        ctx.keepdim = keepdim
        ctx.input_dtype = inp.dtype
        _, _, valid = _masked_input_parts(inp, dtype)
        count = sum_dim(valid, dim=dims, keepdim=True)
        ctx.save_for_backward(valid, count)
        if inp.is_complex() or dtype.is_complex:
            return _complex_mean(inp, dim, keepdim, dtype)
        return nanmean(inp.detach(), dim, keepdim, dtype=dtype)

    @staticmethod
    def backward(ctx, grad):
        valid, count = ctx.saved_tensors
        if not ctx.keepdim:
            for dim in sorted(ctx.dims):
                grad = grad.unsqueeze(dim)
        if grad.is_complex():
            physical = grad.conj() if grad.is_conj() else grad
            parts = torch.view_as_real(physical)
            real = _mean_backward(parts[..., 0], valid, count)
            if not ctx.input_dtype.is_complex:
                result = torch.empty_like(valid, dtype=ctx.input_dtype)
                copy_(result, real)
                return result, None, None, None
            imag = _mean_backward(parts[..., 1], valid, count)
            if grad.is_conj():
                from .neg import neg

                imag = neg(imag)
            result = _assemble_complex(real, imag, ctx.input_dtype)
        else:
            result = torch.empty_like(valid, dtype=ctx.input_dtype)
            if ctx.input_dtype.is_complex:
                real = _mean_backward(grad, valid, count)
                from flag_gems.ops.zeros_like import zeros_like

                result = _assemble_complex(real, zeros_like(real), ctx.input_dtype)
            else:
                _mean_backward(grad, valid, count, out0=result)
        return result, None, None, None


def nanmean(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS NANMEAN")
    if not (inp.is_floating_point() or inp.is_complex()):
        raise NotImplementedError(
            "nanmean(): expected input to have floating point or complex dtype but got "
            f"{inp.dtype}"
        )
    if dtype is not None and not (dtype.is_floating_point or dtype.is_complex):
        raise RuntimeError(
            "nanmean(): could not infer output dtype. Optional dtype must be either "
            f"a floating point or complex dtype. Got: {dtype}"
        )
    complex_output = dtype is not None and dtype.is_complex
    if inp.requires_grad and torch.is_grad_enabled():
        return _NanmeanAutograd.apply(inp, dim, keepdim, dtype)
    if inp.is_complex() or complex_output:
        return _complex_mean(inp, dim, keepdim, dtype or inp.dtype)
    if dim is None:
        result = _nanmean_global(inp, dtype=dtype)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result
    return nanmean_dim(inp, dim=dim, keepdim=keepdim, dtype=dtype)


def nanmean_out(inp, dim=None, keepdim=False, *, dtype=None, out=None):
    logger.debug("GEMS NANMEAN_OUT")
    if out is None:
        raise RuntimeError("nanmean(): missing required out tensor")
    if torch.is_grad_enabled() and (inp.requires_grad or out.requires_grad):
        raise RuntimeError(
            "nanmean(): functions with out= arguments don't support automatic differentiation"
        )
    if out.device != inp.device:
        raise RuntimeError(
            "nanmean: expected result tensor to be on the same device as input"
        )
    if dtype is not None and dtype != out.dtype:
        raise RuntimeError(
            "nanmean: provided dtype must match dtype of result. Got "
            f"{out.dtype} and {dtype}."
        )
    result = nanmean(inp, dim=dim, keepdim=keepdim, dtype=dtype or out.dtype)
    if out.shape != result.shape:
        out.resize_(result.shape)
    if out.is_complex():
        out_parts = torch.view_as_real(out)
        result_parts = torch.view_as_real(result)
        copy_(out_parts, result_parts)
    else:
        copy_(out, result)
    return out
