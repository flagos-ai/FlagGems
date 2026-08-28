import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def nanmean_kernel_1(
    inp,
    mid_sum,
    mid_cnt,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float64):
        cdtype = tl.float64
    elif tl.constexpr(mid_sum.dtype.element_ty == tl.float64):
        cdtype = tl.float64
    else:
        cdtype = tl.float32

    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M

    x = tl.load(inp_ptrs, mask=mask, other=0.0).to(cdtype)
    is_nan = x != x
    valid = mask & (~is_nan)
    x = tl.where(valid, x, 0.0)

    sum_val = tl.sum(x, axis=0)
    cnt_val = tl.sum(valid.to(cdtype), axis=0)
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
    if tl.constexpr(mid_sum.dtype.element_ty == tl.float64):
        cdtype = tl.float64
    else:
        cdtype = tl.float32

    _sum = tl.zeros((), dtype=cdtype)
    _cnt = tl.zeros((), dtype=cdtype)

    for start in range(0, mid_size, BLOCK_SIZE):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < mid_size
        sv = tl.load(mid_sum + idx, mask=mask, other=0.0).to(cdtype)
        cv = tl.load(mid_cnt + idx, mask=mask, other=0.0).to(cdtype)
        _sum += tl.sum(sv, axis=0)
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
    if tl.constexpr(inp.dtype.element_ty == tl.float64):
        cdtype = tl.float64
    elif tl.constexpr(out.dtype.element_ty == tl.float64):
        cdtype = tl.float64
    else:
        cdtype = tl.float32

    _sum = tl.zeros((), dtype=cdtype)
    _cnt = tl.zeros((), dtype=cdtype)

    for start in range(0, M, BLOCK_SIZE):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < M
        x = tl.load(inp + idx, mask=mask, other=0.0).to(cdtype)
        is_nan = x != x
        valid = mask & (~is_nan)
        x = tl.where(valid, x, 0.0)
        _sum += tl.sum(x, axis=0)
        _cnt += tl.sum(valid.to(cdtype), axis=0)

    tl.store(out, _sum / _cnt)


def _compute_dtype(inp, dtype):
    if inp.dtype == torch.float64 or dtype == torch.float64:
        return torch.float64
    return torch.float32


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

    compute_dtype = _compute_dtype(inp, dtype)
    block_size = max(triton.next_power_of_2(math.ceil(math.sqrt(M))), 4096)
    mid_size = triton.cdiv(M, block_size)
    mid_sum = torch.empty(mid_size, dtype=compute_dtype, device=inp.device)
    mid_cnt = torch.empty(mid_size, dtype=compute_dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        nanmean_kernel_1[(mid_size, 1, 1)](inp, mid_sum, mid_cnt, M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        nanmean_kernel_2[(1,)](mid_sum, mid_cnt, out, mid_size, BLOCK_SIZE=block_mid)

    return out


def _dim_block_n(args):
    tile_budget = 4096
    tile_n = min(args["N"], tile_budget // args["BLOCK_K"])
    return max(1, triton.next_power_of_2(tile_n))


def _dim_block_k(M, K):
    num_sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    target_waves = 4 if M <= 4 else 2
    target_blocks = target_waves * num_sms
    ideal = max(1, triton.cdiv(M * K, target_blocks))
    return min(8192, triton.next_power_of_2(K), triton.next_power_of_2(ideal))


@libentry()
@triton.heuristics(values={"BLOCK_N": _dim_block_n})
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
    if tl.constexpr(inp.dtype.element_ty == tl.float64):
        cdtype = tl.float64
    elif tl.constexpr(out.dtype.element_ty == tl.float64):
        cdtype = tl.float64
    else:
        cdtype = tl.float32

    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
    k_mask = k < K

    sum_acc = tl.zeros([BLOCK_N, BLOCK_K], dtype=cdtype)
    count_acc = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.int32)

    for start_n in range(0, N, BLOCK_N):
        n = start_n + tl.arange(0, BLOCK_N)[:, None]
        mask = (n < N) & k_mask
        offsets = pid_m * N * K + n * K + k
        val = tl.load(inp + offsets, mask=mask, other=0.0).to(cdtype)
        valid = mask & (val == val)
        val = tl.where(valid, val, 0.0)
        sum_acc += val
        count_acc += valid.to(tl.int32)

    result = tl.sum(sum_acc, axis=0) / tl.sum(count_acc, axis=0)
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
    if tl.constexpr(inp.dtype.element_ty == tl.float64):
        cdtype = tl.float64
    elif tl.constexpr(out.dtype.element_ty == tl.float64):
        cdtype = tl.float64
    else:
        cdtype = tl.float32

    rows = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = rows < M
    sum_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
    count_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)

    for start_n in range(0, N, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask & (cols < N)
        val = tl.load(inp + rows * N + cols, mask=mask, other=0.0).to(cdtype)
        valid = mask & (val == val)
        sum_acc += tl.where(valid, val, 0.0)
        count_acc += valid.to(tl.int32)

    result = tl.sum(sum_acc, axis=1) / tl.sum(count_acc, axis=1)
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
    if inp.is_complex():
        valid = ~(torch.isnan(inp.real) | torch.isnan(inp.imag))
        factor = valid.detach().sum(dim=dim, keepdim=keepdim)
        real_dtype = (
            torch.float64
            if dtype in (torch.float64, torch.complex128)
            else torch.float32
        )
        real_mask = (
            ~torch.isnan(inp.real)
            if dtype is not None and not dtype.is_complex
            else valid
        )
        real_sum = torch.where(real_mask, inp.real, 0).sum(
            dim=dim, keepdim=keepdim, dtype=real_dtype
        )
        real_mean = real_sum / factor
        if dtype is not None and not dtype.is_complex:
            return real_mean.to(dtype)
        imag_sum = torch.where(valid, inp.imag, 0).sum(
            dim=dim, keepdim=keepdim, dtype=real_dtype
        )
        return torch.complex(real_mean, imag_sum / factor).to(dtype or inp.dtype)
    if inp.requires_grad:
        cpu_inp = inp.to("cpu")
        return torch.nanmean(cpu_inp, dim=dim, keepdim=keepdim, dtype=dtype).to(
            inp.device
        )
    if dim is None:
        result = _nanmean_global(inp, dtype=dtype)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result
    return nanmean_dim(inp, dim=dim, keepdim=keepdim, dtype=dtype)


def nanmean_out(inp, dim=None, keepdim=False, *, dtype=None, out=None):
    logger.debug("GEMS NANMEAN_OUT")
    result = nanmean(inp, dim=dim, keepdim=keepdim, dtype=dtype)
    if out.shape != result.shape:
        out.resize_(result.shape)
    out.copy_(result)
    return out
