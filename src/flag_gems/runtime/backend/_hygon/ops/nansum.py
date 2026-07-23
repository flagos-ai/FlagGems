import logging
import math
from functools import reduce

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _nan_to_zero_kernel(
    inp,
    out,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    val = tl.load(inp + offsets, mask=mask, other=0.0)
    val = tl.where(val != val, 0.0, val)
    tl.store(out + offsets, val, mask=mask)


@libentry()
@triton.jit
def nansum_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float64):
        cdtype = tl.float64
    else:
        cdtype = tl.float32

    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M

    x = tl.load(inp_ptrs, mask=mask, other=0.0).to(cdtype)
    x = tl.where(x != x, 0.0, x)

    sum_val = tl.sum(x, axis=0)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("nansum_final_reduce"),
    key=["mid_size"],
)
@triton.jit
def nansum_kernel_2(
    mid,
    out,
    mid_size,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(mid.dtype.element_ty == tl.float64):
        cdtype = tl.float64
    else:
        cdtype = tl.float32

    _sum = tl.zeros((), dtype=cdtype)

    for start in range(0, mid_size, BLOCK_SIZE):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < mid_size
        val = tl.load(mid + idx, mask=mask, other=0.0).to(cdtype)
        val = tl.where(val != val, 0.0, val)
        _sum += tl.sum(val, axis=0)

    tl.store(out, _sum)


def _nansum_global(inp, *, dtype=None):
    if dtype is None:
        dtype = inp.dtype
        if dtype == torch.bool:
            inp = inp.to(torch.int64)
            dtype = torch.int64

    if inp.numel() == 0:
        return torch.tensor(0, dtype=dtype, device=inp.device)

    if not inp.is_contiguous():
        inp = inp.contiguous()
    M = inp.numel()

    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        nansum_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        nansum_kernel_2[(1,)](mid, out, mid_size)
    return out


def nansum(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_HYGON NANSUM")
    if dim is None:
        return _nansum_global(inp, dtype=dtype)
    return nansum_dim(inp, dim=dim, keepdim=keepdim, dtype=dtype)


def nansum_out(inp, dim=None, keepdim=False, *, dtype=None, out=None):
    logger.debug("GEMS_HYGON NANSUM_OUT")
    result = nansum(inp, dim=dim, keepdim=keepdim, dtype=dtype)
    out.copy_(result)
    return out


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("nansum_dim"),
    key=["M", "N"],
)
@triton.jit
def nansum_dim_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float64):
        cdtype = tl.float64
    else:
        cdtype = tl.float32

    pid = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + pid * N
    out = out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        val = tl.load(inp + cols, mask, other=0.0).to(cdtype)
        val = tl.where(val != val, 0.0, val)
        _sum += val

    result = tl.sum(_sum, axis=1)[:, None]
    tl.store(out, result, row_mask)


def _nansum_heur_tile_k(args):
    MAX_TILE_K = 8192
    NUM_SMS = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count

    if args["M"] <= 4:
        MIN_TILE_K = 8
        if args["M"] <= 1 and args["K"] > 32768 and args["N"] > 256:
            if args["input_ptr"].dtype == torch.float32:
                TARGET_WAVES = 8
            else:
                # Prefer heavier blocks (TARGET_WAVES=4) unless iterations
                # per block would exceed 16, then use 8 for more parallelism.
                test_tk = MIN_TILE_K
                ub = min(args["K"], MAX_TILE_K)
                while test_tk <= ub:
                    n_blocks = args["M"] * triton.cdiv(args["K"], test_tk)
                    if n_blocks / NUM_SMS >= 4:
                        if test_tk * 2 <= ub:
                            test_tk *= 2
                        else:
                            break
                    else:
                        break
                test_tn = triton.cdiv(32768, test_tk)
                test_tn = min(test_tn, args["N"])
                test_tn = triton.next_power_of_2(test_tn + 1) // 2
                test_iters = triton.cdiv(args["N"], test_tn)
                TARGET_WAVES = 8 if test_iters > 16 else 4
        else:
            TARGET_WAVES = 4

        tile_k = MIN_TILE_K
        upper_bound = min(args["K"], MAX_TILE_K)
        best_tile_k = tile_k
        while tile_k <= upper_bound:
            num_blocks = args["M"] * triton.cdiv(args["K"], tile_k)
            num_waves = num_blocks / NUM_SMS
            if num_waves >= TARGET_WAVES:
                best_tile_k = tile_k
                if tile_k * 2 <= upper_bound:
                    tile_k *= 2
                else:
                    break
            else:
                break
        return best_tile_k
    elif args["M"] <= 64:
        TARGET_WAVES = 2
        tile_k = 1
        upper_bound = min(args["K"], MAX_TILE_K)
        prev_tile_k = tile_k
        while tile_k <= upper_bound:
            num_blocks = args["M"] * triton.cdiv(args["K"], tile_k)
            num_waves = num_blocks / NUM_SMS
            if num_waves > TARGET_WAVES:
                prev_tile_k = tile_k
                if tile_k * 2 <= upper_bound:
                    tile_k *= 2
                else:
                    break
            else:
                break
        return prev_tile_k
    else:
        tile_k = 1
        upper_bound = min(args["K"], MAX_TILE_K)
        while tile_k <= upper_bound:
            num_blocks = args["M"] * triton.cdiv(args["K"], tile_k)
            num_waves = num_blocks / NUM_SMS
            if (num_waves > 1) and (tile_k * 2 <= upper_bound):
                tile_k *= 2
            else:
                break
        return tile_k


def _nansum_heur_tile_n_non_inner(args):
    tile_budget = 32768 if args["M"] <= 1 else 8192
    tile_n = triton.cdiv(tile_budget, args["TILE_K"])
    if args["M"] <= 1:
        tile_n = min(tile_n, args["N"])
        return triton.next_power_of_2(tile_n + 1) // 2
    return tile_n


def _nansum_heur_one_tile_per_cta(args):
    return args["TILE_N"] >= args["N"]


def _nansum_heur_num_warps_non_inner(args):
    tile_size = args["TILE_N"] * args["TILE_K"]
    if args.get("ONE_TILE_PER_CTA") and args["M"] <= 1:
        return 4
    if tile_size < 2048:
        return 4
    elif tile_size < 4096:
        return 8
    else:
        return 16


def _nansum_heur_tile_n_inner(args):
    if args["N"] <= (32 * 1024):
        return triton.next_power_of_2(args["N"])
    else:
        return 4096


def _nansum_heur_tile_m_inner(args):
    NUM_SMS = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count

    if args["M"] <= NUM_SMS * 16 or args["N"] > 256:
        return 1

    TARGET_GRID = NUM_SMS * 6
    tile_m = triton.next_power_of_2(max(args["M"] // TARGET_GRID, 1))
    max_tile_m = max(16384 // args["TILE_N"], 1)
    return min(tile_m, max_tile_m, 256)


def _nansum_heur_num_warps_inner(args):
    tile_size = args["TILE_M"] * args["TILE_N"]
    if tile_size < 2048:
        return 4
    elif tile_size < 4096:
        return 8
    else:
        return 16


@libentry()
@triton.heuristics(
    values=dict(
        TILE_K=_nansum_heur_tile_k,
        TILE_N=_nansum_heur_tile_n_non_inner,
        ONE_TILE_PER_CTA=_nansum_heur_one_tile_per_cta,
        num_warps=_nansum_heur_num_warps_non_inner,
    ),
)
@triton.jit
def nansum_dim_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float64):
        cdtype = tl.float64
    else:
        cdtype = tl.float32

    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        inp_offset = pid_m * N * K + n_offsets * K + k_offsets
        mask = (n_offsets < N) & (k_offsets < K)
        inp = tl.load(input_ptr + inp_offset, mask=mask, other=0.0).to(cdtype)
        inp = tl.where(inp != inp, 0.0, inp)
        out = tl.sum(inp, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        tl.store(output_ptr + out_offset, out, mask=k_offsets < K)
    else:
        _sum = tl.zeros([TILE_N, TILE_K], dtype=cdtype)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            inp_offsets = pid_m * N * K + n_offsets * K + k_offsets
            mask = (n_offsets < N) & (k_offsets < K)
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0.0).to(cdtype)
            inp = tl.where(inp != inp, 0.0, inp)
            _sum += inp
        out = tl.sum(_sum, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        tl.store(output_ptr + out_offset, out, mask=k_offsets < K)


@libentry()
@triton.heuristics(
    values=dict(
        TILE_N=_nansum_heur_tile_n_inner,
        TILE_M=_nansum_heur_tile_m_inner,
        num_warps=_nansum_heur_num_warps_inner,
    ),
)
@triton.jit
def nansum_dim_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float64):
        cdtype = tl.float64
    else:
        cdtype = tl.float32

    pid = ext.program_id(0)
    m_start = pid * TILE_M + tl.arange(0, TILE_M)[:, None]
    m_mask = m_start < M

    _sum = tl.zeros([TILE_M, TILE_N], dtype=cdtype)
    for start_n in range(0, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)[None, :]
        inp_offsets = m_start * N + n_offsets
        mask = m_mask & (n_offsets < N)
        inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0.0).to(cdtype)
        inp = tl.where(inp != inp, 0.0, inp)
        _sum += inp

    result = tl.sum(_sum, axis=1)[:, None]
    tl.store(output_ptr + m_start, result, mask=m_mask)


def _normalize_dims(dim, ndim):
    if isinstance(dim, (list, tuple)) and len(dim) == 0:
        return []
    if dim is None:
        return list(range(ndim))
    if isinstance(dim, int):
        dim = [dim]
    dims = [d if d >= 0 else d + ndim for d in dim]
    return sorted(set(dims), reverse=True)


def _squeeze_dims(result, dims):
    for d in sorted(dims, reverse=True):
        result = result.squeeze(dim=d)
    return result


def nansum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    if dtype is None:
        dtype = inp.dtype
        if dtype == torch.bool:
            inp = inp.to(torch.int64)
            dtype = torch.int64

    dims = _normalize_dims(dim, inp.ndim)

    # dim=[] -> reduce all
    if len(dims) == 0:
        result = _nansum_global(inp, dtype=dtype)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result

    # empty tensor
    if inp.numel() == 0:
        out_shape = list(inp.shape)
        if keepdim:
            for d in dims:
                out_shape[d] = 1
        else:
            for d in dims:
                out_shape.pop(d)
        return torch.zeros(out_shape, dtype=dtype, device=inp.device)

    # full-dimensional reduction -> delegate to global
    if len(dims) == inp.ndim:
        result = _nansum_global(inp, dtype=dtype)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result

    shape = list(inp.shape)

    # single-dim path
    if len(dims) == 1:
        dim = dims[0]
        if not inp.is_contiguous():
            inp = inp.contiguous()
        N = shape[dim]
        M = reduce(lambda x, y: x * y, shape[:dim], 1)
        K = inp.numel() // (M * N)

        out_shape = list(shape)
        out_shape[dim] = 1

        if N <= 1:
            NANZERO_BLOCK = 4096
            grid = (triton.cdiv(inp.numel(), NANZERO_BLOCK),)
            out = torch.empty_like(inp, dtype=dtype)
            with torch_device_fn.device(inp.device):
                _nan_to_zero_kernel[grid](
                    inp, out, inp.numel(), BLOCK_SIZE=NANZERO_BLOCK
                )
            out = out.reshape(out_shape)
            return out if keepdim else _squeeze_dims(out, dims)

        out = torch.empty(out_shape, dtype=dtype, device=inp.device)

        with torch_device_fn.device(inp.device):
            if K > 1:
                grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
                nansum_dim_kernel_non_inner[grid](out, inp, M, N, K)
            else:
                grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), 1, 1)
                nansum_dim_kernel_inner[grid](out, inp, M, N)

        return out if keepdim else _squeeze_dims(out, dims)

    # multi-dim path
    inp = dim_compress(inp, dims)
    N = 1
    for d in dims:
        N *= shape[d]
        shape[d] = 1
    M = inp.numel() // N

    if N <= 1:
        NANZERO_BLOCK = 4096
        grid = (triton.cdiv(inp.numel(), NANZERO_BLOCK),)
        out = torch.empty_like(inp, dtype=dtype)
        with torch_device_fn.device(inp.device):
            _nan_to_zero_kernel[grid](inp, out, inp.numel(), BLOCK_SIZE=NANZERO_BLOCK)
        out = out.reshape(shape)
        return out if keepdim else _squeeze_dims(out, dims)

    out = torch.empty(M, dtype=dtype, device=inp.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        nansum_dim_kernel[grid](inp, out, M, N)

    out = out.reshape(shape)
    return out if keepdim else _squeeze_dims(out, dims)
