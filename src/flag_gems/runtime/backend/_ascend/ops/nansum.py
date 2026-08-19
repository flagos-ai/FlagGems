import logging
import math
import threading

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_DEFAULT_NUM_VECTOR_CORES = 40
_num_vector_cores_cache = None

_MAX_UB_INNER = 8192
_MAX_UB_NON_INNER = 8192
_MAX_UB_GLOBAL = 10240
_MAX_UB_GLOBAL_FP16 = 12288


def _global_ub_cap(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return _MAX_UB_GLOBAL_FP16
    return _MAX_UB_GLOBAL


_SAFE_TILE_N_NON_INNER = 1024


def _get_num_vector_cores() -> int:
    global _num_vector_cores_cache
    if _num_vector_cores_cache is not None:
        return _num_vector_cores_cache
    try:
        from triton.backends.ascend import driver

        device_id = torch.npu.current_device()
        _num_vector_cores_cache = driver.active.utils.get_device_properties(
            device_id
        ).get("num_vectorcore", _DEFAULT_NUM_VECTOR_CORES)
    except Exception:
        _num_vector_cores_cache = _DEFAULT_NUM_VECTOR_CORES
    return _num_vector_cores_cache


_precompiled_inner: dict = {}
_precompiled_non_inner: dict = {}
_precompiled_multi_dim: dict = {}
_precompiled_lock = threading.Lock()


def _launch_inner_cached(
    out_flat, work, M, N, tile_m, tile_n, num_warps, num_stages, grid
):
    cache_key = (tile_m, tile_n, work.dtype)
    device = torch_device_fn.current_device()
    device_cache = _precompiled_inner.get(device)
    if device_cache is None:
        device_cache = {}
        _precompiled_inner[device] = device_cache

    compiled = device_cache.get(cache_key)
    if compiled is None:
        with _precompiled_lock:
            compiled = device_cache.get(cache_key)
            if compiled is None:
                compiled, _ = nansum_dim_kernel_inner.run(
                    out_flat,
                    work,
                    M,
                    N,
                    TILE_M=tile_m,
                    TILE_N=tile_n,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    grid=grid,
                    warmup=True,
                )
                device_cache[cache_key] = compiled

    g = grid + (1, 1) if len(grid) == 1 else grid + (1,) if len(grid) == 2 else grid
    compiled[g[:3]](out_flat, work, M, N)


def _launch_non_inner_cached(
    out_flat, work, M, N, K, tile_n, tile_k, one_tile, num_warps, num_stages, grid
):
    cache_key = (tile_n, tile_k, one_tile, work.dtype)
    device = torch_device_fn.current_device()
    device_cache = _precompiled_non_inner.get(device)
    if device_cache is None:
        device_cache = {}
        _precompiled_non_inner[device] = device_cache

    compiled = device_cache.get(cache_key)
    if compiled is None:
        with _precompiled_lock:
            compiled = device_cache.get(cache_key)
            if compiled is None:
                compiled, _ = nansum_dim_kernel_non_inner.run(
                    out_flat,
                    work,
                    M,
                    N,
                    K,
                    TILE_N=tile_n,
                    TILE_K=tile_k,
                    ONE_TILE_PER_CTA=one_tile,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    grid=grid,
                    warmup=True,
                )
                device_cache[cache_key] = compiled

    g = grid + (1, 1) if len(grid) == 1 else grid + (1,) if len(grid) == 2 else grid
    compiled[g[:3]](out_flat, work, M, N, K)


def _launch_multi_dim_cached(
    work, out_flat, M, N, block_m, block_n, num_warps, num_stages, grid
):
    cache_key = (block_m, block_n, work.dtype)
    device = torch_device_fn.current_device()
    device_cache = _precompiled_multi_dim.get(device)
    if device_cache is None:
        device_cache = {}
        _precompiled_multi_dim[device] = device_cache

    compiled = device_cache.get(cache_key)
    if compiled is None:
        with _precompiled_lock:
            compiled = device_cache.get(cache_key)
            if compiled is None:
                compiled, _ = nansum_dim_kernel.run(
                    work,
                    out_flat,
                    M,
                    N,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    grid=grid,
                    warmup=True,
                )
                device_cache[cache_key] = compiled

    g = grid + (1, 1) if len(grid) == 1 else grid + (1,) if len(grid) == 2 else grid
    compiled[g[:3]](work, out_flat, M, N)


def _nansum_heur_tile_n_inner(args):
    N = args["N"]
    M = args.get("M", 1)
    ub_tile_n = min(triton.next_power_of_2(N), _MAX_UB_INNER)

    # Small-N fast path: one tile covers the whole row, skip dynamic balancing
    if N <= 256:
        return ub_tile_n

    # Base min_tile_m from N (SIMD ops within a single row)
    if N >= 8192:
        min_tile_m = 1
    elif N >= 4096:
        min_tile_m = 2
    elif N >= 2048:
        min_tile_m = 4
    elif N >= 1024:
        min_tile_m = 8
    else:
        min_tile_m = 32

    if M > 262144:
        min_tile_m = max(min_tile_m, 32)
    elif M > 65536:
        min_tile_m = max(min_tile_m, 16)
    elif M > 16384:
        min_tile_m = max(min_tile_m, 8)

    return min(ub_tile_n, _MAX_UB_INNER // min_tile_m)


def _nansum_heur_tile_m_inner(args):
    M, N = args["M"], args["N"]
    num_cores = _get_num_vector_cores()
    tile_n = _nansum_heur_tile_n_inner(args)
    iters_per_row = max(1, (N + tile_n - 1) // tile_n)

    # Small-N fast path: one iteration per row, pack rows to UB limit
    if N <= 256:
        max_tile_m = max(_MAX_UB_INNER // tile_n, 1)
        tile_m = max(1, min(max_tile_m, M // max(1, num_cores)))
        return min(tile_m, max_tile_m, 512)

    if iters_per_row >= 4:
        target_waves = 4
    elif iters_per_row >= 2:
        target_waves = 2
    else:
        target_waves = 1

    target_grid = max(1, num_cores * target_waves)
    tile_m = triton.next_power_of_2(max(1, M // target_grid))

    # UB constraint: inner kernel ~5x buffer overhead
    max_tile_m = max(_MAX_UB_INNER // tile_n, 1)

    # Min work per block to avoid excessive grid for small N (e.g. [1024,16])
    MIN_WORK_PER_BLOCK = 2048
    min_tile_m = max(1, MIN_WORK_PER_BLOCK // tile_n)
    tile_m = max(tile_m, min(min_tile_m, max_tile_m))

    return min(tile_m, max_tile_m, 512)


def _nansum_heur_tile_k(args):
    M, K = args["M"], args["K"]
    num_cores = _get_num_vector_cores()
    MAX_TILE_K = min(_MAX_UB_NON_INNER, K)
    MIN_TILE_N = 8

    if M <= 1:
        if K > 1024:
            MIN_TILE_K_FOR_BW = 16
        else:
            MIN_TILE_K_FOR_BW = 64
        MIN_TILE_K = max(8, MIN_TILE_K_FOR_BW)
        effective_max_tk = min(MAX_TILE_K, _MAX_UB_NON_INNER // MIN_TILE_N)
        if K > 131072:
            TARGET_WAVES = 16
        elif K > 32768:
            TARGET_WAVES = 8
        elif K > 8192:
            TARGET_WAVES = 4
        elif K > 1024:
            TARGET_WAVES = 2
        else:
            TARGET_WAVES = 1
    elif M <= 4:
        MIN_TILE_K = 8
        TARGET_WAVES = 4
        effective_max_tk = MAX_TILE_K
    elif M <= 64:
        MIN_TILE_K = 1
        TARGET_WAVES = 2
        effective_max_tk = MAX_TILE_K
    else:
        min_total_blocks = num_cores * 16
        effective_upper = max(1, (M * K) // min_total_blocks)
        upper = min(K, MAX_TILE_K, effective_upper)
        tile_k = max(1, triton.cdiv(_MAX_UB_NON_INNER, _SAFE_TILE_N_NON_INNER))
        while tile_k <= upper:
            if tile_k * 2 <= upper:
                tile_k *= 2
            else:
                break
        return tile_k

    tile_k = MIN_TILE_K
    upper = min(K, effective_max_tk)
    best = tile_k
    while tile_k <= upper:
        num_blocks = M * triton.cdiv(K, tile_k)
        if num_blocks / num_cores >= TARGET_WAVES:
            best = tile_k
            if tile_k * 2 <= upper:
                tile_k *= 2
            else:
                break
        else:
            break
    return best


def _nansum_heur_tile_n_non_inner(args):
    M = args.get("M", 1)
    if M <= 1:
        tile_budget = min(_MAX_UB_NON_INNER * 3 // 2, 11520)
        tile_n = triton.cdiv(tile_budget, args["TILE_K"])
        tile_n = min(tile_n, args["N"])
        tile_n = triton.next_power_of_2(tile_n + 1) // 2
    else:
        tile_n = triton.cdiv(_MAX_UB_NON_INNER, args["TILE_K"])
    return min(tile_n, _SAFE_TILE_N_NON_INNER)


def _nansum_heur_one_tile_per_cta(args):
    return args["TILE_N"] >= args["N"]


def _nansum_heur_num_warps_inner(args):
    tile_size = args.get("TILE_M", 1) * args.get("TILE_N", 1)
    # Small-N fast path: tile covers one or few rows → minimal warps needed
    if args.get("N", 0) <= 256:
        return 4
    if tile_size <= 1024:
        return 4
    elif tile_size <= 4096:
        return 8
    else:
        return 8


def _nansum_heur_num_warps_non_inner(args):
    tile_size = args.get("TILE_N", 1) * args.get("TILE_K", 1)
    if args.get("ONE_TILE_PER_CTA") and args.get("M", 1) <= 1:
        return 4
    if tile_size <= 1024:
        return 4
    elif tile_size <= 4096:
        return 8
    else:
        return 8


def _nansum_heur_multi_dim_config(args):
    M, N = args["M"], args["N"]
    if M == 1:
        if N <= 1024:
            return 1, triton.next_power_of_2(N), 4
        elif N <= 4096:
            return 1, 2048, 4
        elif N <= 16384:
            return 1, 4096, 4
        else:
            return 1, 4096, 8
    if M <= 4:
        if N <= 64:
            return 8, 32, 2
        elif N <= 256:
            return 4, 128, 4
        elif N <= 1024:
            return 1, 512, 4
        elif N <= 4096:
            return 1, 1024, 4
        else:
            return 1, 2048, 8
    if M <= 16:
        if N <= 64:
            return 8, 64, 4
        elif N <= 256:
            return 8, 256, 4
        elif N <= 1024:
            return 4, 512, 8
        elif N <= 4096:
            return 4, 1024, 8
        else:
            return 2, 2048, 8
    if M <= 64:
        if N <= 64:
            return 8, 64, 4
        elif N <= 256:
            return 8, 256, 8
        elif N <= 1024:
            return 4, 512, 8
        else:
            return 4, 1024, 8
    if N <= 64:
        return 8, 64, 4
    elif N <= 256:
        return 8, 256, 8
    elif N <= 1024:
        return 4, 512, 8
    elif N <= 4096:
        return 4, 1024, 8
    else:
        return 2, 2048, 8


def _normalize_dims(dim, ndim):
    if isinstance(dim, int):
        d = dim if dim >= 0 else dim + ndim
        return [d]
    if isinstance(dim, (list, tuple)):
        if len(dim) == 0:
            return []
        if len(dim) == 1:
            d = dim[0]
            return [d if d >= 0 else d + ndim]
        dims = [d if d >= 0 else d + ndim for d in dim]
        return sorted(set(dims), reverse=True)
    # dim is None → reduce all dims
    return list(range(ndim))


def _squeeze_dims(result, dims):
    # dims is already sorted in descending order from _normalize_dims
    for d in dims:
        result = result.squeeze(dim=d)
    return result


@libentry()
@triton.jit
def _nan_to_zero_kernel(
    inp,
    out,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    num_blocks = ext.num_programs(0)
    for start in range(pid * BLOCK_SIZE, N, num_blocks * BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        val = tl.load(inp + offsets, mask=mask, other=0.0)
        val = tl.where(val != val, 0.0, val)
        tl.store(out + offsets, val, mask=mask)


@libentry()
@triton.jit
def nansum_global_reduce(
    inp,
    out,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    num_blocks = ext.num_programs(0)
    _sum = tl.zeros((), dtype=tl.float32)

    for start in range(pid * BLOCK_SIZE, M, num_blocks * BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(inp + offsets, mask=mask, other=0.0)
        x = tl.where(x != x, 0.0, x)
        _sum += tl.sum(x, axis=0)
    tl.atomic_add(out, _sum)


@libentry()
@triton.jit
def nansum_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    num_blocks = ext.num_programs(0)
    _sum = tl.zeros((), dtype=tl.float32)

    for start in range(pid * BLOCK_SIZE, M, num_blocks * BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(inp + offsets, mask=mask, other=0.0)
        x = tl.where(x != x, 0.0, x)
        _sum += tl.sum(x, axis=0)
    tl.store(mid + pid, _sum)


@libentry()
@triton.jit
def nansum_kernel_2(
    mid,
    out,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    _sum = tl.zeros((), dtype=tl.float32)

    for start in range(0, N, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        val = tl.load(mid + offsets, mask=mask, other=0.0)
        _sum += tl.sum(val, axis=0)
    tl.store(out, _sum)


@libentry()
@triton.jit
def nansum_dim_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    pid = ext.program_id(0)
    stride = ext.num_programs(0)

    for m_start in range(pid * TILE_M, M, stride * TILE_M):
        m_offsets = m_start + tl.arange(0, TILE_M)[:, None]
        m_mask = m_offsets < M

        _sum = tl.zeros([TILE_M, TILE_N], dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[None, :]
            inp_offsets = m_offsets * N + n_offsets
            mask = m_mask & (n_offsets < N)
            val = tl.load(input_ptr + inp_offsets, mask=mask, other=0.0)
            if tl.constexpr(input_ptr.dtype.element_ty != tl.float32):
                val = val.to(tl.float32)
            val = tl.where(val != val, 0.0, val)
            _sum += val

        result = tl.sum(_sum, axis=1)[:, None]
        tl.store(output_ptr + m_offsets, result, mask=m_mask)


@libentry()
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
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        inp_offset = pid_m * N * K + n_offsets * K + k_offsets
        mask = (n_offsets < N) & (k_offsets < K)
        val = tl.load(input_ptr + inp_offset, mask=mask, other=0.0)
        if tl.constexpr(input_ptr.dtype.element_ty != tl.float32):
            val = val.to(tl.float32)
        val = tl.where(val != val, 0.0, val)
        out = tl.sum(val, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        tl.store(output_ptr + out_offset, out, mask=k_offsets < K)
    else:
        _sum = tl.zeros([TILE_N, TILE_K], dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            inp_offsets = pid_m * N * K + n_offsets * K + k_offsets
            mask = (n_offsets < N) & (k_offsets < K)
            val = tl.load(input_ptr + inp_offsets, mask=mask, other=0.0)
            if tl.constexpr(input_ptr.dtype.element_ty != tl.float32):
                val = val.to(tl.float32)
            val = tl.where(val != val, 0.0, val)
            _sum += val
        out = tl.sum(_sum, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        tl.store(output_ptr + out_offset, out, mask=k_offsets < K)


@libentry()
@triton.jit
def nansum_dim_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = ext.program_id(0) * BLOCK_M
    stride = ext.num_programs(0) * BLOCK_M

    for m_start in range(pid, M, stride):
        m_offsets = m_start + tl.arange(0, BLOCK_M)[:, None]
        m_mask = m_offsets < M

        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for start_n in range(0, N, BLOCK_N):
            n_offsets = start_n + tl.arange(0, BLOCK_N)[None, :]
            val = tl.load(
                inp + m_offsets * N + n_offsets,
                mask=m_mask & (n_offsets < N),
                other=0.0,
            )
            val = tl.where(val != val, 0.0, val)
            _sum += val

        result = tl.sum(_sum, axis=1)[:, None]
        tl.store(out + m_offsets, result, mask=m_mask)


def nansum(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_ASCEND NANSUM")

    if dim is not None:
        return nansum_dim(inp, dim=dim, keepdim=keepdim, dtype=dtype)

    if dtype is None:
        dtype = inp.dtype

    if inp.numel() == 0:
        return torch.tensor(0, dtype=dtype, device=inp.device)

    if not inp.is_contiguous():
        work = inp.contiguous()
    else:
        work = inp
    if inp.dtype == torch.bool:
        work = work.to(torch.int64)
    M = work.numel()

    num_cores = _get_num_vector_cores()

    _TWO_STAGE_THRESHOLD = 100 * 1024 * 1024
    use_two_stage = M > _TWO_STAGE_THRESHOLD

    if M <= 4096:
        grid_size = 4
        block_size = triton.next_power_of_2(M)
        num_warps = 4
    elif M <= 32768:
        grid_size = 4
        block_size = max(4096, triton.next_power_of_2(max(1, math.ceil(M / grid_size))))
        block_size = min(block_size, _global_ub_cap(work.dtype))
        num_warps = 4 if block_size <= 4096 else 8
    elif M <= 131072:
        grid_size = 16
        block_size = max(4096, triton.next_power_of_2(max(1, math.ceil(M / grid_size))))
        block_size = min(block_size, _global_ub_cap(work.dtype))
        num_warps = 8
    else:
        grid_size = num_cores * 2 if use_two_stage else num_cores
        block_size = max(4096, triton.next_power_of_2(max(1, math.ceil(M / grid_size))))
        block_size = min(block_size, _global_ub_cap(work.dtype))
        num_warps = 8

    with torch_device_fn.device(work.device):
        if use_two_stage:
            mid = torch.empty(grid_size, dtype=torch.float32, device=work.device)
            nansum_kernel_1[(grid_size,)](
                work,
                mid,
                M,
                block_size,
                num_warps=num_warps,
                num_stages=2,
            )
            final_block = triton.next_power_of_2(min(grid_size, _MAX_UB_GLOBAL))
            out = torch.zeros([], dtype=torch.float32, device=work.device)
            nansum_kernel_2[(1,)](
                mid,
                out,
                grid_size,
                final_block,
                num_warps=4,
                num_stages=2,
            )
        else:
            out = torch.zeros([], dtype=torch.float32, device=work.device)
            nansum_global_reduce[(grid_size,)](
                work,
                out,
                M,
                block_size,
                num_warps=num_warps,
                num_stages=2,
            )

    return out.to(dtype)


def nansum_out(inp, dim=None, keepdim=False, *, dtype=None, out=None):
    logger.debug("GEMS_ASCEND NANSUM_OUT")
    result = nansum(inp, dim=dim, keepdim=keepdim, dtype=dtype)
    out.copy_(result)
    return out


def nansum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_ASCEND NANSUM_DIM")

    if dtype is None:
        dtype = inp.dtype

    if inp.numel() == 0:
        out_shape = list(inp.shape)
        if dim is None:
            out_shape = [1] * inp.ndim if keepdim else []
        else:
            dims = _normalize_dims(dim, inp.ndim)
            if keepdim:
                for d in dims:
                    out_shape[d] = 1
            else:
                for d in dims:  # already sorted descending
                    out_shape.pop(d)
        return torch.zeros(out_shape, dtype=dtype, device=inp.device)

    dims = _normalize_dims(dim, inp.ndim)

    if len(dims) == inp.ndim:
        result = nansum(inp, dtype=dtype)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result

    if not inp.is_contiguous():
        work = inp.contiguous()
    else:
        work = inp
    if work.dtype == torch.bool:
        work = work.to(torch.int64)

    shape = work.shape
    ndim = len(shape)

    if len(dims) == 1:
        dim = dims[0]
        N = shape[dim]

        M = 1
        for i in range(dim):
            M *= shape[i]
        K = 1
        for i in range(dim + 1, ndim):
            K *= shape[i]

        out_shape = list(shape)
        out_shape[dim] = 1
        num_cores = _get_num_vector_cores()

        if N <= 1:
            out = torch.empty_like(work, dtype=torch.float32)
            numel = work.numel()
            nz_block_size = triton.next_power_of_2(min(numel, 8192))
            grid = (min(triton.cdiv(numel, nz_block_size), num_cores),)
            with torch_device_fn.device(work.device):
                _nan_to_zero_kernel[grid](
                    work,
                    out,
                    numel,
                    BLOCK_SIZE=nz_block_size,
                    num_warps=4,
                    num_stages=2,
                )
            out = out.reshape(out_shape)
            if not keepdim:
                out = out.squeeze(dim=dim)
            return out.to(dtype)

        out_flat = torch.empty(out_shape, dtype=torch.float32, device=work.device)

        if K > 1:
            tile_k = _nansum_heur_tile_k({"M": M, "K": K, "N": N})
            tile_n = _nansum_heur_tile_n_non_inner({"M": M, "N": N, "TILE_K": tile_k})
            one_tile = _nansum_heur_one_tile_per_cta({"TILE_N": tile_n, "N": N})
            nw = _nansum_heur_num_warps_non_inner(
                {
                    "TILE_N": tile_n,
                    "TILE_K": tile_k,
                    "ONE_TILE_PER_CTA": one_tile,
                    "M": M,
                }
            )
            _launch_non_inner_cached(
                out_flat,
                work,
                M,
                N,
                K,
                tile_n,
                tile_k,
                one_tile,
                nw,
                2,
                (M, triton.cdiv(K, tile_k)),
            )
        else:
            tile_n = _nansum_heur_tile_n_inner({"N": N, "M": M})
            tile_m = _nansum_heur_tile_m_inner({"M": M, "N": N, "TILE_N": tile_n})
            nw = _nansum_heur_num_warps_inner(
                {"TILE_M": tile_m, "TILE_N": tile_n, "N": N}
            )
            MIN_DATA_PER_BLOCK = 256 * 1024
            total_elems = M * N
            max_grid = max(
                num_cores * 8,
                min(num_cores * 256, max(1, total_elems // MIN_DATA_PER_BLOCK)),
            )
            grid_size = min(triton.cdiv(M, tile_m), max_grid)
            _launch_inner_cached(
                out_flat,
                work,
                M,
                N,
                tile_m,
                tile_n,
                nw,
                2,
                (grid_size,),
            )

        result = out_flat.to(dtype)
        return result if keepdim else _squeeze_dims(result, dims)

    # Multi-dim reduction: dim_compress + 2D tile kernel
    work = dim_compress(work, dims)
    N = 1
    shape_list = list(shape)  # shape may be tuple (torch.Size), need mutable list
    for d in dims:
        N *= shape_list[d]
        shape_list[d] = 1
    M = work.numel() // N
    num_cores = _get_num_vector_cores()

    if N <= 1:
        out = torch.empty_like(work, dtype=torch.float32)
        numel = work.numel()
        nz_block_size = triton.next_power_of_2(min(numel, 8192))
        grid = (min(triton.cdiv(numel, nz_block_size), num_cores),)
        with torch_device_fn.device(work.device):
            _nan_to_zero_kernel[grid](
                work,
                out,
                numel,
                BLOCK_SIZE=nz_block_size,
                num_warps=4,
                num_stages=2,
            )
        out = out.reshape(shape_list)
        result = out.to(dtype)
        return result if keepdim else _squeeze_dims(result, dims)

    out_flat = torch.empty(M, dtype=torch.float32, device=work.device)
    block_m, block_n, nw = _nansum_heur_multi_dim_config({"M": M, "N": N})
    max_tasks = triton.cdiv(M, block_m)
    grid_size = max_tasks if max_tasks <= num_cores else min(max_tasks, num_cores * 6)

    _launch_multi_dim_cached(
        work,
        out_flat,
        M,
        N,
        block_m,
        block_n,
        nw,
        2,
        (grid_size,),
    )

    out_flat = out_flat.reshape(shape_list)
    result = out_flat.to(dtype)
    return result if keepdim else _squeeze_dims(result, dims)
