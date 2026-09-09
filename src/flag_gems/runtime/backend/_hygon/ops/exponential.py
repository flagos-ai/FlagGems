import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._hygon import heuristics_config_utils as _hcu
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


@triton.jit
def safe_fast_log(x):
    min_normal = x * 0.0 + 1.17549435e-38
    max_u = x * 0.0 + 0.99999994
    x = tl.minimum(tl.maximum(x, min_normal), max_u)
    return tl.math.log(x)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK": 1024}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK": 2048}, num_warps=16, num_stages=4),
    ],
    key=["N", "is_double", "dtype_code"],
)
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N", "dtype_code"])
def fused_exponential_kernel(
    out_ptr,
    N,
    is_double,
    inv_lambd,
    eps_minus,
    philox_seed,
    philox_offset,
    dtype_code,
    BLOCK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    if is_double:
        d0 = uint_to_uniform_float(paste_u64(r0, r2))
        d1 = uint_to_uniform_float(paste_u64(r1, r3))
        y0 = transform_exponential(d0, inv_lambd, eps_minus)
        y1 = transform_exponential(d1, inv_lambd, eps_minus)
        UNROLL = 2
        start = tl.program_id(0).to(tl.uint64) * BLOCK * UNROLL
        off_0 = start + tl.arange(0, BLOCK)
        off_1 = off_0 + BLOCK
        tl.store(out_ptr + off_0, y0, mask=off_0 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_1, y1, mask=off_1 < N, eviction_policy="evict_first")
    else:
        f0 = uint_to_uniform_float(r0)
        f1 = uint_to_uniform_float(r1)
        f2 = uint_to_uniform_float(r2)
        f3 = uint_to_uniform_float(r3)
        y0 = transform_exponential(f0, inv_lambd, eps_minus)
        y1 = transform_exponential(f1, inv_lambd, eps_minus)
        y2 = transform_exponential(f2, inv_lambd, eps_minus)
        y3 = transform_exponential(f3, inv_lambd, eps_minus)
        UNROLL = 4
        start = tl.program_id(0).to(tl.uint64) * BLOCK * UNROLL
        off_0 = start + tl.arange(0, BLOCK)
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK
        tl.store(out_ptr + off_0, y0, mask=off_0 < N, eviction_policy="evict_last")
        tl.store(out_ptr + off_1, y1, mask=off_1 < N, eviction_policy="evict_last")
        tl.store(out_ptr + off_2, y2, mask=off_2 < N, eviction_policy="evict_last")
        tl.store(out_ptr + off_3, y3, mask=off_3 < N, eviction_policy="evict_last")


@triton.jit
def paste_u64(hi: tl.uint32, lo: tl.uint32):
    hi = hi.to(tl.uint64) << 32
    x = hi | lo.to(tl.uint64)
    return x


@triton.jit
def transform_exponential(u, inv_lambd, eps_minus):
    is_min = u >= 1.0 + eps_minus
    log = tl.where(is_min, eps_minus, safe_fast_log(u))
    v = -inv_lambd * log
    return v


def _philox_seed_offset(increment, generator=None):
    gen = generator
    if gen is None:
        try:
            gen = torch_device_fn.default_generators[torch_device_fn.current_device()]
        except Exception:
            gen = None
    if gen is not None:
        try:
            seed = int(gen.initial_seed())
            offset = int(gen.get_offset())
            gen.set_offset(offset + (increment + 3) // 4 * 4)
            return seed, offset
        except (AttributeError, RuntimeError, TypeError):
            pass
    return philox_backend_seed_offset(increment, generator=generator)


_compiled_cache = {}
_COMPILED_CACHE_LIMIT = 256
_FAST_LAUNCH_MAX_N = 1 << 25


def _cache_key(x_, N):
    return (x_.device.index, x_.dtype, N)


def _fast_launch(
    x_, N, is_double, inv_lambd, eps_minus, philox_seed, philox_offset, dtype_code
):
    if is_double or N > _FAST_LAUNCH_MAX_N or x_.data_ptr() % 16 != 0:
        return False
    key = _cache_key(x_, N)
    entry = _compiled_cache.get(key)
    if entry is None:
        return False
    compiled, grid0 = entry
    try:
        compiled[(grid0, 1, 1)](
            x_,
            N,
            is_double,
            inv_lambd,
            eps_minus,
            philox_seed,
            philox_offset,
            dtype_code,
        )
    except Exception:
        _compiled_cache[key] = None
        return False
    return True


def _harvest_compiled(x_, N, is_double, UNROLL, dtype_code, launch_args):
    if is_double or N > _FAST_LAUNCH_MAX_N or x_.data_ptr() % 16 != 0:
        return
    key = _cache_key(x_, N)
    if key in _compiled_cache:
        return
    try:
        jit_fn = fused_exponential_kernel.fn
        if not hasattr(jit_fn, "warmup"):
            raise RuntimeError("no warmup")
        cfg = getattr(fused_exponential_kernel, "best_config", None)
        if cfg is not None:
            BLOCK = cfg.kwargs["BLOCK"]
            num_warps = cfg.num_warps
            num_stages = cfg.num_stages
        else:
            BLOCK = _hcu.exponential_heur_block({"N": N})
            num_warps = _hcu.exponential_heur_num_warps({"N": N})
            num_stages = None
        grid0 = triton.cdiv(N, BLOCK * UNROLL)
        launch_kw = {"BLOCK": BLOCK, "num_warps": num_warps, "grid": (grid0,)}
        if num_stages is not None:
            launch_kw["num_stages"] = num_stages
        compiled = jit_fn.warmup(*launch_args, **launch_kw)
        compiled[(grid0, 1, 1)](*launch_args)
        entry = (compiled, grid0)
    except Exception:
        entry = None
    if len(_compiled_cache) >= _COMPILED_CACHE_LIMIT:
        _compiled_cache.clear()
    _compiled_cache[key] = entry


def exponential(x, lambd: float = 1.0, *, generator=None):
    logger.debug("GEMS_HYGON EXPONENTIAL")
    dtype = x.dtype
    device = x.device
    assert dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    is_double = dtype in (torch.float64,)
    UNROLL = 2 if is_double else 4
    N = x.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
    increment = triton.cdiv(N, UNROLL) + 2048
    philox_seed, philox_offset = _philox_seed_offset(increment, generator=generator)
    eps = torch.finfo(dtype).eps
    eps_minus = -0.5 * eps
    inv_lambd = 1.0 / lambd
    dtype_code = _DTYPES.index(dtype)
    res = torch.empty(x.shape, dtype=dtype, device=device)
    with torch_device_fn.device(device):
        if not _fast_launch(
            res,
            N,
            is_double,
            inv_lambd,
            eps_minus,
            philox_seed,
            philox_offset,
            dtype_code,
        ):
            launch_args = (
                res,
                N,
                is_double,
                inv_lambd,
                eps_minus,
                philox_seed,
                philox_offset,
                dtype_code,
            )
            fused_exponential_kernel[grid_fn](*launch_args)
            _harvest_compiled(res, N, is_double, UNROLL, dtype_code, launch_args)
    return res
