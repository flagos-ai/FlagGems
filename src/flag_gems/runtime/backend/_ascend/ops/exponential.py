import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend import heuristics_config_utils as _hcu
from flag_gems.utils.random_utils import philox_backend_seed_offset

logger = logging.getLogger(__name__)

_PHILOX_SA = tl.constexpr(0xD2511F53)
_PHILOX_SB = tl.constexpr(0xCD9E8D57)
_PHILOX_KEY_A = tl.constexpr(0x9E3779B9)
_PHILOX_KEY_B = tl.constexpr(0xBB67AE85)


@triton.jit
def _mulhi_u32_limb(a, b):
    a0 = a & 0xFFFF
    a1 = (a >> 16) & 0xFFFF
    b0 = b & 0xFFFF
    b1 = (b >> 16) & 0xFFFF
    w0 = a0 * b0
    t = a1 * b0 + ((w0 >> 16) & 0xFFFF)
    w1 = a0 * b1 + (t & 0xFFFF)
    return a1 * b1 + ((t >> 16) & 0xFFFF) + ((w1 >> 16) & 0xFFFF)


@triton.jit
def _philox4x32_10(seed, c0, c1, c2, c3):
    seed64 = seed.to(tl.uint64)
    k0 = (seed64 & 0xFFFFFFFF).to(tl.uint32)
    k1 = ((seed64 >> 32) & 0xFFFFFFFF).to(tl.uint32)
    for _ in tl.static_range(10):
        hi0 = _mulhi_u32_limb(c0, _PHILOX_SA)
        lo0 = c0 * _PHILOX_SA
        hi1 = _mulhi_u32_limb(c2, _PHILOX_SB)
        lo1 = c2 * _PHILOX_SB
        n0 = hi1 ^ c1 ^ k0
        n1 = lo1
        n2 = hi0 ^ c3 ^ k1
        n3 = lo0
        c0, c1, c2, c3 = n0, n1, n2, n3
        k0 = k0 + _PHILOX_KEY_A
        k1 = k1 + _PHILOX_KEY_B
    return c0, c1, c2, c3


@triton.jit
def _uint32_to_uniform_float(r):
    x = r.to(tl.int32, bitcast=True)
    xa = x ^ (x >> 31)
    return xa.to(tl.float32) * 4.6566127342e-10


@triton.heuristics(_hcu.HEURISTICS_CONFIGS["exponential_"])
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def fused_exponential_kernel(
    out_ptr,
    N,
    lambd,
    eps,
    philox_seed,
    philox_offset,
    UNROLL,
    BLOCK: tl.constexpr,
):
    n_workers = tl.num_programs(0)
    pid = tl.program_id(0)
    n_tasks = tl.cdiv(N, BLOCK * UNROLL)
    tasks_per_worker = tl.cdiv(n_tasks, n_workers)

    for task_index in range(tasks_per_worker):
        task_id = pid + task_index * n_workers
        philox_seed_64 = philox_seed.to(tl.int64)
        philox_offset_64 = philox_offset.to(tl.int64)
        c0 = (philox_offset_64 & 0xFFFFFFFF).to(tl.uint32)
        c1 = ((philox_offset_64 >> 32) & 0xFFFFFFFF).to(tl.uint32)
        i4 = task_id * BLOCK + tl.arange(0, BLOCK)
        c0 += i4
        _O = c0 * 0
        r0, r1, r2, r3 = _philox4x32_10(philox_seed_64, c0, c1, _O, _O)
        f0 = _uint32_to_uniform_float(r0)
        f1 = _uint32_to_uniform_float(r1)
        f2 = _uint32_to_uniform_float(r2)
        f3 = _uint32_to_uniform_float(r3)
        y0 = transform_exponential(f0, lambd, eps)
        y1 = transform_exponential(f1, lambd, eps)
        y2 = transform_exponential(f2, lambd, eps)
        y3 = transform_exponential(f3, lambd, eps)
        start = task_id.to(tl.int64) * BLOCK * 4
        off_0 = start + tl.arange(0, BLOCK)
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK
        tl.store(out_ptr + off_0, y0, mask=off_0 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_1, y1, mask=off_1 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_2, y2, mask=off_2 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_3, y3, mask=off_3 < N, eviction_policy="evict_first")


@triton.jit
def transform_exponential(u, lambd, eps):
    eps1 = -0.5 * eps
    is_min = u >= 1.0 + eps1
    tiny = tl.full((), 1.17549435e-38, tl.float32).to(u.dtype)
    u = tl.maximum(u, tiny)
    log = tl.where(is_min, eps1, tl.math.log(u))
    v = -1.0 / lambd * log
    return v


def exponential(x, lambd: float = 1.0, *, generator=None):
    logger.debug("GEMS_ASCEND EXPONENTIAL")
    dtype = x.dtype
    device = x.device
    assert dtype in (torch.float16, torch.bfloat16, torch.float32)
    UNROLL = 4
    N = x.numel()

    def grid_fn(meta):
        grid = triton.cdiv(N, meta["BLOCK"] * UNROLL)
        grid = grid if grid < 240 else 240
        return (grid,)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    eps = torch.finfo(dtype).eps
    res = torch.empty(x.shape, dtype=dtype, device=device)
    with torch_device_fn.device(device):
        if not _fast_launch(res, N, lambd, eps, philox_seed, philox_offset, UNROLL):
            fused_exponential_kernel[grid_fn](
                res, N, lambd, eps, philox_seed, philox_offset, UNROLL
            )
    return res


_compiled_cache = {}


def _fast_launch(res, N, lambd, eps, philox_seed, philox_offset, UNROLL):
    if res.data_ptr() % 16 != 0:
        return False
    jit_fn = fused_exponential_kernel.fn
    if not hasattr(jit_fn, "warmup"):
        return False
    BLOCK = _hcu.exponential_heur_block({"N": N})
    num_warps = _hcu.exponential_heur_num_warps({"N": N})
    grid0 = min(triton.cdiv(N, BLOCK * UNROLL), 240)
    key = (res.device.index, res.dtype, BLOCK, num_warps)
    try:
        compiled = _compiled_cache.get(key)
        if compiled is None:
            compiled = jit_fn.warmup(
                res,
                N,
                lambd,
                eps,
                philox_seed,
                philox_offset,
                UNROLL,
                BLOCK=BLOCK,
                num_warps=num_warps,
                grid=(grid0,),
            )
            _compiled_cache[key] = compiled
        compiled[(grid0, 1, 1)](res, N, lambd, eps, philox_seed, philox_offset, UNROLL)
    except Exception:
        _compiled_cache.pop(key, None)
        return False
    return True
