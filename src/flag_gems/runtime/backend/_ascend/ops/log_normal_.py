import logging

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend import heuristics_config_utils as _hcu
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


@triton.jit
def _pair_uniform_to_normal(u1, u2):
    u1 = tl.maximum(1.0e-7, u1)
    theta = 6.283185307179586 * u2
    r = tl.sqrt(-2.0 * tl.log(u1))
    sin_t = tl.sin(theta)
    cos_t = tl.cos(theta)
    return r * cos_t, r * sin_t


@triton.heuristics(_hcu.HEURISTICS_CONFIGS["log_normal_"])
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N", "mean", "std"])
def fused_log_normal_kernel(
    out_ptr,
    N,
    mean,
    std,
    philox_seed,
    philox_offset,
    UNROLL,
    BLOCK: tl.constexpr,
):
    n_workers = tl.num_programs(0)
    pid = tl.program_id(0)
    n_tasks = tl.cdiv(N, BLOCK * UNROLL)
    tasks_per_worker = tl.cdiv(n_tasks, n_workers)

    philox_seed_64 = philox_seed.to(tl.int64)
    philox_offset_64 = philox_offset.to(tl.int64)
    c0 = (philox_offset_64 & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset_64 >> 32) & 0xFFFFFFFF).to(tl.uint32)

    for task_index in range(tasks_per_worker):
        task_id = pid + task_index * n_workers
        i4 = task_id * BLOCK + tl.arange(0, BLOCK)
        c0_ = c0 + i4
        _O = c0_ * 0
        r0, r1, r2, r3 = tl.philox(philox_seed_64, c0_, c1, _O, _O)

        f0 = uint_to_uniform_float(r0)
        f1 = uint_to_uniform_float(r1)
        f2 = uint_to_uniform_float(r2)
        f3 = uint_to_uniform_float(r3)

        n0, n1 = _pair_uniform_to_normal(f0, f1)
        n2, n3 = _pair_uniform_to_normal(f2, f3)

        y0 = tl.exp(n0 * std + mean)
        y1 = tl.exp(n1 * std + mean)
        y2 = tl.exp(n2 * std + mean)
        y3 = tl.exp(n3 * std + mean)

        start = task_id.to(tl.int64) * BLOCK * 4
        off_0 = start + tl.arange(0, BLOCK)
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK

        tl.store(out_ptr + off_0, y0, mask=off_0 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_1, y1, mask=off_1 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_2, y2, mask=off_2 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_3, y3, mask=off_3 < N, eviction_policy="evict_first")


def log_normal_distribution(shape, device, mean, std, out, *, generator=None):
    N = volume(shape)

    UNROLL = 4
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )

    def grid_fn(meta):
        grid = triton.cdiv(N, meta["BLOCK"] * UNROLL)
        grid = grid if grid < 240 else 240
        return (grid,)

    with torch_device_fn.device(device):
        fused_log_normal_kernel[grid_fn](
            out, N, mean, std, philox_seed, philox_offset, UNROLL
        )
    return out


def log_normal_(self, mean=1.0, std=2.0, *, generator=None):
    logger.debug("GEMS_ASCEND LOG_NORMAL_")
    shape = self.shape
    device = self.device

    log_normal_distribution(shape, device, mean, std, self, generator=generator)
    return self
