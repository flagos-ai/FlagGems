import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
device_ = device


@triton.heuristics(runtime.get_heuristic_config("rand"))
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def rand_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
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
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    tl.store(out_ptr + off_0, r0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, r1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, r2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, r3, mask=off_3 < N, eviction_policy="evict_first")


def choose_unroll(N, core=64, clusters=12):
    for u in (16, 1):
        if triton.cdiv(N, clusters * u) >= core:
            return u
    return 1


# @triton.heuristics(runtime.get_heuristic_config("rand"))
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def rand_kernel_1(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    UNROLL: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    off_0 = tl.program_id(0) * BLOCK * UNROLL + tl.arange(0, BLOCK)
    tl.store(out_ptr + off_0, r0, mask=off_0 < N, eviction_policy="evict_first")


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def rand_kernel_2(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    UNROLL: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r4, r5, r6, r7 = tl.philox(philox_seed, c0 + 1, c1, _O, _O)
    r8, r9, r10, r11 = tl.philox(philox_seed, c0 + 2, c1, _O, _O)
    r12, r13, r14, r15 = tl.philox(philox_seed, c0 + 3, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)
    r4 = uint_to_uniform_float(r4)
    r5 = uint_to_uniform_float(r5)
    r6 = uint_to_uniform_float(r6)
    r7 = uint_to_uniform_float(r7)
    r8 = uint_to_uniform_float(r8)
    r9 = uint_to_uniform_float(r9)
    r10 = uint_to_uniform_float(r10)
    r11 = uint_to_uniform_float(r11)
    r12 = uint_to_uniform_float(r12)
    r13 = uint_to_uniform_float(r13)
    r14 = uint_to_uniform_float(r14)
    r15 = uint_to_uniform_float(r15)
    off_0 = tl.program_id(0) * BLOCK * UNROLL + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    off_4 = off_3 + BLOCK
    off_5 = off_4 + BLOCK
    off_6 = off_5 + BLOCK
    off_7 = off_6 + BLOCK
    off_8 = off_7 + BLOCK
    off_9 = off_8 + BLOCK
    off_10 = off_9 + BLOCK
    off_11 = off_10 + BLOCK
    off_12 = off_11 + BLOCK
    off_13 = off_12 + BLOCK
    off_14 = off_13 + BLOCK
    off_15 = off_14 + BLOCK
    tl.store(out_ptr + off_0, r0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, r1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, r2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, r3, mask=off_3 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_4, r4, mask=off_4 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_5, r5, mask=off_5 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_6, r6, mask=off_6 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_7, r7, mask=off_7 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_8, r8, mask=off_8 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_9, r9, mask=off_9 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_10, r10, mask=off_10 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_11, r11, mask=off_11 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_12, r12, mask=off_12 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_13, r13, mask=off_13 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_14, r14, mask=off_14 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_15, r15, mask=off_15 < N, eviction_policy="evict_first")


def rand(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS RAND")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    # grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
    cluster_num = 12
    UNROLL = choose_unroll(N)
    BLOCK_SIZE = min(triton.next_power_of_2(triton.cdiv(N, cluster_num * UNROLL)), 1024)
    grid_fn = triton.cdiv(N, BLOCK_SIZE * UNROLL)
    # (TODO) Using Triton autotuner makes kernel parameters opaque to the caller,
    # hence we cannot obtain the per thread offset as in Pytorch.
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)
    with torch_device_fn.device(device):
        if UNROLL <= 4:
            rand_kernel_1[(grid_fn,)](
                out, N, philox_seed, philox_offset, BLOCK_SIZE, UNROLL
            )
        else:
            rand_kernel_2[(grid_fn,)](
                out, N, philox_seed, philox_offset, BLOCK_SIZE, UNROLL
            )
    return out
