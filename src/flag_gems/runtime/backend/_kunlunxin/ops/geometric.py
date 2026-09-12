# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)
UNROLL = 4


@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "p"])
def geometric_kernel(
    out_ptr,
    N,
    p,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += offsets
    zeros = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, zeros, zeros)

    log1p_minus_p = tl.log(1.0 - p)
    y0 = tl.ceil(tl.log(uint_to_uniform_float(r0)) / log1p_minus_p)
    y1 = tl.ceil(tl.log(uint_to_uniform_float(r1)) / log1p_minus_p)
    y2 = tl.ceil(tl.log(uint_to_uniform_float(r2)) / log1p_minus_p)
    y3 = tl.ceil(tl.log(uint_to_uniform_float(r3)) / log1p_minus_p)

    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    tl.store(out_ptr + off_0, y0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, y1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, y2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, y3, mask=off_3 < N, eviction_policy="evict_first")


def _launch_config(N):
    if N <= 512:
        return 512, 4
    if N <= 1024:
        return 1024, 8
    return 1024, 16


def geometric(input, p=0.5, *, generator=None):
    logger.debug("GEMS_KUNLUNXIN GEOMETRIC")
    out = torch.empty_like(input, device=input.device)
    N = volume(out.shape)
    if N == 0:
        return out

    BLOCK, num_warps = _launch_config(N)
    grid = (triton.cdiv(N, BLOCK * UNROLL),)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(input.device):
        geometric_kernel[grid](
            out,
            N,
            p,
            philox_seed,
            philox_offset,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    return out


def geometric_(input, p=0.5, *, generator=None):
    logger.debug("GEMS_KUNLUNXIN GEOMETRIC_")
    N = volume(input.shape)
    if N == 0:
        return input

    BLOCK, num_warps = _launch_config(N)
    grid = (triton.cdiv(N, BLOCK * UNROLL),)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(input.device):
        geometric_kernel[grid](
            input,
            N,
            p,
            philox_seed,
            philox_offset,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    return input
