# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)
NUM_SIPS = 24


@triton.jit
def poisson_small_lambda(lam, seed, c0, c1, z, MAX_ITERS: tl.constexpr):
    L = tl.exp(-lam)
    k = (lam * 0).to(tl.int32)
    p = lam * 0.0 + 1.0
    for _ in range(MAX_ITERS):
        r0, r1, r2, r3 = tl.philox(seed, c0, c1, z, z)
        u = uint_to_uniform_float(r0)
        u = tl.maximum(u, 1e-10)
        p = p * u
        k = tl.where(p > L, k + 1, k)
        c0 = c0 + 1
    return k.to(tl.float32)


@triton.jit
def poisson_large_lambda(lam, seed, c0, c1, z):
    r0, r1, r2, r3 = tl.philox(seed, c0, c1, z, z)
    u1 = uint_to_uniform_float(r0)
    u2 = uint_to_uniform_float(r1)
    u1 = tl.maximum(u1, 1e-10)
    two_pi = 6.283185307179586
    r = tl.sqrt(-2.0 * tl.log(u1))
    theta = two_pi * u2
    normal_sample = r * tl.cos(theta)
    result = lam + tl.sqrt(lam) * normal_sample
    result = tl.maximum(result, 0.0)
    result = tl.floor(result + 0.5)
    return result


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def poisson_kernel(
    inp_ptr,
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    LAMBDA_THRESHOLD: tl.constexpr,
    MAX_ITERS: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0_base = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(pid, (N + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N
        lam = tl.load(inp_ptr + off, mask=mask, other=0.0).to(tl.float32)
        lam = tl.maximum(lam, 0.0)
        use_small = lam < LAMBDA_THRESHOLD
        c0_small = c0_base + off.to(tl.uint32) * MAX_ITERS
        z = c0_small * 0
        small_result = poisson_small_lambda(
            lam, philox_seed, c0_small, c1, z, MAX_ITERS
        )
        c0_large = c0_base + off.to(tl.uint32)
        z_large = c0_large * 0
        large_result = poisson_large_lambda(lam, philox_seed, c0_large, c1, z_large)
        result = tl.where(use_small, small_result, large_result)
        tl.store(out_ptr + off, result, mask=mask)


def poisson(input, generator=None):
    logger.debug("GEMS_ENFLAME POISSON")
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), f"Unsupported dtype: {input.dtype}"

    inp = input.contiguous()
    N = volume(inp.shape)
    out = torch.empty_like(inp)
    if N == 0:
        return out

    LAMBDA_THRESHOLD = 30
    MAX_ITERS = 64
    BLOCK = 8192
    grid = min((N + BLOCK - 1) // BLOCK, NUM_SIPS * 2)
    increment = triton.cdiv(N * MAX_ITERS, 4)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )

    with torch_device_fn.device(inp.device):
        poisson_kernel[(grid,)](
            inp,
            out,
            N,
            philox_seed,
            philox_offset,
            BLOCK=BLOCK,
            LAMBDA_THRESHOLD=LAMBDA_THRESHOLD,
            MAX_ITERS=MAX_ITERS,
            num_warps=4,
        )
    return out
