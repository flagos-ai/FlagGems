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

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@triton.jit
def transform_exponential_f32(u, inv_lambd, eps_minus):
    log = tl.where(u >= 1.0 + eps_minus, eps_minus, tl.math.log(u))
    return -inv_lambd * log


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def fused_exponential_kernel_f32(
    out_ptr,
    N,
    inv_lambd,
    eps_minus,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    c0 += i
    z = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, z, z)
    # When a uint32 sample happens to be exactly 0, u = 0 and log(0) = -inf, producing inf in the output (probability
    # 2^-32 per sample;
    # at 1G elements this is almost certain to happen, turning the tests' mean≈1 assertion into inf).
    # Lift 0 to 1 in the integer domain (the smallest nonzero sample of u is 2^-31, giving y = 21.49);
    # this only adds one integer select and does not affect the later floating-point vectorization (clamping in the
    # float domain pushes log off the hardware vector path, measured 50% slower).
    one = z + 1  # uint32 constant 1 with the same dtype as the philox output
    r0 = tl.maximum(r0, one)
    r1 = tl.maximum(r1, one)
    r2 = tl.maximum(r2, one)
    r3 = tl.maximum(r3, one)

    y0 = transform_exponential_f32(uint_to_uniform_float(r0), inv_lambd, eps_minus)
    y1 = transform_exponential_f32(uint_to_uniform_float(r1), inv_lambd, eps_minus)
    y2 = transform_exponential_f32(uint_to_uniform_float(r2), inv_lambd, eps_minus)
    y3 = transform_exponential_f32(uint_to_uniform_float(r3), inv_lambd, eps_minus)

    start = pid.to(tl.uint64) * BLOCK * 4
    off0 = start + tl.arange(0, BLOCK)
    off1 = off0 + BLOCK
    off2 = off1 + BLOCK
    off3 = off2 + BLOCK

    tl.store(out_ptr + off0, y0, mask=off0 < N)
    tl.store(out_ptr + off1, y1, mask=off1 < N)
    tl.store(out_ptr + off2, y2, mask=off2 < N)
    tl.store(out_ptr + off3, y3, mask=off3 < N)


def exponential_(x, lambd: float = 1.0, *, generator=None):
    logger.debug("GEMS_ENFLAME EXPONENTIAL_")

    dtype = x.dtype
    device = x.device
    inplace = x.is_contiguous()
    assert dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)

    N = x.numel()
    inv_lambd = 1.0 / lambd
    eps_minus = -0.5 * torch.finfo(dtype).eps

    out = x if inplace else torch.empty_like(x)

    UNROLL = 4
    if N >= 67108864:
        BLOCK = 32768
    elif N >= 65536:
        BLOCK = 8192
    else:
        BLOCK = min(triton.next_power_of_2(N), 1024)
    grid_size = triton.cdiv(N, BLOCK * UNROLL)
    # Every program draws BLOCK philox counters, including the lanes its masked
    # tail stores drop, so a call consumes exactly grid_size * BLOCK counters.
    # `N // UNROLL` under-reserves whenever BLOCK > 1 (e.g. small N), and the
    # next call then replays part of this call's random stream (Issue #7162).
    increment = grid_size * BLOCK
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )

    # Write directly to the target dtype: the kernel computes in f32 and store implicitly converts to fp16/bf16,
    # numerically equivalent to the old "write f32 intermediate buffer -> .to(dtype) -> copy_" path (both round the f32
    # result once to the
    # target precision), but saves a full f32 allocation, one read+write of an f32 buffer, and two extra kernel
    # launches.
    out_buf = out.view(-1) if inplace else out

    with torch_device_fn.device(device):
        fused_exponential_kernel_f32[(grid_size,)](
            out_buf,
            N,
            inv_lambd,
            eps_minus,
            philox_seed,
            philox_offset,
            BLOCK,
            num_warps=1,
        )

    if not inplace:
        x.copy_(out)
    return x
