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

"""poisson candidate v2: lambda-dependent Knuth loop, 4 uniforms per draw.

The production gcu400 ``poisson_kernel`` accepts ``MAX_ITERS`` as a compile-time
constant, so ``poisson__opt`` could only trim it from 64 to 16 -- a number that
is still wrong at both ends: benchmark inputs draw lambda ~ U(0,1) and need ~2
uniforms, while correctness inputs with lambda=5 need ~10.  This file launches
its own kernel instead:

* the Knuth loop bound comes from the *block-max* lambda with a tail margin
  (``lam + 3*sqrt(lam) + 3`` uniforms, capped at 64). ``N ~ 1 + Poisson(lam)``,
  so the truncation probability per element is <= 1e-5 over the whole range the
  small path serves (lam < 30);
* one ``tl.philox`` produces four uniforms, so a lam~1 block costs 2 philox
  calls instead of 16;
* the normal-approximation branch is only entered when some lane in the block
  actually needs it (the benchmark's U(0,1) input never does).

Per-element counters stay at stride 64 (``increment`` matches), so the random
streams of neighbouring elements never overlap.  The sequence differs from the
old implementation, which is fine: poisson is a sampler and the tests check
mean/variance, not specific values.

Counters are built in uint32 arithmetic like the original ``poisson.py`` kernel
(int64 ops are emulated and cost ~1.0-1.8 us at 1M elements); a constexpr widens
to int64 only for tensors big enough that ``off * STRIDE`` would wrap.

Measured (op-level do_bench median, one config per process, 3 repeats each,
spread < 0.1 us):

    N=4096      40.9 -> 5.7 us
    N=2^16      41.9 -> 10.9 us (fp32) / 13.6 us (fp16, bf16)
    N=2^20     432.6 -> 86.2 us (fp32) / 99.9 us (fp16, bf16)

BLOCK is 512 for tiny inputs (the whole benchmark-small case is one 4096-element
tile; 512 spreads it over 8 CTAs), 2048 for 16-bit dtypes and 4096 for fp32
above that.  The 16-bit split is worth its own bucket: loading half/bf16 and
widening it stalls (tcu: 1D data hazard 24%, scalar hazard 33%, vs 8%/10% for
fp32), and the 2048 tile is 16% faster than 4096 there while fp32 prefers 4096.
"""

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._enflame.gcu400.ops.poisson import NUM_SIPS
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

__all__ = ["poisson"]

_BLOCK_SMALL = 512
_BLOCK_LARGE = 4096
_BLOCK_LARGE_16BIT = 2048
_SMALL_N = 16384
_LAMBDA_THRESHOLD = 30.0
# Cap on uniforms per element (hard limit of the Knuth loop).  lam < 30 needs
# at most ~49 uniforms, so the cap never binds on the small-lambda path.
_MAX_UNIFORMS = 64
_PHILOX_STRIDE = 64
# off * STRIDE must stay inside uint32 for the fast counter path.
_U32_COUNTER_MAX_ELEMS = 2**32 // _PHILOX_STRIDE


@triton.jit
def _poisson_large(lam, seed, c0, c1, z):
    """Normal approximation (Box-Muller), used only for lam > threshold."""
    r0, r1, r2, r3 = tl.philox(seed, c0, c1, z, z)
    u1 = tl.maximum(uint_to_uniform_float(r0), 1e-10)
    u2 = uint_to_uniform_float(r1)
    two_pi = 6.283185307179586
    radius = tl.sqrt(-2.0 * tl.log(u1))
    result = lam + tl.sqrt(lam) * (radius * tl.cos(two_pi * u2))
    return tl.floor(tl.maximum(result, 0.0) + 0.5)


@triton.jit
def _knuth_step(p, k, L, r):
    u = tl.maximum(uint_to_uniform_float(r), 1e-10)
    p = p * u
    return p, tl.where(p > L, k + 1, k)


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def poisson_kernel_dyn(
    inp_ptr,
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    LAMBDA_THRESHOLD: tl.constexpr,
    MAX_UNIFORMS: tl.constexpr,
    STRIDE: tl.constexpr,
    WIDE: tl.constexpr,
):
    seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0_base = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    ar = tl.arange(0, BLOCK)
    for blk in tl.range(pid, (N + BLOCK - 1) // BLOCK, npid):
        off = blk * BLOCK + ar
        mask = off < N
        lam = tl.maximum(
            tl.load(inp_ptr + off, mask=mask, other=0.0).to(tl.float32), 0.0
        )
        bmax = tl.max(lam)
        # Knuth needs N ~ 1 + Poisson(lam) uniforms; three sigma covers the tail.
        need = tl.minimum(bmax + 3.0 * tl.sqrt(bmax) + 3.0, MAX_UNIFORMS * 1.0)
        nit = (need * 0.25).to(tl.int32) + 1

        if WIDE:
            counter = c0_base.to(tl.int64) + off.to(tl.int64) * STRIDE
            c0 = (counter & 0xFFFFFFFF).to(tl.uint32)
            c1v = ((counter >> 32) & 0xFFFFFFFF).to(tl.uint32)
        else:
            # uint32 keeps the per-element counter arithmetic off the (emulated)
            # int64 path; off * STRIDE cannot wrap because WIDE is False.
            c0 = c0_base + off.to(tl.uint32) * STRIDE
            c1v = c1
        z = c0 * 0

        L = tl.exp(-lam)
        p = lam * 0.0 + 1.0
        k = (lam * 0).to(tl.int32)
        for _ in tl.range(0, nit):
            r0, r1, r2, r3 = tl.philox(seed, c0, c1v, z, z)
            p, k = _knuth_step(p, k, L, r0)
            p, k = _knuth_step(p, k, L, r1)
            p, k = _knuth_step(p, k, L, r2)
            p, k = _knuth_step(p, k, L, r3)
            c0 = c0 + 1

        result = k.to(tl.float32)
        if bmax > LAMBDA_THRESHOLD:
            large = _poisson_large(lam, seed, c0, c1v, z)
            result = tl.where(lam <= LAMBDA_THRESHOLD, result, large)
        tl.store(out_ptr + off, result, mask=mask)


def poisson(input, generator=None):
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

    block = _BLOCK_SMALL
    if N > _SMALL_N:
        if inp.dtype in (torch.float16, torch.bfloat16):
            block = _BLOCK_LARGE_16BIT
        else:
            block = _BLOCK_LARGE
    grid = min((N + block - 1) // block, NUM_SIPS * 2)
    increment = triton.cdiv(N * _PHILOX_STRIDE, 4)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(inp.device):
        poisson_kernel_dyn[(grid,)](
            inp,
            out,
            N,
            philox_seed,
            philox_offset,
            BLOCK=block,
            LAMBDA_THRESHOLD=_LAMBDA_THRESHOLD,
            MAX_UNIFORMS=_MAX_UNIFORMS,
            STRIDE=_PHILOX_STRIDE,
            WIDE=N > _U32_COUNTER_MAX_ELEMS,
            num_warps=4,
        )
    return out
