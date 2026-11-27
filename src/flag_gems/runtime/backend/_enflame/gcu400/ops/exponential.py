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

"""gcu400-specific out-of-place ``exponential``.

The generic implementation launches ``cdiv(N, BLOCK * 8)`` programs with an
autotuned BLOCK of at most 2048.  For the benchmark shape ``(1024**3,)`` that
exceeds the GCU400 grid.x limit of 65535 (measured: ``Required: 2097152``),
so the op fails with ``OutOfResources`` before doing any work.

This variant keeps one program per ``BLOCK * 4`` elements (grid.x stays inside
the limit for every benchmark shape) and draws its uniforms from the 32-bit
hash used by the rest of this backend's RNG kernels instead of the
philox4x32-10 chain:

* philox4x32-10 costs ~20 dependent integer vector ops per output element,
  which made the kernel compute bound: 1 Gi 16-bit ran at 0.5 of the memory
  roofline (measured 9.20 ms for 4 GiB of traffic, versus 2.79 ms at the
  1.4 TB/s the same board reaches on a pure copy);
* the xxhash32-style ``_fmix32`` finalizer is 8 integer ops per element and is
  the same generator family as ``randn``/``native_dropout`` in this backend,
  so the statistical contract of ``aten::exponential`` is unchanged (the
  accuracy suite asserts |mean - 1|, |var - 1| < 0.05, ``y.min() > 0`` and the
  far-left-tail atom bounds, none of which pin a bitwise stream);
* the per-call state still comes from and advances the torch generator through
  ``philox_backend_seed_offset``, so successive calls draw fresh samples and a
  seeded generator reproduces the same call sequence.

Two further constants of the previous kernel are gone rather than tuned:
``u`` is now built as ``k * 2**-24`` with ``k = max(hash >> 8, 1)``, which is
strictly inside ``(0, 1)``, so neither the ``u == 0`` guard on the integer side
nor the ``u >= 1 + eps`` select on the float side is needed (dropping the
select keeps the log on the hardware vector path and removes one compare plus
one select per element).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.libentry import libentry
from flag_gems.utils.random_utils import philox_backend_seed_offset

from .randn import _fmix32

logger = logging.getLogger(__name__)

UNROLL = 4
U24 = tl.constexpr(5.960464477539063e-8)  # 2**-24
FP16_MIN_NORMAL = 6.103515625e-05  # 2**-14

# Post-sweep policy (GCU400, official do_bench kernel mode, median).  The hash
# kernel removes most of the per-element integer work, so the optimum moves
# from "one huge warp-less tile" to a mid-size tile with two warps:
#   1 Gi fp16: b32768/w1 9.20 ms (previous policy) -> b8192/w2 6.53 ms
#   2^24 fp16: b16384/w1 0.157 ms                 -> b8192/w2 0.110 ms
#   1 Gi fp32: b16384/w2 7.86 ms (b8192/w2 8.01 ms, within noise)
# Small inputs keep a 1024-wide tile and a single warp, which measured fastest
# there (4096 elements: 4.8 us) because the launch floor dominates.
BIG_BLOCK = 8192
BIG_WARPS = 2
SMALL_BLOCK = 1024
SMALL_WARPS = 1
SMALL_N = 65536


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def exponential_kernel(
    out_ptr,
    N,
    inv_lambd,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    MASKED: tl.constexpr = True,
    WIDE: tl.constexpr = False,
    MIN_Y: tl.constexpr = 0.0,
):
    # One avalanche of the per-call RNG state: keying the per-element hash with
    # the raw offset would make successive same-shape calls return the same
    # multiset of samples (see the note in randn.py).  Scalar prologue, once
    # per program.
    rng_state = philox_offset.to(tl.int64) ^ philox_seed.to(tl.int64)
    s0 = _fmix32((rng_state ^ (rng_state >> 32)).to(tl.uint32))

    pid = tl.program_id(0)
    if WIDE:
        # Element offsets past the int32 domain need 64-bit arithmetic.
        off_0 = pid.to(tl.int64) * (BLOCK * 4) + tl.arange(0, BLOCK)
    else:
        off_0 = pid * (BLOCK * 4) + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    h0 = _fmix32(off_0.to(tl.uint32) ^ s0)
    h1 = _fmix32(off_1.to(tl.uint32) ^ s0)
    h2 = _fmix32(off_2.to(tl.uint32) ^ s0)
    h3 = _fmix32(off_3.to(tl.uint32) ^ s0)

    # 24-bit uniform on (0, 1): ``h >> 8`` is in [0, 2**24 - 1] and the integer
    # floor of 1 turns u = 0 into u = 2**-24, so log stays finite and the
    # largest sample is 16.64.  u < 1 upholds ``exponential must be positive``.
    u0 = tl.maximum(h0 >> 8, 1).to(tl.float32) * U24
    u1 = tl.maximum(h1 >> 8, 1).to(tl.float32) * U24
    u2 = tl.maximum(h2 >> 8, 1).to(tl.float32) * U24
    u3 = tl.maximum(h3 >> 8, 1).to(tl.float32) * U24

    # y = -inv_lambd * ln(u).  The log2(u) * ln2 rewrite that helps the randn
    # kernel was measured here and is *slower* for every dtype of this one
    # (1 Gi: fp16 7.08 -> 6.53 ms, fp32 8.70 -> 7.98 ms), so the direct tl.log
    # form stays and no per-dtype log arm is needed.
    y0 = -inv_lambd * tl.log(u0)
    y1 = -inv_lambd * tl.log(u1)
    y2 = -inv_lambd * tl.log(u2)
    y3 = -inv_lambd * tl.log(u3)

    if MIN_Y > 0.0:
        # Only 16-bit stores need this: the left tail of Exp(1) reaches
        # 5.96e-8, which is a fp16 *subnormal*, and a flush-to-zero store would
        # turn it into 0 and break the ``y.min() > 0`` contract.  Flooring at
        # the smallest normal fp16 value moves ~6e-5 of the mass onto
        # 6.1e-5, which is far below any statistical tolerance.
        y0 = tl.maximum(y0, MIN_Y)
        y1 = tl.maximum(y1, MIN_Y)
        y2 = tl.maximum(y2, MIN_Y)
        y3 = tl.maximum(y3, MIN_Y)

    if MASKED:
        tl.store(out_ptr + off_0, y0, mask=off_0 < N)
        tl.store(out_ptr + off_1, y1, mask=off_1 < N)
        tl.store(out_ptr + off_2, y2, mask=off_2 < N)
        tl.store(out_ptr + off_3, y3, mask=off_3 < N)
    else:
        tl.store(out_ptr + off_0, y0)
        tl.store(out_ptr + off_1, y1)
        tl.store(out_ptr + off_2, y2)
        tl.store(out_ptr + off_3, y3)


def exponential(x, lambd: float = 1.0, *, generator=None):
    logger.debug("GEMS_ENFLAME EXPONENTIAL")

    dtype = x.dtype
    device = x.device
    assert dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)

    N = x.numel()
    inv_lambd = 1.0 / lambd
    out = torch.empty_like(x)
    if N == 0:
        return out

    if N >= SMALL_N:
        BLOCK = BIG_BLOCK
        num_warps = BIG_WARPS
    else:
        BLOCK = min(triton.next_power_of_2(N), SMALL_BLOCK)
        num_warps = SMALL_WARPS

    group = BLOCK * UNROLL
    grid_size = triton.cdiv(N, group)
    # Every program hashes BLOCK counters, including the lanes its masked tail
    # stores drop, so a call consumes exactly grid_size * BLOCK counters.  A
    # short increment would let the next call replay part of this call's stream
    # (Issue #7162).
    philox_seed, philox_offset = philox_backend_seed_offset(
        grid_size * BLOCK, generator=generator
    )

    with torch_device_fn.device(device):
        exponential_kernel[(grid_size,)](
            out,
            N,
            inv_lambd,
            philox_seed,
            philox_offset,
            BLOCK,
            MASKED=N % group != 0,
            WIDE=N > 2**31 - group,
            MIN_Y=FP16_MIN_NORMAL if dtype is torch.float16 else 0.0,
            num_warps=num_warps,
        )

    return out
