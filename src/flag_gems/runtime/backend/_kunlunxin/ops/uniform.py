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

# Kunlunxin (XPU) override of uniform_.
#
# The kernel was decorated with @triton.heuristics(get_heuristic_config("uniform")),
# letting the heuristic supply BLOCK *and* num_warps at launch time. On the XPU
# triton fork this heuristic-supplied launch path is pathologically slow /
# intermittent (IR baseline harness/perf_ir_3/ir-uniform_-dev0.log: many shapes
# stall at ~2400-3200 ms, gems speedup 0.000-0.001, even large shapes flip between
# ~4.5 ms and ~2600 ms across runs).
#
# Launching the *same* kernel with BLOCK and num_warps passed explicitly from
# Python (heuristic decorator removed) is robustly fast at every size. So this
# override drops the decorator and computes the launch config in the Python
# wrapper, mirroring the "uniform" heuristic values exactly. Kernel body /
# algorithm is unchanged (zero correctness risk). Same fix as bernoulli_.
import functools
import logging
import struct

import numpy as np
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


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def uniform_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    from_,
    to,
    hi,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    # ``hi`` is the host-computed predecessor of ``to`` in the *output* dtype.
    # Scaling into ``[from_, hi)`` instead of ``[from_, to)`` keeps every stored
    # value strictly below the exclusive ``to`` bound even after the low
    # precision store rounds: the largest scaled value is ``hi - 2**-24 *
    # (hi - from_)``, which rounds to at most ``hi < to``.  This needs no
    # in-kernel clamp, so it costs nothing on the hot path -- an in-kernel dtype
    # convert + ``tl.minimum`` measured ~40% slower on large tensors.  The
    # sampled interval is one output-dtype ulp narrower than ``[from_, to)``.
    scale = hi - from_
    r0 = uint_to_uniform_float(r0) * scale + from_
    r1 = uint_to_uniform_float(r1) * scale + from_
    r2 = uint_to_uniform_float(r2) * scale + from_
    r3 = uint_to_uniform_float(r3) * scale + from_
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    # A masked store alongside the generator compute measures ~1.8x slower than
    # the same store with no mask (67 MB fill: 4.58 ms -> 2.50 ms), so the mask
    # is only emitted for the ragged last program.
    if NEED_MASK:
        tl.store(out_ptr + off_0, r0, mask=off_0 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_1, r1, mask=off_1 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_2, r2, mask=off_2 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_3, r3, mask=off_3 < N, eviction_policy="evict_first")
    else:
        tl.store(out_ptr + off_0, r0, eviction_policy="evict_first")
        tl.store(out_ptr + off_1, r1, eviction_policy="evict_first")
        tl.store(out_ptr + off_2, r2, eviction_policy="evict_first")
        tl.store(out_ptr + off_3, r3, eviction_policy="evict_first")


UNROLL = 4


def _launch_config(N):
    # Mirrors the "uniform" heuristic values (uniform_heur_block /
    # uniform_heur_num_warps), but computed in Python and passed explicitly so
    # the slow heuristic-supplied launch path is never taken.
    if N <= 512:
        return 512, 4
    elif N <= 1024:
        return 1024, 8
    else:
        return 1024, 16


@functools.lru_cache(maxsize=None)
def _predecessor(to, dtype, from_):
    """Predecessor of ``to`` in ``dtype`` (largest value strictly < ``to``).

    For a degenerate/inverted interval (``from_ >= to``) every sample is
    ``from_`` anyway, so the clamp target is ``to`` itself.  Computed on the
    host (numpy / bit arithmetic, no torch operator call) so the kernel never
    has to bitcast the output dtype -- this TritonXPU build rejects the
    bfloat16 intra-kernel bitcast the generic helper uses.
    """
    to = float(to)
    if float(from_) >= to:
        return to
    if dtype == torch.float32:
        return float(np.nextafter(np.float32(to), np.float32(-np.inf)))
    if dtype == torch.float16:
        return float(np.nextafter(np.float16(to), np.float16(-np.inf)))
    # bfloat16 keeps the high 16 bits of a float32; step the raw bits by one
    # bfloat16 ulp (2**16 in float32-bit space).
    bits = struct.unpack("<I", struct.pack("<f", to))[0]
    if to > 0:
        bits -= 0x10000
    elif to < 0:
        bits += 0x10000
    else:  # +0.0 -> smallest negative bfloat16 subnormal
        bits = 0x80000001
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def uniform_(self, from_=0.0, to=1.0, *, generator=None):
    logger.debug("GEMS_KUNLUNXIN UNIFORM_")
    N = volume(self.shape)
    BLOCK, num_warps = _launch_config(N)
    tile = BLOCK * UNROLL
    grid = (triton.cdiv(N, tile),)
    need_mask = (N % tile) != 0
    hi = _predecessor(to, self.dtype, from_)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(self.device):
        uniform_kernel[grid](
            self,
            N,
            philox_seed,
            philox_offset,
            from_,
            to,
            hi,
            BLOCK=BLOCK,
            NEED_MASK=need_mask,
            num_warps=num_warps,
        )
    return self
