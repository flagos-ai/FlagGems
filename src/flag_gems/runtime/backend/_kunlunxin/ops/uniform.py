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
import logging

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
    r0 = uint_to_uniform_float(r0) * (to - from_) + from_
    r1 = uint_to_uniform_float(r1) * (to - from_) + from_
    r2 = uint_to_uniform_float(r2) * (to - from_) + from_
    r3 = uint_to_uniform_float(r3) * (to - from_) + from_
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    tl.store(out_ptr + off_0, r0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, r1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, r2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, r3, mask=off_3 < N, eviction_policy="evict_first")


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


def uniform_(self, from_=0.0, to=1.0, *, generator=None):
    logger.debug("GEMS_KUNLUNXIN UNIFORM_")
    N = volume(self.shape)
    BLOCK, num_warps = _launch_config(N)
    grid = (triton.cdiv(N, BLOCK * UNROLL),)

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
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    return self
