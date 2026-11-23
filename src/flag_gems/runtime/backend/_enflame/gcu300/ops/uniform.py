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

import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)


@triton.jit
def _next_below(x, DT: tl.constexpr):
    """Largest ``DT`` value strictly less than ``x``.

    ``uniform`` samples from ``[from, to)``. The value is computed in float32
    and then stored in the output dtype, and that last rounding can land exactly
    on the exclusive upper bound for float16/bfloat16 (measured on GCU300:
    17 of 96000 float16 and 133 of 96000 bfloat16 draws of [-3, 3) come out as
    exactly 3.0). Clamping to the output-dtype predecessor of ``to`` keeps the
    stored samples inside the half-open interval.

    A zero input needs care: the bit patterns of ``+0.0`` and ``-0.0`` are the
    bottom of the positive block and the top of the negative block, so the usual
    "-1 for x >= 0, +1 otherwise" step walks *out* of the number line and wraps
    around -- ``+0.0`` (0) minus one is all-ones, and ``-0.0`` plus one is
    all-ones too, both of which are NaN. The true predecessor of either zero is
    ``-smallest_subnormal``, so a zero input is answered directly.
    """
    zero = x == 0
    if DT == tl.float32:
        bits = x.to(tl.int32, bitcast=True)
        step = tl.where(x >= 0, -1, 1).to(tl.int32)
        stepped = (bits + step).to(tl.float32, bitcast=True)
        tiny = tl.full((), 1, tl.int32).to(tl.float32, bitcast=True)
        return tl.where(zero, -tiny, stepped).to(tl.float32)
    elif DT == tl.float64:
        bits = x.to(tl.int64, bitcast=True)
        step = tl.where(x >= 0, -1, 1).to(tl.int64)
        stepped = (bits + step).to(tl.float64, bitcast=True)
        tiny = tl.full((), 1, tl.int64).to(tl.float64, bitcast=True)
        return tl.where(zero, -tiny, stepped).to(tl.float64)
    else:
        bits = x.to(tl.int16, bitcast=True)
        step = tl.where(x >= 0, -1, 1).to(tl.int16)
        stepped = (bits + step).to(DT, bitcast=True)
        tiny = tl.full((), 1, tl.int16).to(DT, bitcast=True)
        return tl.where(zero, -tiny, stepped).to(DT)


@triton.heuristics(runtime.get_heuristic_config("uniform"))
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
    philox_seed = philox_seed.to(tl.uint32)
    philox_offset = philox_offset.to(tl.uint32)
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
    # The scaled values are float32 but are stored in the output dtype, whose
    # rounding can land exactly on the exclusive ``to`` bound (notably for
    # float16/bfloat16). Clamp to the output-dtype predecessor of ``to`` so the
    # samples stay in ``[from, to)`` after the store conversion. When the
    # interval is too narrow to survive the dtype (``to`` rounds down to
    # ``from``) the unclamped value is kept rather than producing ``from``.
    OUT_DT: tl.constexpr = out_ptr.dtype.element_ty
    to_dt = (r0 * 0.0 + to).to(OUT_DT)
    from_dt = (r0 * 0.0 + from_).to(OUT_DT)
    hi = _next_below(to_dt, OUT_DT)
    hi = tl.where(to_dt > from_dt, hi, to_dt)
    r0 = tl.minimum(r0.to(OUT_DT), hi)
    r1 = tl.minimum(r1.to(OUT_DT), hi)
    r2 = tl.minimum(r2.to(OUT_DT), hi)
    r3 = tl.minimum(r3.to(OUT_DT), hi)
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    tl.store(out_ptr + off_0, r0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, r1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, r2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, r3, mask=off_3 < N, eviction_policy="evict_first")


UNROLL = 4


def uniform_(self, from_=0.0, to=1.0, *, generator=None):
    logger.debug("GEMS_ENFLAME UNIFORM_")
    N = volume(self.shape)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    philox_seed = philox_seed % (2 ^ 32)
    philox_offset = philox_offset % (2 ^ 32)
    with torch_device_fn.device(self.device):
        uniform_kernel[grid_fn](self, N, philox_seed, philox_offset, from_, to)
    return self
