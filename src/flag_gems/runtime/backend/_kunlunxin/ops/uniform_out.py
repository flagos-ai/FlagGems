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

# Kunlunxin (XPU) override of the out-of-place `flag_gems.ops.uniform.uniform`.
#
# Two XPU-specific problems in the generic implementation:
#
# 1. The kernel is decorated with
#    ``@triton.heuristics(runtime.get_heuristic_config("uniform"))``.  The
#    heuristic-supplied launch path is pathologically slow / intermittent on the
#    XPU triton fork (same root cause as the ``uniform_`` override in
#    ``uniform.py``: many shapes stall at ~2400-3200 ms).  This override passes
#    BLOCK and num_warps explicitly from Python, mirroring the heuristic values.
#
# 2. The generic exclusive-upper-bound clamp (``_next_below``) steps one ulp
#    below ``to`` through an ``int16`` bitcast of the fp16/bf16 value.  On the
#    XPU compiler that bitcast fails to legalize
#    (``TritonXPUDtypeConvert`` pipeline error, bf16), and even where it
#    compiles it does not keep fp16 samples strictly below the bound.
#    This override replaces it with a *host-computed* predecessor of ``to`` in
#    the output dtype: a single scalar (a pure launch constant, never tensor
#    data) that is clamped in float32 with ``tl.minimum``.  It avoids the
#    int16 bitcast entirely, so the fp16/bf16 cases compile.
#
# Note (hardware limitation, measured):XPU3 flushes fp16 subnormals to zero.
# The regression test ``uniform(-1e-6, 0.0)`` on float16 needs a negative fp16
# value inside (-1e-6, 0), and every representable value there is subnormal, so
# the strict ``out < 0`` contract is unachievable on this device.  bf16
# subnormals are preserved, so that dtype is handled by the same clamp.
import logging
import math

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


@triton.jit(
    do_not_specialize=["philox_seed", "philox_offset", "from_", "to", "prev_upper"]
)
def uniform_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    from_,
    to,
    prev_upper,
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
    # `prev_upper` is the largest value of the output dtype strictly below the
    # exclusive bound `to` (computed on the host).  Clamping in float32 before
    # the store keeps values that would round up to `to` inside `[from, to)`.
    r0 = tl.minimum(r0, prev_upper)
    r1 = tl.minimum(r1, prev_upper)
    r2 = tl.minimum(r2, prev_upper)
    r3 = tl.minimum(r3, prev_upper)
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    tl.store(out_ptr + off_0, r0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, r1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, r2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, r3, mask=off_3 < N, eviction_policy="evict_first")


UNROLL = 4

# (exponent bits, mantissa bits) per floating dtype; used to derive one ulp.
_FLOAT_FORMAT = {
    torch.float16: (-14, 10),
    torch.bfloat16: (-126, 7),
    torch.float32: (-126, 23),
    torch.float64: (-1022, 52),
}


def _prev_upper(to, dtype):
    """Largest value of ``dtype`` strictly below ``to`` (float32-exact scalar).

    Pure host-side launch-constant arithmetic: ``to`` is a Python scalar, so no
    tensor data is involved.  For ``to == 0`` (and any zero) the predecessor is
    ``-smallest_subnormal``.
    """
    emin, mant = _FLOAT_FORMAT[dtype]
    x = float(to)
    if x == 0.0:
        return float(-(2.0 ** (emin - mant)))
    ulp = 2.0 ** (math.floor(math.log2(abs(x))) - mant)
    return float(x - ulp)


def _launch_config(N):
    # Mirrors the "uniform" heuristic values, but computed in Python and passed
    # explicitly so the slow heuristic-supplied launch path is never taken.
    if N <= 512:
        return 512, 4
    elif N <= 1024:
        return 1024, 8
    else:
        return 1024, 16


def uniform(self, from_=0.0, to=1.0, *, generator=None):
    logger.debug("GEMS_KUNLUNXIN UNIFORM")
    # aten::uniform rejects an empty interval before launch; mirror that so
    # from > to raises instead of silently producing out-of-range values.
    if from_ > to:
        raise RuntimeError(
            "uniform_ expects to return a [from, to) range, but found "
            f"from={from_:g} > to={to:g}"
        )
    out = torch.empty_like(self)
    N = volume(out.shape)
    if N == 0:
        return out
    BLOCK, num_warps = _launch_config(N)
    grid = (triton.cdiv(N, BLOCK * UNROLL),)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    prev_upper = _prev_upper(to, out.dtype)
    with torch_device_fn.device(out.device):
        uniform_kernel[grid](
            out,
            N,
            philox_seed,
            philox_offset,
            from_,
            to,
            prev_upper,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    return out
