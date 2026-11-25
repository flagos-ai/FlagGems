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

UNROLL = 4

# Persistent (grid-stride) path parameters: a fixed program count loops over the whole range.
# Measured 192 programs x BLOCK=4096 is 18%~24% faster than a naive "one program per block" grid on large shapes
# (probe_01_uniform_kernels.py gsprog/gsprog2); a non-power-of-two program count of
# 192 better matches GCU's CU count with more balanced load. Small shapes still use the naive grid (a fixed
# program count costs more than it gains).
PERSIST_BLOCK = 4096
PERSIST_NUM_PROG = 192
PERSIST_NUM_WARPS = 4
PERSIST_MIN_NUMEL = 1 << 21  # below 2M elements the naive grid is cheaper


@triton.jit
def _next_below(x, DT: tl.constexpr):
    """Largest ``DT`` value strictly less than ``x``.

    Used to build the exclusive upper bound of the sampling interval in the
    *output* dtype: ``uniform`` samples from ``[from, to)``, so a value that
    rounds up to ``to`` in a low-precision dtype must be stepped back one ulp.

    A zero input needs care: the bit patterns of ``+0.0`` and ``-0.0`` are the
    bottom of the positive block and the top of the negative block, so the
    usual "-1 for x >= 0, +1 otherwise" step walks *out* of the number line and
    wraps around -- ``+0.0`` (0) minus one is all-ones, and ``-0.0``
    (0x8000...0) plus one is all-ones too, both of which are NaN. The true
    predecessor of either zero is ``-smallest_subnormal``, so a zero input is
    answered directly as the negation of the smallest positive subnormal.
    """
    zero = x == 0
    if DT == tl.float32:
        bits = x.to(tl.int32, bitcast=True)
        step = tl.where(x >= 0, -1, 1).to(tl.int32)
        stepped = (bits + step).to(tl.float32, bitcast=True)
        # smallest positive subnormal: the bit pattern with a lone low bit.
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


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "from_", "to", "N"])
def uniform_kernel_gcu400(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    from_,
    to,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    scale = to - from_
    r0 = uint_to_uniform_float(r0) * scale + from_
    r1 = uint_to_uniform_float(r1) * scale + from_
    r2 = uint_to_uniform_float(r2) * scale + from_
    r3 = uint_to_uniform_float(r3) * scale + from_
    # Samples are computed in float32; rounding into a low-precision dtype can land exactly on the exclusive upper bound
    # ``to`` (especially fp16 / bf16), so clamp to ``to``'s
    # predecessor ``hi`` in the output dtype. Clamping is done entirely in the float32 domain: ``hi`` is exactly
    # representable in OUT_DT and
    # strictly less than ``to``, so take min in float32 then do the single fp32->OUT_DT conversion, keeping
    # ``[from, to)`` semantics and avoiding an extra per-element dtype conversion on 16-bit dtypes
    # (measured as the main bottleneck for fp16/bf16 large shapes).
    OUT_DT: tl.constexpr = out_ptr.dtype.element_ty
    to_dt = to.to(OUT_DT)
    from_dt = from_.to(OUT_DT)
    hi = _next_below(to_dt, OUT_DT)
    hi = tl.where(to_dt > from_dt, hi, to_dt)
    if OUT_DT == tl.float64:
        hi_wide = hi.to(tl.float64)
        r0 = tl.minimum(r0.to(tl.float64), hi_wide).to(OUT_DT)
        r1 = tl.minimum(r1.to(tl.float64), hi_wide).to(OUT_DT)
        r2 = tl.minimum(r2.to(tl.float64), hi_wide).to(OUT_DT)
        r3 = tl.minimum(r3.to(tl.float64), hi_wide).to(OUT_DT)
    else:
        hi_narrow = hi.to(tl.float32)
        r0 = tl.minimum(r0, hi_narrow).to(OUT_DT)
        r1 = tl.minimum(r1, hi_narrow).to(OUT_DT)
        r2 = tl.minimum(r2, hi_narrow).to(OUT_DT)
        r3 = tl.minimum(r3, hi_narrow).to(OUT_DT)
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    # When N is divisible by BLOCK*4, use an unmasked direct store: on GCU a 16-bit
    # masked store degrades to per-element writes, measured ~6% slower than unmasked.
    if EVEN:
        tl.store(out_ptr + off_0, r0)
        tl.store(out_ptr + off_1, r1)
        tl.store(out_ptr + off_2, r2)
        tl.store(out_ptr + off_3, r3)
    else:
        tl.store(out_ptr + off_0, r0, mask=off_0 < N)
        tl.store(out_ptr + off_1, r1, mask=off_1 < N)
        tl.store(out_ptr + off_2, r2, mask=off_2 < N)
        tl.store(out_ptr + off_3, r3, mask=off_3 < N)


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "from_", "to", "N"])
def uniform_kernel_persist_gcu400(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    from_,
    to,
    BLOCK: tl.constexpr,
    NUM_PROG: tl.constexpr,
):
    """grid-stride persistent version: a fixed NUM_PROG programs loop over the whole range.

    Each program handles 4*BLOCK elements per iteration (one philox call yields 4 32-bit random numbers),
    and counter intervals [start, start+BLOCK) never overlap between programs or loop iterations,
    so the randomness convention matches the naive version.
    """
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    scale = to - from_
    OUT_DT: tl.constexpr = out_ptr.dtype.element_ty
    to_dt = to.to(OUT_DT)
    hi = _next_below(to_dt, OUT_DT)
    hi = tl.where(to_dt > from_.to(OUT_DT), hi, to_dt)
    hi_narrow = hi.to(tl.float32)
    lane = tl.arange(0, BLOCK)
    _O = lane * 0
    step = NUM_PROG * BLOCK * 4
    start = tl.program_id(0) * BLOCK * 4
    while start < N:
        cur = c0 + (start + lane).to(tl.uint32)
        r0, r1, r2, r3 = tl.philox(philox_seed, cur, c1, _O, _O)
        base = start + lane
        v0 = tl.minimum(uint_to_uniform_float(r0) * scale + from_, hi_narrow).to(OUT_DT)
        v1 = tl.minimum(uint_to_uniform_float(r1) * scale + from_, hi_narrow).to(OUT_DT)
        v2 = tl.minimum(uint_to_uniform_float(r2) * scale + from_, hi_narrow).to(OUT_DT)
        v3 = tl.minimum(uint_to_uniform_float(r3) * scale + from_, hi_narrow).to(OUT_DT)
        off_1 = base + BLOCK
        off_2 = base + BLOCK * 2
        off_3 = base + BLOCK * 3
        tl.store(out_ptr + base, v0, mask=base < N)
        tl.store(out_ptr + off_1, v1, mask=off_1 < N)
        tl.store(out_ptr + off_2, v2, mask=off_2 < N)
        tl.store(out_ptr + off_3, v3, mask=off_3 < N)
        start += step


def uniform_(self, from_=0.0, to=1.0, *, generator=None):
    logger.debug("GEMS_ENFLAME UNIFORM_")
    # aten::uniform_ rejects reversed intervals; the in-place version previously lacked the same check.
    if from_ > to:
        raise RuntimeError(
            "uniform_ expects to return a [from, to) range, but found "
            f"from={from_:g} > to={to:g}"
        )
    # The kernel writes back by flat contiguous indices; non-contiguous views write a contiguous copy then copy back by
    # the original layout.
    if not self.is_contiguous():
        contiguous_out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
        uniform_(contiguous_out, from_, to, generator=generator)
        self.copy_(contiguous_out)
        return self
    N = volume(self.shape)
    dtype = self.dtype

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(self.device):
        # fp64 output has no fp64-domain clamp path in the persistent kernel, so it keeps the naive grid.
        if N >= PERSIST_MIN_NUMEL and dtype != torch.float64:
            # Large shapes use persistent grid-stride: a fixed program count avoids an oversized grid.
            uniform_kernel_persist_gcu400[(PERSIST_NUM_PROG,)](
                self,
                N,
                philox_seed,
                philox_offset,
                from_,
                to,
                BLOCK=PERSIST_BLOCK,
                NUM_PROG=PERSIST_NUM_PROG,
                num_warps=PERSIST_NUM_WARPS,
                num_stages=1,
            )
            return self

        if N <= 16384:
            BLOCK = 256
        elif N <= 65536:
            BLOCK = 1024
        elif dtype == torch.float32:
            BLOCK = 32768
        else:
            BLOCK = 16384

        grid = (triton.cdiv(N, BLOCK * UNROLL),)
        nw = 2 if dtype == torch.float32 else 4
        even = N % (BLOCK * UNROLL) == 0
        uniform_kernel_gcu400[grid](
            self,
            N,
            philox_seed,
            philox_offset,
            from_,
            to,
            BLOCK=BLOCK,
            EVEN=even,
            num_warps=nw,
            num_stages=1,
        )
    return self
