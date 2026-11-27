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

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# GCU grid.x hardware limit (65535): large pointwise tensors use a grid-stride loop.
_MAX_GRID = 65535
# Number of SIPS on GCU400 (resident grid cap consistent with other pointwise ops)
_NUM_SIPS = 24

# Round-to-nearest-even uses the classic fp32 "magic number" add/subtract trick with the
# literal 1.5 * 2**23 = 12582912.0 (kept inline: Triton kernels cannot read plain module
# globals). At or above 2**23 every fp32 value is already integral (ulp >= 1) and the
# magic addend can overshoot by one, so those lanes bypass the trick and keep the input.


def _round_config(n_elements):
    """Picks (BLOCK_SIZE, num_warps, grid_cap).

    The kernel is a pure 1-read-1-write memory stream. Measured on this card, the
    dominant cost for every dtype is CTA launch/wave-scheduling overhead rather than ALU:
    - a persistent 48-CTA grid with a grid-stride loop reaches ~1450 GB/s (fp32) and
      ~760 GB/s (16-bit) on 16M elements, versus 505 / 260 GB/s for one CTA per tile,
      i.e. 2.6-2.9x from the launch shape alone.
    - BLOCK_SIZE 8192 is the measured optimum for the persistent shape (16384 / 32768 are
      1.4x / 1.1x slower because the per-CTA footprint no longer fits the resident budget).
    - num_warps shows no measurable difference from 4 to 16; 8 keeps the register
      footprint moderate for the 32-bit case.
    - tiny tensors (n <= 4096): a single CTA covers everything, avoiding idle spin.
    """
    if n_elements <= 4096:
        return triton.next_power_of_2(max(n_elements, 1)), 4, _NUM_SIPS * 2
    return 8192, 8, _NUM_SIPS * 2


@triton.jit
def _div_by_pow10(val, S: tl.constexpr):
    """Correctly rounded ``val / S`` for integral ``val`` and S = 10^d.

    aten rescales with an IEEE division, but gcu400's fp32 ``/`` and
    ``tl.math.div_rn`` are 1-2 ulp off (26/10f gives 2.6000001 instead of
    2.5999999) and its FMA is not fused, so a reciprocal+residual refinement
    cannot recover the correctly rounded value.  ``val`` is an integer below
    2^24, so the exact quotient is formed in int64 fixed point and rounded
    once, ties to even.  Values outside that range keep the reciprocal
    multiply (their ulp is far larger than the reciprocal's error).
    """
    exact_range = (tl.abs(val) < 16777216.0) & (val == val)
    q0 = val * (1.0 / S)
    bits = q0.to(tl.int32, bitcast=True)
    e0 = ((bits >> 23) & 255) - 127
    k = tl.minimum(tl.maximum(23 - e0, 0), 40)
    u = tl.abs(val).to(tl.int64)
    A = u << k
    T = A // S
    R = A - T * S
    up = (R * 2 > S) | ((R * 2 == S) & (T % 2 == 1))
    M = T + up.to(tl.int64)
    e = e0 + tl.where(M >= (1 << 24), 1, 0)
    M = tl.where(M >= (1 << 24), M >> 1, M)
    e = e + tl.where(M < (1 << 23), -1, 0)
    M = tl.where(M < (1 << 23), M << 1, M)
    fscale = ((e - 23 + 127) << 23).to(tl.float32, bitcast=True)
    f = M.to(tl.float32) * fscale
    f = tl.where(val < 0.0, -f, f)
    return tl.where(val == 0.0, val, tl.where(exact_range, f, q0))


@libentry()
@triton.jit
def _special_round_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    SCALE: tl.constexpr,
    SCALE_INT: tl.constexpr,
    RECIP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    step = tl.num_programs(0) * BLOCK_SIZE
    for block_start in range(pid * BLOCK_SIZE, n_elements, step):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        # PyTorch uses fp32 for intermediate computation (not fp64).
        # Using fp32 matches PyTorch's behavior where fp32 precision truncation
        # can snap intermediate values to exact ties (e.g. 1.2345 * 1000 = 1234.5).
        x_f32 = x.to(tl.float32) * SCALE

        # Banker's rounding (round half to even) by letting the fp32 adder do it: adding
        # and subtracting 1.5*2**23 aligns the mantissa to a one-integer-unit step, so the
        # addend's ties-to-even rounding *is* the required rounding. This replaces the
        # previous floor/frac/modulo/compare chain (7-9 ALU ops plus a float modulo) with
        # two adds and one select, which was the ALU bottleneck for the 16-bit cases.
        abs_x = tl.math.abs(x_f32)
        rounded_abs = (abs_x + 12582912.0) - 12582912.0
        rounded_abs = tl.where(abs_x >= 8388608.0, abs_x, rounded_abs)
        result = tl.where(x_f32 < 0.0, -rounded_abs, rounded_abs)
        if SCALE_INT > 1:
            result = _div_by_pow10(result, SCALE_INT)
        elif SCALE_INT == 0:
            # Negative decimals: the rescale is an exact power-of-ten
            # multiply, which fp32 rounds correctly.
            result = result * RECIP
        # SCALE_INT == 1 (decimals == 0): the rescale is the identity, so the
        # fixed-point division is dead work.  It was ~30% of the 16-bit
        # latency on large tensors, where the kernel is ALU bound.
        result = result.to(x.dtype)

        tl.store(out_ptr + offsets, result, mask=mask)


def _launch(x, out, scale):
    n_elements = out.numel()
    if n_elements == 0:
        return out
    block_size, num_warps, grid_cap = _round_config(n_elements)
    grid = min(triton.cdiv(n_elements, block_size), grid_cap, _MAX_GRID)
    _special_round_kernel[(grid,)](
        x,
        out,
        n_elements,
        SCALE=float(scale),
        SCALE_INT=int(scale) if scale >= 1 else 0,
        RECIP=float(1.0 / scale),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out


def special_round(A, *, decimals=0):
    logger.debug("GEMS_ENFLAME SPECIAL_ROUND")
    logger.debug("GEMS_ENFLAME SPECIAL_ROUND_OUT")
    scale = float(10**decimals)
    out = torch.empty_like(A)
    return _launch(A, out, scale)


def special_round_out(self, out, *, decimals=0):
    scale = float(10**decimals)
    return _launch(self, out, scale)
