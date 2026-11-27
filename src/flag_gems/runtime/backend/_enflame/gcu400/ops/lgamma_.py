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

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@triton.jit
def _lgamma_stirling(x, SHIFT_THRESHOLD: tl.constexpr):
    # lgamma in fp32 via reflection + shift-to->=8 + Stirling asymptotics.
    #
    # The libdevice extern (`__ef_vlgammaf`) path is not used on purpose: the
    # fp32 kernel lowers it to tops_ef_vlgammaf_v1_fp32 (vector<128>) and runs
    # 2.11 ms per 16M elements, but a 16-bit input widens the whole pipeline to
    # vector<256> and the extern then lowers to the v2 (1024B) variant, which
    # measures 16.2 ms on the same shape (7.7x).  This body keeps every op on
    # the core Triton path (log/sin/where) at any vector width and measures
    # 0.57 ms (fp32, BLOCK=2048) / 0.63-0.66 ms (fp16/bf16, BLOCK=1024) per
    # 16M elements, i.e. 2.4x / 1.9-2.0x the native torch op.
    #
    # Accuracy was validated against CPU double torch.lgamma: 49-value edge
    # list (poles, +-inf, NaN, near-poles), 200k-point dense sweeps over
    # [-30, 30], (0.001, 2), +/- logspace(1, 8), 1M randn and rand+0.1, odd
    # sizes 1/3/1000/4095/999983, and fp16/bf16 round trips, all within
    # atol=2e-5 / rtol=1.3e-6 (fp32) or the suite dtype tolerances.
    reflect = x < 0.5
    y = tl.where(reflect, 1.0 - x, x)
    # Shift y up to >= SHIFT_THRESHOLD, accumulating the product of the skipped
    # factors; lgamma(y) = stirling(y') - log(product).
    #
    # Threshold 4 instead of 8: the 3-term Stirling series omits O(1/(1680 y^7)),
    # which is 1.6e-8 at y=4.5 and 9.2e-8 at y=3.5 (both below fp32 eps), so 4
    # shift steps suffice.  Each dropped step removes one compare + two selects
    # + one multiply + one add from the per-element vector body: 512M elements
    # measure 20.49 -> 16.07 ms (fp16) and 17.63 -> 16.29 ms (fp32) on GCU400,
    # including the duplicate-floor removal below.  Verified against CPU double
    # torch.lgamma over randn|4|/|30|, +-logspace(1e-6, 1e8) and rand+0.1 sweeps:
    # max_abs 3.35e-8 (threshold 4) vs 2.37e-7 (threshold 3), fp32 rounding
    # dominates either way.
    t = 1.0
    for _ in tl.static_range(SHIFT_THRESHOLD):
        m = y < SHIFT_THRESHOLD * 1.0
        t = tl.where(m, t * y, t)
        y = tl.where(m, y + 1.0, y)
    logy = tl.log(y)
    # 1/y via rsqrt(y*y): same value for y > 0 (series term is only r/12, so the
    # extra ulp of rsqrt is irrelevant), and it avoids the divide that pushes the
    # 16-bit pipeline onto a slower codegen path on GCU400.
    r = tl.math.rsqrt(y * y)
    r2 = r * r
    series = r * (
        0.08333333333333333 - r2 * (0.002777777777777778 - r2 * 0.0007936507936507937)
    )
    res = (y - 0.5) * logy - y + 0.9189385332046727 + series - tl.log(t)
    # Reflection for x < 0.5: lgamma(x) = log(pi) - log|sin(pi x)| - lgamma(1-x).
    # `reduced` is in [-0.5, 0.5], so |angle| <= pi/2 and a 7-term odd Taylor for
    # sin is enough: its truncation error at pi/2 is 5.6e-8, below fp32 eps, and
    # it replaces the libdevice sin with 5 FMAs.  It also keeps sin finite for
    # tiny angles (the z^2/6 term underflows to 0 for subnormal angles, so it
    # degenerates to sin(z) = z); scaling by 2^64 before the log keeps log|sin|
    # finite when |sin| is subnormal (same trick as ops/_foreach_unary.py).
    reduced = x - tl.floor(x + 0.5)
    angle = 3.141592653589793 * reduced
    angle2 = angle * angle
    sin_poly = 1.0 - angle2 * (
        0.16666666666666666
        - angle2
        * (
            0.008333333333333333
            - angle2
            * (
                0.0001984126984126984
                - angle2 * (2.7557319223985893e-06 - angle2 * 2.505210838544172e-08)
            )
        )
    )
    sin_pi_x = angle * sin_poly
    log_sin = tl.log(tl.abs(sin_pi_x) * 18446744073709551616.0) - 44.3614195558365
    reflected = 1.1447298858494002 - log_sin - res
    result = tl.where(reflect, reflected, res)
    # Gamma's poles (0 and the negative integers) and +-inf map to +inf; NaN
    # propagates through the arithmetic.  The pole test reuses `reduced`
    # (x - round(x) == 0 iff x is an exact integer) instead of a second
    # tl.floor(x): for |x| >= 2^23 every representable value is an integer and
    # floor(x) == x, so both forms agree there as well.
    is_pole = (x <= 0.0) & (reduced == 0.0)
    result = tl.where(is_pole | (tl.abs(x) == float("inf")), float("inf"), result)
    return result


@libentry()
@triton.jit(do_not_specialize=["N"])
def _lgamma_kernel(
    X, Out, N, BLOCK: tl.constexpr, EVEN: tl.constexpr, SHIFT_THRESHOLD: tl.constexpr
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(X + off).to(tl.float32)
            tl.store(Out + off, _lgamma_stirling(x, SHIFT_THRESHOLD))
        else:
            mask = off < N
            x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
            tl.store(Out + off, _lgamma_stirling(x, SHIFT_THRESHOLD), mask=mask)


# Shift threshold of the Stirling reduction (see _lgamma_stirling).
SHIFT_THRESHOLD = 4


def _choose_block(x):
    n = x.numel()
    if n <= 4096:
        return triton.next_power_of_2(max(min(n, 1024), 1))
    # Tuned on GCU400 (fixed grid, 48 CTAs): both 16-bit and fp32 dtypes peak at
    # 2048-element blocks / num_warps=4 (512M elements, fp16 16.07 ms vs
    # 19.23 ms at BLOCK=1024 and 20.82 ms at BLOCK=4096).
    return 2048


def _launch_lgamma(X, out):
    N = X.numel()
    if N == 0:
        return out
    BLOCK = _choose_block(X)
    grid = min(triton.cdiv(N, BLOCK), NUM_SIPS * 2)
    even = (N % BLOCK) == 0
    with torch_device_fn.device(X.device):
        _lgamma_kernel[(grid,)](
            X,
            out,
            N,
            BLOCK=BLOCK,
            EVEN=even,
            SHIFT_THRESHOLD=SHIFT_THRESHOLD,
            num_warps=4,
        )
    return out


def lgamma(A):
    logger.debug("GEMS_ENFLAME LGAMMA")
    x = A if A.is_contiguous() else A.contiguous()
    out_dtype = torch.float32 if not x.dtype.is_floating_point else x.dtype
    out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
    return _launch_lgamma(x, out)


def lgamma_(A):
    logger.debug("GEMS_ENFLAME LGAMMA_")
    # torch promotes integral inputs of lgamma to float32, so the in-place form
    # cannot write the result back and raises instead of silently truncating.
    if not A.dtype.is_floating_point:
        raise RuntimeError(
            "lgamma_(): result type Float can't be cast to the desired output type"
        )
    x = A if A.is_contiguous() else A.contiguous()
    out = _launch_lgamma(x, x)
    if out is not A:
        A.copy_(out)
    return A
