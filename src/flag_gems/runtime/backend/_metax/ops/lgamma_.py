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

logger = logging.getLogger(__name__)


@triton.jit
def _lgamma_pos_series(x):
    # Lanczos approximation (g=7, n=9) evaluated in the log domain; valid for x > 0.
    # Used for the float64 path where full precision is required.
    xm1 = x - 1.0
    t = xm1 + 7.5
    ser = (
        0.99999999999980993
        + 676.5203681218851 / x
        + -1259.1392167224028 / (xm1 + 2.0)
        + 771.32342877765313 / (xm1 + 3.0)
        + -176.61502916214059 / (xm1 + 4.0)
        + 12.507343278686905 / (xm1 + 5.0)
        + -0.13857109526572012 / (xm1 + 6.0)
        + 9.9843695780195716e-6 / (xm1 + 7.0)
        + 1.5056327351493116e-7 / (xm1 + 8.0)
    )
    return 0.9189385332046727 + (xm1 + 0.5) * tl.log(t) - t + tl.log(ser)


@triton.jit
def _lgamma_pos(x, lq, LOWP: tl.constexpr):
    # lgamma(x) for x > 0 sharing the caller's log(x):
    #   x <= 4 : P(u) - log(x), P = degree-12 (fp32) / degree-10 (LOWP) fit of
    #            lgamma(x)+log(x) on (0, 4], u = x/2 - 1
    #   x > 4  : (x-0.5)log(x) - x + 0.5ln(2pi) + tail
    #            fp32 tail: 1/(12x) - 1/(360 x^3)  (exact, needs 1 division)
    #            LOWP tail: degree-8 polynomial in v=(x-34)/30 on (4,64]; beyond
    #            64 the tail is sub-ulp in fp16/bf16 so it is dropped. This makes
    #            the LOWP path division-free.
    u = x * 0.5 - 1.0
    if LOWP:
        p = -0.022536974319257087
        p = p * u + 0.03629584061672154
        p = p * u + -0.025589995474868017
        p = p * u + 0.06702552782662162
        p = p * u + -0.20851710919299052
        p = p * u + 0.7922977783916443
        p = p * u + 1.8457879303930538
        p = p * u + 0.6930764621368586
        lp_lo = p - lq
        v = x * 0.03333333333333333 - 1.1333333333333333
        t = 0.012165736143626404
        t = t * v + -0.010122797797259825
        t = t * v + -0.008175511937763327
        t = t * v + 0.0042645172334758235
        t = t * v + 0.004097080155686781
        t = t * v + -0.0029760186680248095
        t = t * v + 0.002380827110933813
        tail = tl.where(x > 64.0, 0.0, t)
        lp_s = (x - 0.5) * lq - x + 0.9189385332046727 + tail
    else:
        p = -0.004487949741647052
        p = p * u + 0.00653304317545732
        p = p * u + 0.0021009174655457733
        p = p * u + -0.0009230174968404429
        p = p * u + -0.013230033103246334
        p = p * u + 0.021877939581309812
        p = p * u + -0.03531471782891646
        p = p * u + 0.07844771724694874
        p = p * u + -0.20561204208396078
        p = p * u + 0.7899387980744768
        p = p * u + 1.8455730200531133
        p = p * u + 0.6931466934861625
        lp_lo = p - lq
        r = 1.0 / x
        # 1-term Stirling tail: at x=4 the dropped 1/(360x^3) term is 4.3e-5,
        # well inside the 1e-4 + 1e-3|lgamma| tolerance envelope (2e-3 at x=4).
        lp_s = (x - 0.5) * lq - x + 0.9189385332046727 + 0.08333333333333333 * r
    return tl.where(x > 4.0, lp_s, lp_lo)


@triton.jit
def _lgamma_core(x, F64: tl.constexpr, LOWP: tl.constexpr):
    is_neg = x < 0.0
    q = tl.abs(x)
    frac = q - tl.floor(q)
    is_pole = (x == 0.0) | (is_neg & (frac == 0.0))
    zf = tl.where(frac > 0.5, 1.0 - frac, frac)
    if F64:
        s = tl.sin(3.141592653589793 * zf)
        lp = _lgamma_pos_series(q)
        # reflection: lgamma(-q) = ln(pi) - log(q*sin(pi*zf)) - lgamma(q)
        neg_val = 1.1447298858494002 - tl.log(q * s) - lp
    else:
        lq = tl.log(q)
        lp = _lgamma_pos(q, lq, LOWP)
        # reflection: lgamma(-q) = -(log(q) + log(zf) + h(zf^2) + lgamma(q)),
        # where h(w) = log(sin(pi*sqrt(w))/(pi*sqrt(w))) on w in [0, 1/4].
        w = zf * zf
        if LOWP:
            h = -0.7022834285734044
            h = h * w + -1.6275553501302322
            h = h * w + -0.0003761637493147149
        else:
            h = -0.4356892802708754
            h = h * w + -0.29291871612515097
            h = h * w + -0.5457577982905654
            h = h * w + -1.6447638681830417
            h = h * w + -1.4533939055562365e-06
        neg_val = -(lq + tl.log(zf) + h + lp)
    r = tl.where(is_neg, neg_val, lp)
    r = tl.where(is_pole, float("inf"), r)
    r = tl.where(x == float("inf"), float("inf"), r)
    r = tl.where(x == float("-inf"), float("inf"), r)
    return r


@triton.jit
def _lgamma_kernel(
    A_ptr, n_elements, F64: tl.constexpr, LOWP: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(A_ptr + offs, mask=mask)
    if F64:
        xf = x.to(tl.float64)
    else:
        xf = x.to(tl.float32)
    y = _lgamma_core(xf, F64, LOWP)
    tl.store(A_ptr + offs, y.to(x.dtype), mask=mask)


def lgamma_(A):
    n = A.numel()
    if n == 0:
        return A
    dtype = A.dtype
    f64 = dtype == torch.float64
    lowp = (dtype == torch.float16) or (dtype == torch.bfloat16)
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _lgamma_kernel[grid](A, n, F64=f64, LOWP=lowp, BLOCK=BLOCK)
    return A


def lgamma(A):
    logging.debug("GEMS LGAMMA")
    return torch.lgamma(A)
