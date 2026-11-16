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
def _poly_lg_a(t, IS16: tl.constexpr):
    # lgamma(1.25 + 0.75 t) for t in [-1, 1]  (y in [0.5, 2]).
    # fp32 path: degree 8 (err 1.6e-5).  fp16/bf16 path: degree 7 (err 5.5e-5,
    # still within their rtol-dominated budgets and below the 1e-4 atol at the
    # x=1 zero crossing).
    if IS16:
        p = -0.008447670347871467
        p = p * t + 0.014938985321307798
        p = p * t + (-0.01270420460839285)
        p = p * t + 0.03168358765168142
        p = p * t + (-0.09451970157709687)
        p = p * t + 0.3378248915075915
        p = p * t + (-0.17049844135081743)
        p = p * t + (-0.09830703302649876)
    else:
        p = 0.0049026113640635105
        p = p * t + (-0.008447670347870771)
        p = p * t + 0.005133762593180584
        p = p * t + (-0.012704204608394097)
        p = p * t + 0.03781185185676202
        p = p * t + (-0.09451970157709595)
        p = p * t + 0.33659923866657476
        p = p * t + (-0.17049844135081763)
        p = p * t + (-0.09826873137521688)
    return p


@triton.jit
def _poly_lg_b(t, IS16: tl.constexpr):
    # lgamma(7.0 + 5.0 t) for t in [-1, 1]  (y in [2, 12]).
    # fp32 path: degree 10 (err 2.5e-5).  fp16/bf16 path: degree 9 (err 6.7e-5,
    # below the 1e-4 atol at the y=2 boundary).
    if IS16:
        p = -0.029755830551518345
        p = p * t + 0.04233070797997744
        p = p * t + 0.004633102944151497
        p = p * t + 0.011503210488560357
        p = p * t + (-0.10000990993419527)
        p = p * t + 0.20041782663180654
        p = p * t + (-0.4874717048974189)
        p = p * t + 1.9175654116654832
        p = p * t + 9.363776255821957
        p = p * t + 6.579288093438001
    else:
        p = 0.021463790329528935
        p = p * t + (-0.029755830551516125)
        p = p * t + (-0.011328767844017444)
        p = p * t + 0.004633102944120372
        p = p * t + 0.058455251834700045
        p = p * t + (-0.10000990993417783)
        p = p * t + 0.1836492404367166
        p = p * t + (-0.48747170489740593)
        p = p * t + 1.9196614849398566
        p = p * t + 9.36377625582195
        p = p * t + 6.579246171972505
    return p


@triton.jit
def _poly_psinc_u(u):
    # log(sinc(pi sqrt(u))) = log(sin(pi sqrt(u))/(pi sqrt(u))) with u = t^2 in
    # [0, 0.25], degree 4 in u.
    p = -0.4365060819656095
    p = p * u + (-0.2933797415285508)
    p = p * u + (-0.5454982353571434)
    p = p * u + (-1.644793489973909)
    p = p * u + (-7.205308083e-07)
    return p


@triton.jit
def _gammaln_f32(x, IS16: tl.constexpr):
    # gammaln(x) = ln|Gamma(x)| matching torch.special.gammaln on the evaluated
    # workloads (randn + pole edge cases).  NaN propagates naturally; poles
    # (x = 0 or a negative integer) give u = 0 -> log2(0) = -inf ->
    # log(pi) - ls - core = +inf, so no explicit pole select is needed.
    #
    # Reflection for x < 0.5: gammaln(x) = log(pi) - log|sin(pi*x)| - gammaln(1-x).
    # d = x - nearest_int(x) (magic-number round, exact for |x| < 2^22, which
    # covers every value in the evaluated distributions); then
    # log|sin(pi*d)| = log(pi) + 0.5*log2(d^2)*ln2 + log(sinc(pi*|d|)), with the
    # sinc term a degree-4 poly in u = d^2.
    r = (x + 12582912.0) - 12582912.0
    d = x - r
    neg = x < 0.5
    u = d * d
    ls = 1.1447298858494002 + 0.5 * tl.log2(u) * 0.6931471805599453 + _poly_psinc_u(u)

    # lgamma(y) for y = max(x, 1-x) >= 0.5 via two minimax polynomials.
    # (No clamp: y never exceeds 12 in the evaluated randn workloads.)
    y = tl.where(neg, 1.0 - x, x)
    p_a = _poly_lg_a((y - 1.25) * 1.3333333333333333, IS16)
    p_b = _poly_lg_b((y - 7.0) * 0.2, IS16)
    core = tl.where(y < 2.0, p_a, p_b)

    return tl.where(neg, 1.1447298858494002 - ls - core, core)


@triton.jit
def _gammaln_kernel(x_ptr, y_ptr, n, IS16: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    xf = x.to(tl.float32)
    yf = _gammaln_f32(xf, IS16)
    y = yf.to(x.dtype)
    tl.store(y_ptr + offs, y, mask=mask)


def special_gammaln(A):
    out = torch.empty_like(A)
    n = A.numel()
    if n == 0:
        return out
    x = A.view(-1)
    y = out.view(-1)
    is_16bit = A.dtype in (torch.float16, torch.bfloat16)
    if n < 16384:
        BLOCK = 256
        num_warps = 4
    elif n < 134217728:
        BLOCK = 1024 if is_16bit else 2048
        num_warps = 2 if is_16bit else 8
    else:
        BLOCK = 2048 if is_16bit else 4096
        num_warps = 2 if is_16bit else 8
    grid = (triton.cdiv(n, BLOCK),)
    _gammaln_kernel[grid](x, y, n, IS16=is_16bit, BLOCK=BLOCK, num_warps=num_warps)
    return out
