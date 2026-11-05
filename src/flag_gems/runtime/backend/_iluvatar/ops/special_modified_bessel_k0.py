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

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice as tld

# ---------------------------------------------------------------------------
# special_modified_bessel_k0: K0(x) elementwise, matching
# torch.special.modified_bessel_k0 semantics:
#   x == 0 (incl. -0.0) -> +inf ; x < 0 -> nan ; +inf -> 0 ; nan -> nan
#
# Piecewise evaluation (all in input dtype):
#   (0, 2]   : K0(x) = -ln(x/2)*I0(x) - gamma*I0(x) + sum_k H_k (x^2/4)^k/(k!)^2
#              (log-power series; well conditioned for x <= 2)
#   (2, 16]  : K0(x) = exp(-x - ln(x)/2) * G(u)
#              G(u) = Chebyshev fit of sqrt(x)*exp(x)*K0(x) in u = (ln x - zm)/zh,
#              evaluated via even/odd split in w = u^2 (deg 16, 9+8 Horner FMAs)
#   (16, inf): K0(x) = exp(-x - ln(x)/2)*sqrt(pi/2) * sum_k (-1)^k a_k x^-k
#              a_k = ((2k-1)!!)^2 / (k! 8^k), Horner in inv_x (6 terms, fp32)
#
# exp(-x)/sqrt(x) is evaluated as exp(-x - 0.5*ln(x)) sharing one log and one exp.
# fp32 fast path uses libdevice fast_logf/fast_expf; fp64 keeps accurate tl.log/tl.exp
# and the Clenshaw/recurrence forms (correctness-only workloads).
# ---------------------------------------------------------------------------

_GAMMA = tl.constexpr(0.57721566490153286060651209)
_LN2 = tl.constexpr(0.693147180559945309417232121458)
_SQRT_PI_2 = tl.constexpr(1.2533141373155002512078826424)  # sqrt(pi/2)
_ZM = tl.constexpr(1.7328679513998633)  # (ln2 + ln16)/2
_INV_ZH = tl.constexpr(0.9617966939269759)  # 1/((ln16 - ln2)/2)

# 1/(k!)^2 for k = 0..16
_INV_FACT2 = tl.constexpr(
    (
        1.0,
        1.0,
        0.25,
        0.027777777777777776,
        0.001736111111111111,
        6.944444444444444e-05,
        1.9290123456790124e-06,
        3.936759890764382e-08,
        6.151185684944348e-10,
        7.594091145610924e-12,
        7.594091145610925e-14,
        6.276081841000765e-16,
        4.358365861806087e-18,
        2.5789451422527105e-20,
        1.3157842962516883e-22,
        5.84791849465639e-25,
        2.2843831894498118e-27,
    )
)

# H_k/(k!)^2 for k = 1..16
_SERIES_S = tl.constexpr(
    (
        1.0,
        0.375,
        0.05092592592592593,
        0.003616898148148148,
        1.585648148148148e-04,
        4.72608024691358e-06,
        1.0208210975292198e-07,
        1.6718143711739155e-09,
        2.148384211531542e-11,
        2.224249736525898e-13,
        1.8952831151530028e-15,
        1.3525013811801745e-17,
        8.201317563762432e-20,
        4.278369452479303e-22,
        1.9404901199433318e-24,
        7.722681165341764e-27,
    )
)

# Chebyshev coefficients (u in [-1,1]) of G(x) = sqrt(x)*exp(x)*K0(x),
# u = (ln(x) - zm)/zh, x in [2,16] (fp64 reference path, Clenshaw)
_CZ = tl.constexpr(
    (
        1.2225166152138542,
        0.026201462915342606,
        -0.005484236167287214,
        0.000646103969284398,
        -3.123428150509246e-05,
        -2.9548033262328333e-06,
        6.272241888731216e-07,
        -2.839463032690986e-08,
        -4.227059469921566e-09,
        6.626071739258636e-10,
        -5.255066000087535e-12,
        -7.040711699034538e-12,
        5.555735309487951e-13,
        4.335278089534191e-14,
        -9.156894572959324e-15,
        4.776429207276447e-17,
        9.95554838928744e-17,
    )
)

# Even/odd split of the degree-16 power basis of the z-fit: G(u) = E(u^2) + u*O(u^2)
_EVEN_C = tl.constexpr(
    (
        1.227968985654208,
        -0.010707173084895646,
        -0.00028065475928805515,
        2.1145336129879527e-05,
        -5.302424556505299e-07,
        -6.482981265367673e-09,
        1.4215655942043333e-09,
        -8.806221672648961e-11,
        3.2622340962017085e-12,
    )
)
_ODD_C = tl.constexpr(
    (
        0.0242485817947458,
        0.0026418407667928327,
        -4.380160821274271e-05,
        -2.2191761758008573e-06,
        1.901721083374389e-07,
        -7.782399041932735e-09,
        1.7463835244236966e-10,
        7.825701613201731e-13,
    )
)

# Asymptotic coefficients with alternating signs: acc = 1 + c1/x + ... + c6/x^6
_ASYC = tl.constexpr(
    (
        -0.125,
        0.0703125,
        -0.0732421875,
        0.112152099609375,
        -0.22710800170898438,
        0.5725014209747314,
    )
)


@triton.jit
def _cheb(u, c, D: tl.constexpr):
    # Clenshaw evaluation of sum_k c[k] T_k(u)
    b1 = tl.zeros_like(u)
    b2 = tl.zeros_like(u)
    for k in tl.static_range(D - 1, -1, -1):
        b0 = 2.0 * u * b1 - b2 + c[k]
        b2 = b1
        b1 = b0
    return b1 - u * b2


@triton.jit
def _k0(
    x,
    SERIES_N: tl.constexpr,
    FIT_D: tl.constexpr,
    ASY_N: tl.constexpr,
    FASTEXP: tl.constexpr,
    FASTLOG: tl.constexpr,
    SPLIT: tl.constexpr,
    EVEN_N: tl.constexpr,
    ODD_N: tl.constexpr,
):
    if FASTLOG:
        z = tld.fast_logf(x)
    else:
        z = tl.log(x)

    # --- (0, 2]: log-power series ---
    t = x * x * 0.25
    i0 = _INV_FACT2[SERIES_N]
    for k in tl.static_range(SERIES_N - 1, -1, -1):
        i0 = i0 * t + _INV_FACT2[k]
    s = _SERIES_S[SERIES_N - 1]
    for k in tl.static_range(SERIES_N - 2, -1, -1):
        s = s * t + _SERIES_S[k]
    s = s * t  # series has no constant term; starts at t^1
    k0_small = i0 * (-(z - _LN2) - _GAMMA) + s

    # --- (2, 16]: scaled Chebyshev fit; exp(-x)/sqrt(x) = exp(-x - ln(x)/2) ---
    if FASTEXP:
        exs = tld.fast_expf(-x - 0.5 * z)
    else:
        exs = tl.exp(-x - 0.5 * z)

    if SPLIT:
        u = (z - _ZM) * _INV_ZH
        w = u * u
        e = _EVEN_C[EVEN_N - 1]
        for k in tl.static_range(EVEN_N - 2, -1, -1):
            e = e * w + _EVEN_C[k]
        o = _ODD_C[ODD_N - 1]
        for k in tl.static_range(ODD_N - 2, -1, -1):
            o = o * w + _ODD_C[k]
        g = e + u * o
        k0_fit = exs * g

        # --- (16, inf): asymptotic, Horner in inv_x ---
        iv = 1.0 / x
        acc = _ASYC[ASY_N - 1]
        for k in tl.static_range(ASY_N - 2, -1, -1):
            acc = acc * iv + _ASYC[k]
        acc = acc * iv + 1.0
        k0_asy = exs * _SQRT_PI_2 * acc
    else:
        g = _cheb((z - _ZM) * _INV_ZH, _CZ, FIT_D)
        k0_fit = exs * g

        # --- (16, inf): asymptotic (recurrence) ---
        iv = 1.0 / x
        p = 1.0
        acc = 0.0
        for k in tl.static_range(0, ASY_N):
            acc += p
            p = -p * (((2 * k + 1) * (2 * k + 1)) * 0.125 / (k + 1)) * iv
        k0_asy = exs * _SQRT_PI_2 * acc

    r = tl.where(x <= 2.0, k0_small, tl.where(x <= 16.0, k0_fit, k0_asy))
    r = tl.where(x < 0.0, float("nan"), r)
    r = tl.where(x == 0.0, float("inf"), r)
    return r


@triton.jit
def _k0_kernel(
    x_ptr,
    out_ptr,
    numel,
    BLOCK: tl.constexpr,
    SERIES_N: tl.constexpr,
    FIT_D: tl.constexpr,
    ASY_N: tl.constexpr,
    FASTEXP: tl.constexpr,
    FASTLOG: tl.constexpr,
    SPLIT: tl.constexpr,
    EVEN_N: tl.constexpr,
    ODD_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask)
    r = _k0(x, SERIES_N, FIT_D, ASY_N, FASTEXP, FASTLOG, SPLIT, EVEN_N, ODD_N)
    tl.store(out_ptr + offs, r, mask=mask)


def special_modified_bessel_k0(self):
    out = torch.empty_like(self)
    numel = self.numel()
    if numel == 0:
        return out
    if self.dtype == torch.float64:
        series_n, fit_d, asy_n = 16, 16, 34
        fast = fastlog = split = False
        even_n = odd_n = 0
        block, nw = 1024, 4
    else:
        series_n, fit_d, asy_n = 8, 10, 6
        fast = fastlog = split = True
        even_n, odd_n = 9, 8
        block, nw = 512, 4
    grid = (triton.cdiv(numel, block),)
    _k0_kernel[grid](
        self,
        out,
        numel,
        BLOCK=block,
        SERIES_N=series_n,
        FIT_D=fit_d,
        ASY_N=asy_n,
        FASTEXP=fast,
        FASTLOG=fastlog,
        SPLIT=split,
        EVEN_N=even_n,
        ODD_N=odd_n,
        num_warps=nw,
    )
    return out
