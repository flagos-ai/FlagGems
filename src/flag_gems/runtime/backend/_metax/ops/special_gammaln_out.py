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


def _gammaln_kernel(
    x_ptr, y_ptr, n_elements, SIMPLE_DEG: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    xin = tl.load(x_ptr + offs, mask=mask, other=1.0)
    x = xin.to(tl.float32)
    z = tl.where(x >= 0.5, x, 1.0 - x)

    if tl.max(z) >= 8.0:
        # General path: Stirling series with shift-up product-log correction.
        is_shift = z < 8.0
        t = tl.where(is_shift, z + 8.0, z)
        inv = 1.0 / t
        w = inv * inv
        corr = inv * (
            0.08333333333333333
            - w
            * (
                0.002777777777777778
                - w * (0.0007936507936507937 - w * 0.0005952380952380953)
            )
        )
        s = tl.math.fma(tl.log(t), t - 0.5, -t) + 0.9189385332046727 + corr
        prod = (z + 0.0) * (z + 1.0) * (z + 2.0) * (z + 3.0)
        prod = prod * (z + 4.0) * (z + 5.0) * (z + 6.0) * (z + 7.0)
        s = s - tl.where(is_shift, tl.log(prod), 0.0)
        is_pole = (x <= 0.0) & (x == tl.floor(x))
        # fr = exact distance to the nearest integer, in [0, 0.5].
        fr = tl.abs(x - tl.floor(x + 0.5))
        # Reflection: lgamma(x) = ln(pi) - ln|sin(pi x)| - lgamma(1-x), with
        # ln|sin(pi fr)| = ln(pi fr) + ln(sinc(pi fr)).  The sinc log is a
        # degree-9 polynomial q(fr^2) (error 2e-7); multiplying fr by pi
        # before the log keeps the argument normal even for subnormal fr.
        vv = fr * fr
        uq = tl.math.fma(vv, 2.0, -1.0)
        qacc = -0.0007543119718320668
        qacc = qacc * uq + -0.006569105666130781
        qacc = qacc * uq + -0.02698003500699997
        qacc = qacc * uq + -0.070713110268116
        qacc = qacc * uq + -0.1368228793144226
        qacc = qacc * uq + -0.218607559800148
        qacc = qacc * uq + -0.32305991649627686
        qacc = qacc * uq + -0.5101579427719116
        qacc = qacc * uq + -1.3451014757156372
        qacc = qacc * uq + -1.0266709327697754
        y = tl.where(
            x >= 0.5, s, 1.1447298858494002 - tl.log(3.141592653589793 * fr) - qacc - s
        )
        y = tl.where(is_pole, float("inf"), y)
        y = tl.where((x == float("inf")) | ((x > 1.0e36) & (y != y)), float("inf"), y)
    else:
        # Fast path (z in [0.5, 8)): h(z) = lgamma(z) + ln z as a power-basis
        # polynomial in u = (2z - 8.5)/7.5, then s = h - ln z.  fp32 uses a
        # degree-12 fit (error 1.0e-5); fp16/bf16 outputs are quantized coarsely
        # enough that a degree-9 fit (error 1.7e-4) plus a degree-5 sinc-log
        # fit (error 2.9e-7) stays well inside tolerance and saves 6 FMAs.
        lnz = tl.log(z)
        us = tl.math.fma(z, 0.26666666666666666, -1.1333333333333333)
        if SIMPLE_DEG:
            acc = -0.024941788986325264
            acc = acc * us + 0.03504735976457596
            acc = acc * us + 0.0019227073062211275
            acc = acc * us + 0.011964146047830582
            acc = acc * us + -0.08059623092412949
            acc = acc * us + 0.159642294049263
            acc = acc * us + -0.38346970081329346
            acc = acc * us + 1.473739743232727
            acc = acc * us + 5.849823951721191
            acc = acc * us + 3.5613977909088135
        else:
            acc = 0.010197031311690807
            acc = acc * us + -0.01351966243237257
            acc = acc * us + -0.011071274988353252
            acc = acc * us + 0.01046862080693245
            acc = acc * us + 0.023314960300922394
            acc = acc * us + -0.03162567317485809
            acc = acc * us + 0.03204618766903877
            acc = acc * us + -0.06678149849176407
            acc = acc * us + 0.15121126174926758
            acc = acc * us + -0.3857722878456116
            acc = acc * us + 1.4748328924179077
            acc = acc * us + 5.849930286407471
            acc = acc * us + 3.5613763332366943
        s = acc - lnz
        # Reflection as in the general path: exact fr, sinc-log polynomial.
        fr = tl.abs(x - tl.floor(x + 0.5))
        vv = fr * fr
        uq = tl.math.fma(vv, 2.0, -1.0)
        if SIMPLE_DEG:
            qacc = -0.01248519029468298
            qacc = qacc * uq + -0.07405003905296326
            qacc = qacc * uq + -0.21489930152893066
            qacc = qacc * uq + -0.4603884816169739
            qacc = qacc * uq + -1.3323280811309814
            qacc = qacc * uq + -1.0252739191055298
        else:
            qacc = -0.0007543119718320668
            qacc = qacc * uq + -0.006569105666130781
            qacc = qacc * uq + -0.02698003500699997
            qacc = qacc * uq + -0.070713110268116
            qacc = qacc * uq + -0.1368228793144226
            qacc = qacc * uq + -0.218607559800148
            qacc = qacc * uq + -0.32305991649627686
            qacc = qacc * uq + -0.5101579427719116
            qacc = qacc * uq + -1.3451014757156372
            qacc = qacc * uq + -1.0266709327697754
        y = tl.where(
            x >= 0.5, s, 1.1447298858494002 - tl.log(3.141592653589793 * fr) - qacc - s
        )

    y = y.to(xin.dtype)
    tl.store(y_ptr + offs, y, mask=mask)


_BLOCK = 512
_NUM_WARPS = 1  # large workloads: B512/W1 = 16 elems/thread, max ILP
_SMALL_N = 1 << 20  # at/below this element count, prefer more warps
_SMALL_WARPS = 4  # small shapes: hide fixed launch overhead
_TINY_N = 1 << 16  # tiny shapes: smaller blocks give more parallel CTAs
_TINY_BLOCK = 512


def special_gammaln_out(A, *, out=None):
    if out is None:
        out = torch.empty_like(A)
    n = A.numel()
    if n == 0:
        return out
    # Zero-copy: contiguous tensors are indexed linearly by the kernel, so no
    # reshape call is needed (avoids two Python view dispatches per launch,
    # which matters at the ~9us launch-bound 64x64 scale).
    x = A if A.is_contiguous() else A.reshape(-1)
    y = out if out.is_contiguous() else out.reshape(-1)
    simple = A.dtype != torch.float32
    if n < _TINY_N:
        grid = (triton.cdiv(n, _TINY_BLOCK),)
        _gammaln_kernel[grid](
            x, y, n, SIMPLE_DEG=simple, BLOCK=_TINY_BLOCK, num_warps=_SMALL_WARPS
        )
    elif n <= _SMALL_N:
        grid = (triton.cdiv(n, _BLOCK),)
        _gammaln_kernel[grid](
            x, y, n, SIMPLE_DEG=simple, BLOCK=_BLOCK, num_warps=_SMALL_WARPS
        )
    else:
        grid = (triton.cdiv(n, _BLOCK),)
        _gammaln_kernel[grid](
            x, y, n, SIMPLE_DEG=simple, BLOCK=_BLOCK, num_warps=_NUM_WARPS
        )
    return out
