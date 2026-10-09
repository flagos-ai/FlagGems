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


def _y0_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    xin = tl.load(x_ptr + offs, mask=mask)
    x = xin.to(tl.float32)

    # ---- ln(x) via exponent extraction + Chebyshev log2(1+t) ----
    bits = x.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 255) - 127
    m = ((bits & 8388607) | 1065353216).to(tl.float32, bitcast=True)
    t = m - 1.0
    b1 = 0.0
    b2 = 0.0
    b0 = -0.0007758643478155136 + 2.0 * t * b1 - b2
    b2 = b1
    b1 = b0
    b0 = 0.00736964400857687 + 2.0 * t * b1 - b2
    b2 = b1
    b1 = b0
    b0 = -0.038701143115758896 + 2.0 * t * b1 - b2
    b2 = b1
    b1 = b0
    b0 = 0.15031495690345764 + 2.0 * t * b1 - b2
    b2 = b1
    b1 = b0
    b0 = -0.506317138671875 + 2.0 * t * b1 - b2
    b2 = b1
    b1 = b0
    b0 = 1.8564926385879517 + 2.0 * t * b1 - b2
    b2 = b1
    b1 = b0
    log2m = -0.46838679909706116 + t * b1 - b2
    log2x = e.to(tl.float32) + log2m
    lnx = log2x * 0.6931471824645996

    # ---- J0(x) = (z-z1)(z-z2)(z-z3) * G(z), G Chebyshev in z (x in [0, 11.5]) ----
    z = x * x
    zv = z * 0.015122873708605766 - 1.0
    b1 = 0.0
    b2 = 0.0
    b0 = -1.7662836526710635e-08 + 2.0 * zv * b1 - b2
    b2 = b1
    b1 = b0
    b0 = 1.470450996521322e-07 + 2.0 * zv * b1 - b2
    b2 = b1
    b1 = b0
    b0 = -9.551602033752715e-07 + 2.0 * zv * b1 - b2
    b2 = b1
    b1 = b0
    b0 = 4.569631528283935e-06 + 2.0 * zv * b1 - b2
    b2 = b1
    b1 = b0
    b0 = -1.52469901877339e-05 + 2.0 * zv * b1 - b2
    b2 = b1
    b1 = b0
    b0 = 3.3124648325610906e-05 + 2.0 * zv * b1 - b2
    b2 = b1
    b1 = b0
    Gv = -2.1714875401812606e-05 + zv * b1 - b2
    J0 = (
        (z - 5.783185958862305)
        * (z - 30.471261978149414)
        * (z - 74.88700866699219)
        * Gv
    )

    # ---- R(x) = Y0(x) - (2/pi) ln(x/2) J0(x), Chebyshev in x (x in [0, 11.5]) ----
    xn = x * 0.17391304671764374 - 1.0
    b1 = 0.0
    b2 = 0.0
    b0 = -0.00016433862037956715 + 2.0 * xn * b1 - b2
    b2 = b1
    b1 = b0
    b0 = 0.00037805194733664393 + 2.0 * xn * b1 - b2
    b2 = b1
    b1 = b0
    b0 = 0.002360419137403369 + 2.0 * xn * b1 - b2
    b2 = b1
    b1 = b0
    b0 = -0.004128860309720039 + 2.0 * xn * b1 - b2
    b2 = b1
    b1 = b0
    b0 = -0.021984761580824852 + 2.0 * xn * b1 - b2
    b2 = b1
    b1 = b0
    b0 = 0.025391334667801857 + 2.0 * xn * b1 - b2
    b2 = b1
    b1 = b0
    b0 = 0.11646384745836258 + 2.0 * xn * b1 - b2
    b2 = b1
    b1 = b0
    b0 = -0.05987974628806114 + 2.0 * xn * b1 - b2
    b2 = b1
    b1 = b0
    b0 = -0.2736000418663025 + 2.0 * xn * b1 - b2
    b2 = b1
    b1 = b0
    b0 = -0.01517760381102562 + 2.0 * xn * b1 - b2
    b2 = b1
    b1 = b0
    b0 = 0.12276669591665268 + 2.0 * xn * b1 - b2
    b2 = b1
    b1 = b0
    b0 = -0.20524205267429352 + 2.0 * xn * b1 - b2
    b2 = b1
    b1 = b0
    Rv = 0.16293668746948242 + xn * b1 - b2

    # ---- Y0(x) = (2/pi) ln(x/2) J0(x) + R(x) ----
    y = 0.6366197466850281 * (lnx - 0.6931471824645996) * J0 + Rv
    ysel = tl.where(x == 0.0, float("-inf"), y)
    ysel = tl.where(x < 0.0, float("nan"), ysel)
    tl.store(y_ptr + offs, ysel.to(xin.dtype), mask=mask)


def special_bessel_y0(A):
    out = torch.empty_like(A)
    n = A.numel()
    if n < (1 << 16):
        BLOCK = 256
        num_warps = 4
    elif n < (1 << 26):
        BLOCK = 1024
        num_warps = 2
    else:
        BLOCK = 2048
        num_warps = 2
    grid = (triton.cdiv(n, BLOCK),)
    _y0_kernel[grid](A, out, n, BLOCK=BLOCK, num_warps=num_warps)
    return out
