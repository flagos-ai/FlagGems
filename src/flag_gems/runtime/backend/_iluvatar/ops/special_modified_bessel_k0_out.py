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

import triton
import triton.language as tl


@triton.jit
def _bessel_k0_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    xf = tl.load(x_ptr + offs, mask=mask, other=1.0).to(tl.float32)

    # K0(x) for x in (0, 2]: polyA(x^2) - ln(x/2) * I0(x), z = x^2
    A0: tl.constexpr = -0.5772156662238627
    A1: tl.constexpr = 0.10569610951863766
    A2: tl.constexpr = 0.014418425003611863
    A3: tl.constexpr = 0.0005452805203049161
    A4: tl.constexpr = 1.0168234548592042e-05
    A5: tl.constexpr = 1.2624375317665714e-07
    C1: tl.constexpr = 0.25
    C2: tl.constexpr = 0.015625
    C3: tl.constexpr = 0.00043402777777777775
    C4: tl.constexpr = 6.781684027777777e-06
    C5: tl.constexpr = 6.781684027777777e-08

    z = xf * xf
    pa = ((((A5 * z + A4) * z + A3) * z + A2) * z + A1) * z + A0
    i0 = ((((C5 * z + C4) * z + C3) * z + C2) * z + C1) * z + 1.0
    lv = tl.log(xf) - 0.6931471805599453  # ln(x/2)
    small = pa - lv * i0

    # K0(x) for x > 2: exp(-x) * polyB(8/x - 2) / sqrt(x)
    B0: tl.constexpr = 1.218595320076757
    B1: tl.constexpr = -0.015535826056784944
    B2: tl.constexpr = 0.0007583365990185348
    B3: tl.constexpr = -5.972755518543968e-05
    B4: tl.constexpr = 6.09727252241585e-06
    B5: tl.constexpr = -9.028674026957553e-07
    B6: tl.constexpr = 1.4817562974450183e-07

    w = 8.0 / xf - 2.0
    pb = (((((B6 * w + B5) * w + B4) * w + B3) * w + B2) * w + B1) * w + B0
    large = tl.exp(-xf) * pb * tl.rsqrt(xf)

    y = tl.where(xf <= 2.0, small, large)
    y = tl.where(xf <= 0.0, float("inf"), y)
    tl.store(out_ptr + offs, y, mask=mask)


def special_modified_bessel_k0_out(self, out):
    x = self
    n = x.numel()
    BLOCK_SIZE = 512
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _bessel_k0_kernel[grid](
        x.reshape(-1),
        out.reshape(-1),
        n,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out
