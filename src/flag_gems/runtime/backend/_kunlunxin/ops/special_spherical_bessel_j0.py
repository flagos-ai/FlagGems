# Copyright 2026, The FlagOS Contributors.
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

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

# j0(x) = sin(x) / x for |x| > _POLY_MAX and the Taylor series of sin(x)/x
# below it.  The series is used because the absolute error of sin() here is
# ~1e-7 regardless of magnitude, which the division amplifies by 1/|x|.
_POLY_MAX = tl.constexpr(2.0)
_C1 = tl.constexpr(1.0 / 6.0)
_C2 = tl.constexpr(1.0 / 120.0)
_C3 = tl.constexpr(1.0 / 5040.0)
_C4 = tl.constexpr(1.0 / 362880.0)
_C5 = tl.constexpr(1.0 / 39916800.0)
_C6 = tl.constexpr(1.0 / 6227020800.0)
_C7 = tl.constexpr(1.0 / 1307674368000.0)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def spherical_bessel_j0_func(x):
    x_f32 = x.to(tl.float32)
    x2 = x_f32 * x_f32
    poly = 1.0 - x2 * (
        _C1 - x2 * (_C2 - x2 * (_C3 - x2 * (_C4 - x2 * (_C5 - x2 * (_C6 - x2 * _C7)))))
    )
    ax = tl.abs(x_f32)
    safe_x = tl.where(ax > _POLY_MAX, x_f32, 1.0)
    quotient = tl.sin(x_f32) / safe_x
    result = tl.where(ax > _POLY_MAX, quotient, poly)
    return result.to(x.dtype)


def special_spherical_bessel_j0(A):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_SPHERICAL_BESSEL_J0")
    return spherical_bessel_j0_func(A)
