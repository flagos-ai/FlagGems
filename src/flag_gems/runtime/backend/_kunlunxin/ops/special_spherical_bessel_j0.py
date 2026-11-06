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

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)

# Series coefficients are written as literals: a Triton @jit body cannot read
# non-constexpr module-level python globals.
#   -1/6         = -0.16666666666666666
#    1/120       =  0.008333333333333333
#   -1/5040      = -0.0001984126984126984
#    1/362880    =  2.7557319223985893e-06
#   -1/39916800  = -2.505210838544172e-08
#    1/6227020800=  1.6059043836821613e-10


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def spherical_bessel_j0_func(x):
    xf = x.to(tl.float32)
    ax = tl.abs(xf)
    use_series = ax < 0.5

    # ----- |x| < 0.5 : ATen's degree-6 series in t = x^2 -----
    t = xf * xf
    ans_series = 1.0 + t * (
        -0.16666666666666666
        + t
        * (
            0.008333333333333333
            + t
            * (
                -0.0001984126984126984
                + t
                * (
                    2.7557319223985893e-06
                    + t * (-2.505210838544172e-08 + t * 1.6059043836821613e-10)
                )
            )
        )
    )

    # ----- |x| >= 0.5 : sin(x)/x ; clamp the series lanes away from 0 -----
    xs = tl.where(use_series, 1.0, xf)
    ans_sin = tl.sin(xs) / xs

    result = tl.where(use_series, ans_series, ans_sin)
    # j0(+-inf) = 0 (ATen returns 0 for infinite input; sin(inf)/inf is NaN).
    result = tl.where(ax == float("inf"), 0.0, result)
    return result.to(x.dtype)


def special_spherical_bessel_j0(A):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_SPHERICAL_BESSEL_J0")
    return spherical_bessel_j0_func(A)
