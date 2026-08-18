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
import math

import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_atan2 = tl_extra_shim.atan2


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def atan2_kernel(x, y):
    x = x.to(tl.float32)
    y = y.to(tl.float32)
    result = _atan2(x, y)
    pi = math.pi

    x_negative = x.to(tl.int32, bitcast=True) < 0
    y_negative = y.to(tl.int32, bitcast=True) < 0

    # XPU atan2f returns zero for (+/-0, negative axis). Preserve the signed
    # zero on the positive axis and return signed pi on the negative axis.
    zero_result = tl.where(
        y_negative,
        tl.where(x_negative, -pi, pi),
        x,
    )
    result = tl.where(x == 0.0, zero_result, result)

    # XPU atan2f returns NaN when both arguments are infinite.
    x_inf = (x == float("inf")) | (x == -float("inf"))
    y_inf = (y == float("inf")) | (y == -float("inf"))
    inf_angle = tl.where(y_negative, 2.356194490192345, 0.7853981633974483)
    inf_angle = tl.where(x_negative, -inf_angle, inf_angle)
    result = tl.where(x_inf & y_inf, inf_angle, result)

    return tl.where((x != x) | (y != y), float("nan"), result)


def atan2(input, other):
    logger.debug("GEMS_KUNLUNXIN ATAN2")
    return atan2_kernel(input, other)


def atan2_out(input, other, out):
    logger.debug("GEMS_KUNLUNXIN ATAN2_OUT")
    return atan2_kernel(input, other, out0=out)


def atan2_(input, other):
    logger.debug("GEMS_KUNLUNXIN ATAN2_")
    atan2_kernel(input, other, out0=input)
    return input
