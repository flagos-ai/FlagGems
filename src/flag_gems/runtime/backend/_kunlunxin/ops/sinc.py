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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sinc_func(x):
    px = math.pi * x.to(tl.float32)
    px2 = px * px
    near_zero = tl.abs(px) < 1.0e-2
    series = 1.0 + px2 * (-1.0 / 6.0 + px2 * (1.0 / 120.0 - px2 / 5040.0))
    return tl.where(near_zero, series, tl.sin(px) / px)


def sinc(A):
    logger.debug("GEMS_KUNLUNXIN SINC")
    return sinc_func(A)


def sinc_(A):
    logger.debug("GEMS_KUNLUNXIN SINC_")
    sinc_func(A, out0=A)
    return A
