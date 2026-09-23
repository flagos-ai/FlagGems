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


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def sinc_func(x):
    x_f32 = x.to(tl.float32)
    nearest = tl.floor(x_f32 + 0.5)
    remainder = x_f32 - nearest
    parity = nearest - 2.0 * tl.floor(nearest * 0.5)
    sign = 1.0 - 2.0 * parity

    u = 3.141592653589793 * remainder
    u2 = u * u
    sinc_u = -1.0 / 1307674368000.0
    sinc_u = sinc_u * u2 + 1.0 / 6227020800.0
    sinc_u = sinc_u * u2 - 1.0 / 39916800.0
    sinc_u = sinc_u * u2 + 1.0 / 362880.0
    sinc_u = sinc_u * u2 - 1.0 / 5040.0
    sinc_u = sinc_u * u2 + 1.0 / 120.0
    sinc_u = sinc_u * u2 - 1.0 / 6.0
    sinc_u = sinc_u * u2 + 1.0

    denominator = tl.where(x_f32 == 0.0, 1.0, x_f32)
    result = sign * (remainder / denominator) * sinc_u
    return tl.where(x_f32 == 0.0, 1.0, result)


def sinc(A):
    logger.debug("GEMS_KUNLUNXIN SINC")
    return sinc_func(A)


def sinc_(A):
    logger.debug("GEMS_KUNLUNXIN SINC_")
    sinc_func(A, out0=A)
    return A


def special_sinc(A):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_SINC")
    return sinc_func(A)
