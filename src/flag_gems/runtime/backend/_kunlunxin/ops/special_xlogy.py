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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def special_xlogy_func(x, y):
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    y_is_nan = y_fp32 != y_fp32
    x_is_zero = x_fp32 == 0.0
    log_y = tl.log(y_fp32)
    prod = x_fp32 * log_y
    return tl.where(y_is_nan, float("nan"), tl.where(x_is_zero, 0.0, prod))


def special_xlogy(A, B):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_XLOGY")
    return special_xlogy_func(A, B)


def special_xlogy_(A, B):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_XLOGY_")
    return special_xlogy_func(A, B, out0=A)
