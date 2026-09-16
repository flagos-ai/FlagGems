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
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def acosh_func(x):
    # acosh(x) = log(x + sqrt(x*x - 1))
    # Domain: [1, inf), returns NaN for x < 1 (except x=1 returns 0)
    x32 = x.to(tl.float32)
    # Using the formula: acosh(x) = log(x + sqrt(x*x - 1))
    sqrt_term = tl.sqrt(x32 * x32 - 1.0)
    result = tl.log(x32 + sqrt_term)
    return result.to(x.dtype)


def acosh(A):
    logger.debug("GEMS_KUNLUNXIN ACOSH")
    return acosh_func(A)


def acosh_(A):
    logger.debug("GEMS_KUNLUNXIN ACOSH_")
    acosh_func(A, out0=A)
    return A
