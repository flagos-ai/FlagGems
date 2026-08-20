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
    unroll_num=4,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def exp2_func(x):
    LN2 = 0.69314718056
    return tl.exp(x.to(tl.float32) * LN2)


def exp2(A):
    logger.debug("GEMS_KUNLUNXIN EXP")
    return exp2_func(A)


def exp2_(A):
    logger.debug("GEMS_KUNLUNXIN EXP_")
    return exp2_func(A, out0=A)
