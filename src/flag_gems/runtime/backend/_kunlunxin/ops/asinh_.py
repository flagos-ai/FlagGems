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


# asinh(x) = sign(x) * log(|x| + sqrt(x^2 + 1))
@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def asinh__func(x):
    x_fp32 = x.to(tl.float32)
    abs_x = tl.abs(x_fp32)
    r = tl.where(
        abs_x > 1e16,
        abs_x + abs_x,
        abs_x + tl.sqrt(abs_x * abs_x + 1.0),
    )
    y = tl.log(r)
    result = tl.where(x_fp32 < 0.0, -y, y)
    return result.to(x.dtype)


def asinh_(A):
    logger.debug("GEMS_KUNLUNXIN ASINH_")
    asinh__func(A, out0=A)
    return A
