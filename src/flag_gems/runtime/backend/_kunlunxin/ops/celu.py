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
    isCloseVectorization=True,  # TODO: Wait LLVM FIX
)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
# celu another way: max(0, x) + alpha * (exp(min(0, x) / alpha) - 1), getting smaller instrs.
def celu_forward_kernel(x, alpha):
    inv_alpha = 1.0 / alpha

    pos_part = tl.maximum(0.0, x)

    neg_part_input = x - pos_part

    return pos_part + alpha * (tl.exp(neg_part_input * inv_alpha) - 1.0)


def celu(A, alpha=1.0):
    logger.debug("GEMS_KUNLUNXIN CELU")
    return celu_forward_kernel(A, alpha)


def celu_(A, alpha=1.0):
    logger.debug("GEMS_KUNLUNXIN CELU_")
    return celu_forward_kernel(A, alpha, out0=A)
