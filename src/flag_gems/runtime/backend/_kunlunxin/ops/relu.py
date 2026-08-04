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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
# relu another way: maximum(x, 0)
# tl.maximum(x, 0) to one max_instr，but tl.where two instr compare and select
def relu_forward(x):
    return tl.maximum(x, 0)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def relu_backward(x, dy):
    return tl.where(x > 0, dy, 0)


def relu(self):
    logger.debug("GEMS_KUNLUNXIN RELU")
    output = relu_forward(self)
    return output


def relu_(A):
    logger.debug("GEMS_KUNLUNXIN RELU_")
    out = relu_forward(A, out0=A)
    return out
