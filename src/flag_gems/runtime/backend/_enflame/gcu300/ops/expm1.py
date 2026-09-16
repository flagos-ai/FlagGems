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


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def expm1_func(x):
    return tl.exp(x.to(tl.float32)) - 1.0


def expm1(A):
    logger.debug("GEMS_ENFLAME EXPM1")
    return expm1_func(A)


def expm1_(A):
    logger.debug("GEMS_ENFLAME EXPM1_")
    return expm1_func(A, out0=A)


def expm1_out(A, out):
    logger.debug("GEMS_ENFLAME EXPM1_OUT")
    return expm1_func(A, out0=out)
