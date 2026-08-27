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


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def celu_func(x, alpha):
    x_fp32 = x.to(tl.float32)
    alpha_fp32 = alpha.to(tl.float32)
    return tl.where(
        x_fp32 > 0, x_fp32, alpha_fp32 * (tl.exp(x_fp32 / alpha_fp32) - 1.0)
    ).to(x.dtype)


def celu(A, alpha=1.0):
    logger.debug("GEMS_ENFLAME CELU")
    return celu_func(A, alpha)


def celu_(A, alpha=1.0):
    logger.debug("GEMS_ENFLAME CELU_")
    celu_func(A, alpha, out0=A)
    return A
