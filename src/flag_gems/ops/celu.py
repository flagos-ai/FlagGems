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

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def celu_forward_kernel(x, alpha):
    return tl.where(
        x > 0,
        x,
        alpha * (tl.exp(x / alpha) - 1),
    )


def celu(A, alpha=1.0):
    logger.debug("GEMS CELU")
    return celu_forward_kernel(A, alpha)


def celu_(A, alpha=1.0):
    logger.debug("GEMS CELU_")
    return celu_forward_kernel(A, alpha, out0=A)
