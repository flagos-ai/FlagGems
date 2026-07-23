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


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def reciprocal_func(x, inplace):
    return 1.0 / x.to(tl.float32)


def reciprocal(A):
    logger.debug("GEMS_CAMBRICON RECIPROCAL")
    return reciprocal_func(A, False)


def reciprocal_(A):
    logger.debug("GEMS_CAMBRICON RECIPROCAL_")
    return reciprocal_func(A, True, out0=A)
