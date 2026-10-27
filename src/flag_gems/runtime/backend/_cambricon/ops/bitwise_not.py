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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def bitwise_not_func(x, inplace):
    return ~x


def bitwise_not(A):
    logger.debug("GEMS_CAMBRICON BITWISE_NOT")
    return bitwise_not_func(A, False)


def bitwise_not_(A):
    logger.debug("GEMS_CAMBRICON BITWISE_NOT_")
    bitwise_not_func(A, True, out0=A)
    return A
