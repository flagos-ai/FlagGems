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

import torch
import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def bitwise_not_func(x):
    return ~x


def bitwise_not(A):
    logger.debug("GEMS_ENFLAME BITWISE_NOT")
    if A.dtype == torch.int64:
        A = A.to(torch.int32)
    return bitwise_not_func(A)


def bitwise_not_(A):
    logger.debug("GEMS_ENFLAME BITWISE_NOT_")
    if A.dtype == torch.int64:
        A = A.to(torch.int32)
    bitwise_not_func(A, out0=A)
    return A
