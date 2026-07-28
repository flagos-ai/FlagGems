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


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clamp_min_func(x, min_val):
    return tl.maximum(x, min_val)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clamp_min_scalar_func(x, min_val):
    return tl.maximum(x, min_val)


def clamp_min(A, min_val):
    logger.debug("GEMS_ENFLAME CLAMP_MIN")
    if isinstance(min_val, (int, float)):
        return clamp_min_scalar_func(A, min_val)
    return clamp_min_func(A, min_val)


def clamp_min_(A, min_val):
    logger.debug("GEMS_ENFLAME CLAMP_MIN_")
    if isinstance(min_val, (int, float)):
        clamp_min_scalar_func(A, min_val, out0=A)
    else:
        clamp_min_func(A, min_val, out0=A)
    return A
