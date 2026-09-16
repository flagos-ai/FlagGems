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


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")]
)
@triton.jit
def gt_func(x, y, inplace):
    return x > y


def gt(A, B):
    logger.debug("GEMS_CAMBRICON GT")
    return gt_func(A, B, False)


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")]
)
@triton.jit
def gt_func_scalar(x, y, inplace):
    return x > y


def gt_scalar(A, B):
    logger.debug("GEMS_CAMBRICON GT_SCALAR")
    return gt_func_scalar(A, B, False)


def gt_tensor_(A, B):
    logger.debug("GEMS_CAMBRICON GT_ TENSOR")
    return gt_func(A, B, True, out0=A)


def gt_scalar_(A, B):
    logger.debug("GEMS_CAMBRICON GT_ SCALAR")
    return gt_func_scalar(A, B, True, out0=A)
