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


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_and_func(x, y):
    return x & y


def bitwise_and_tensor(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_AND")
    if A.dtype == torch.int64:
        A = A.to(torch.int32)
    if B.dtype == torch.int64:
        B = B.to(torch.int32)
    return bitwise_and_func(A, B)


def bitwise_and_tensor_(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_AND_")
    if A.dtype == torch.int64:
        A = A.to(torch.int32)
    if B.dtype == torch.int64:
        B = B.to(torch.int32)
    return bitwise_and_func(A, B, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_and_func_scalar(x, y):
    return x & y


def bitwise_and_scalar(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_AND_SCALAR")
    return bitwise_and_func_scalar(A, B)


def bitwise_and_scalar_(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_AND_SCALAR_")
    return bitwise_and_func_scalar(A, B, out0=A)


def bitwise_and_scalar_tensor(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_AND_SCALAR_TENSOR")
    if isinstance(A, torch.Tensor) and A.dtype == torch.int64:
        A = A.to(torch.int32)
    if isinstance(B, torch.Tensor) and B.dtype == torch.int64:
        B = B.to(torch.int32)
    return bitwise_and_func_scalar(B, A)
