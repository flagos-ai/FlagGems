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
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def square_func(x):
    return x * x


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def square_func_hp(x):
    # Compute the product in fp32 so it is correctly rounded back to the narrow
    # output dtype. The AMD GPU's native bf16 multiply is not correctly-rounded
    # against a fp32 reference, so bf16 x*x otherwise drifts by 1 ULP.
    xf = x.to(tl.float32)
    return xf * xf


def _kernel_for(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return square_func_hp
    return square_func


def square(A):
    logger.debug("GEMS SQUARE")
    return _kernel_for(A.dtype)(A)


def square_out(A, *, out=None):
    logger.debug("GEMS SQUARE_OUT")
    fn = _kernel_for(A.dtype)
    if out is None:
        return fn(A)
    fn(A, out0=out)
    return out


def square_(A):
    logger.debug("GEMS SQUARE_")
    _kernel_for(A.dtype)(A, out0=A)
    return A
