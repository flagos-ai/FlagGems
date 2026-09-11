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

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)
_pow = tl_extra_shim.pow


@pointwise_dynamic(promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func(x, exponent):
    if x.type.element_ty == tl.bfloat16:
        return _pow(x.to(tl.float32), exponent.to(tl.float32))
    elif x.type.element_ty == tl.float16:
        return _pow(x.to(tl.float32), exponent.to(tl.float32))
    else:
        return _pow(x.to(tl.float64), exponent.to(tl.float64))


def pow_tensor_tensor(A, exponent):
    logger.debug("GEMS_HYGON POW_TENSOR_TENSOR")
    return pow_func(A, exponent)


def pow_tensor_tensor_(A, exponent):
    logger.debug("GEMS_HYGON POW_TENSOR_TENSOR_")
    return pow_func(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_tensor_scalar(x, exponent):
    if x.type.element_ty == tl.bfloat16:
        return _pow(x.to(tl.float32), exponent.to(tl.float32))
    elif x.type.element_ty == tl.float16:
        return _pow(x.to(tl.float32), exponent.to(tl.float32))
    else:
        return _pow(x.to(tl.float64), exponent.to(tl.float64))


def pow_tensor_scalar(A, exponent):
    logger.debug("GEMS_HYGON POW_TENSOR_SCALAR")
    return pow_func_tensor_scalar(A, exponent)


def pow_tensor_scalar_(A, exponent):
    logger.debug("GEMS_HYGON POW_TENSOR_SCALAR_")
    return pow_func_tensor_scalar(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_scalar_tensor(x, exponent):
    if exponent.type.element_ty == tl.bfloat16:
        return _pow(x.to(tl.float32), exponent.to(tl.float32))
    elif exponent.type.element_ty == tl.float16:
        return _pow(x.to(tl.float32), exponent.to(tl.float32))
    else:
        return _pow(x.to(tl.float64), exponent.to(tl.float64))


def pow_scalar(A, exponent):
    logger.debug("GEMS_HYGON POW_SCALAR")
    return pow_func_scalar_tensor(A, exponent)
