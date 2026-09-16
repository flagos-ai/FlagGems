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

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
_pow = tl_extra_shim.pow


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")]
)
@triton.jit
def pow_func(x, exponent, inplace):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_tensor_tensor(A, exponent):
    logger.debug("GEMS_CAMBRICON POW_TENSOR_TENSOR")
    return pow_func(A, exponent, False)


def pow_tensor_tensor_(A, exponent):
    logger.debug("GEMS_CAMBRICON POW_TENSOR_TENSOR_")
    return pow_func(A, exponent, True, out0=A)


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")]
)
@triton.jit
def pow_func_tensor_scalar(x, exponent, inplace):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_tensor_scalar_int(x, exponent):
    tmp = x.to(dtype=tl.float32)
    result = tl.full(x.shape, 1, tmp.dtype)
    n = tl.abs(exponent)
    if exponent == 0:
        result = result
    elif n == 1:
        result = tmp
    elif n == 2:
        result = tmp * tmp
    elif n == 3:
        result = tmp * tmp
        result = result * tmp
    elif n == 4:
        result = tmp * tmp
        result = result * result
    else:
        while n > 0:
            if n % 2 == 1:
                result = result * tmp
            tmp = tmp * tmp
            n = n // 2
    if exponent < 0:
        result = 1 / result
    return result


def pow_tensor_scalar(A, exponent):
    logger.debug("GEMS_CAMBRICON POW_TENSOR_SCALAR")
    if int(exponent) == exponent:
        return pow_func_tensor_scalar_int(A, int(exponent))
    return pow_func_tensor_scalar(A, exponent, False)


def pow_tensor_scalar_(A, exponent):
    logger.debug("GEMS_CAMBRICON POW_TENSOR_SCALAR_")
    return pow_func_tensor_scalar(A, exponent, True, out0=A)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_scalar_tensor(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_scalar(A, exponent):
    logger.debug("GEMS_CAMBRICON POW_SCALAR")
    return pow_func_scalar_tensor(A, exponent)
