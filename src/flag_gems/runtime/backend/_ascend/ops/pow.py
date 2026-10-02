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

_pow = tl_extra_shim.pow

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func(x, exponent):
    # Ascend's libdevice `pow` only supplies matching-dtype entries --
    # (f32, f32), (f16, f16), (bf16, bf16) -- so a 16-bit exponent has to be
    # widened along with the base; passing it through raised
    # "KeyError: (float32, float16)" in the codegen. A single `is_fp32()` test
    # covers that, and it also drops the former three-way `a or b or c`
    # condition, which the triton-ascend 3.2 frontend selects for the cann850
    # backend rejects with "chained boolean operators (A or B or C) are not
    # supported" (issue #5732).
    if tl.constexpr(exponent.dtype.is_fp32()):
        return _pow(x.to(tl.float32), exponent)
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_tensor_tensor(A, exponent):
    logger.debug("GEMS_ASCEND POW_TENSOR_TENSOR")
    return pow_func(A, exponent)


def pow_tensor_tensor_(A, exponent):
    logger.debug("GEMS_ASCEND POW_TENSOR_TENSOR_")
    out = pow_func(A, exponent)
    A.copy_(out)
    return A


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_tensor_scalar(x, exponent):
    if tl.constexpr(exponent.dtype.is_fp32()):
        return _pow(x.to(tl.float32), exponent)
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_tensor_scalar(A, exponent):
    logger.debug("GEMS_ASCEND POW_TENSOR_SCALAR")
    return pow_func_tensor_scalar(A, exponent)


def pow_tensor_scalar_(A, exponent):
    logger.debug("GEMS_ASCEND POW_TENSOR_SCALAR_")
    out = pow_func_tensor_scalar(A, exponent)
    A.copy_(out)
    return A


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_scalar_tensor(x, exponent):
    if tl.constexpr(exponent.dtype.is_fp32()):
        return _pow(x.to(tl.float32), exponent)
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_scalar(A, exponent):
    logger.debug("GEMS_ASCEND POW_SCALAR")
    return pow_func_scalar_tensor(A, exponent)
