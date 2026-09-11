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

from ..utils.pointwise_dynamic import pointwise_dynamic

try:
    from triton.language.extra.cuda.libdevice import pow as _pow
except ImportError:
    try:
        from triton.language.math import pow as _pow
    except ImportError:
        from triton.language.libdevice import pow as _pow

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_fast(x, exponent):
    x_f = x.to(tl.float32)
    e_f = exponent.to(tl.float32)
    abs_x = tl.abs(x_f)
    result = tl.math.exp2(e_f * tl.math.log2(abs_x))
    is_neg = x_f < 0.0
    e_int = e_f.to(tl.int32)
    is_int = e_f == e_int.to(tl.float32)
    is_odd = (e_int & 1) != 0
    result = tl.where(is_neg & is_int & is_odd, -result, result)
    result = tl.where(is_neg & ~is_int, float("nan"), result)
    return result


@pointwise_dynamic(promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_safe(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_tensor_tensor(A, exponent):
    logger.debug("GEMS_ENFLAME POW_TENSOR_TENSOR")
    if A.dtype == torch.float32:
        return pow_func_safe(A, exponent)
    return pow_func_fast(A, exponent)


def pow_tensor_tensor_(A, exponent):
    logger.debug("GEMS_ENFLAME POW_TENSOR_TENSOR_")
    if A.dtype == torch.float32:
        return pow_func_safe(A, exponent, out0=A)
    return pow_func_fast(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_tensor_scalar(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_tensor_scalar(A, exponent):
    logger.debug("GEMS_ENFLAME POW_TENSOR_SCALAR")
    return pow_func_tensor_scalar(A, exponent)


def pow_tensor_scalar_(A, exponent):
    logger.debug("GEMS_ENFLAME POW_TENSOR_SCALAR_")
    return pow_func_tensor_scalar(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_scalar_tensor(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_scalar(A, exponent):
    logger.debug("GEMS_ENFLAME POW_SCALAR")
    return pow_func_scalar_tensor(A, exponent)
