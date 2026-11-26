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


@triton.jit
def _fmod_fp32(x, y):
    """Exact fmod for fp32 operands (result takes the sign of ``x``).

    gcu300's libdevice ``fmod`` subtracts ``y * trunc(x / y)`` using the
    approximate fp32 quotient, so whenever the true quotient is an integer the
    rounded quotient lands one step below it and the remainder comes out a full
    ``|y|`` too large.  Compute the truncated quotient from the approximate
    division, then use the exact FMA residual to repair a quotient that landed
    one step away from ``trunc(x / y)``; the corrected residual is exact.
    """
    quotient = x / y
    truncated = tl.where(quotient >= 0.0, tl.floor(quotient), tl.ceil(quotient))
    remainder = tl.math.fma(-y, truncated, x)
    # A correct residual has the sign of x and is smaller than |y|.  Anything
    # else means the truncated quotient is off by one.
    misrounded = (remainder != 0.0) & (
        ((remainder < 0.0) != (x < 0.0)) | (tl.abs(remainder) >= tl.abs(y))
    )
    truncated = tl.where(misrounded & (remainder * y > 0.0), truncated + 1.0, truncated)
    truncated = tl.where(misrounded & (remainder * y < 0.0), truncated - 1.0, truncated)
    remainder = tl.math.fma(-y, truncated, x)
    # fmod(finite x, +-inf) is x; the fma above produces a NaN for it.
    return tl.where(
        (tl.abs(y) == float("inf")) & (tl.abs(x) != float("inf")), x, remainder
    )


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def fmod_func(x, y):
    # Convert to float32 for computation to avoid libdevice float16/bfloat16 issues
    dtype = x.dtype
    result = _fmod_fp32(x.to(tl.float32), y.to(tl.float32))
    return result.to(dtype)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def fmod_func_tensor_scalar(x, y):
    # Convert to float32 for computation to avoid libdevice float16/bfloat16 issues
    dtype = x.dtype
    result = _fmod_fp32(x.to(tl.float32), y.to(tl.float32))
    return result.to(dtype)


def fmod_tensor(A, B):
    logger.debug("GEMS_ENFLAME FMOD_TENSOR")
    return fmod_func(A, B)


def fmod_scalar(A, B):
    logger.debug("GEMS_ENFLAME FMOD_SCALAR")
    return fmod_func_tensor_scalar(A, B)


def fmod_tensor_(A, B):
    logger.debug("GEMS_ENFLAME FMOD_TENSOR_")
    return fmod_func(A, B, out0=A)


def fmod_scalar_(A, B):
    logger.debug("GEMS_ENFLAME FMOD_SCALAR_")
    return fmod_func_tensor_scalar(A, B, out0=A)
