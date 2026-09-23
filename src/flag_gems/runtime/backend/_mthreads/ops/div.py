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

"""
Specialized div for mthreads backend.

1. Complex / real division:
   The common op (src/flag_gems/ops/div.py) handles complex/real division via
   _call_complex_dispatch, which converts real operands to complex tensors using
   torch.complex(a, torch.zeros_like(a)). This triggers empty_kernel with complex64
   output, but mthreads Triton does not support complex64 in canonicalize_dtype
   (KeyError: 'complex64').
   Fix: decompose into real/imag parts directly.

2. Trunc division:
   The common op uses trunc(div_rn(x, y)), where div_rn is supposed to be IEEE 754
   RNE floating-point division. But mthreads Triton does not provide a hardware
   div_rn, so the fallback floor(x/y + 0.5) is used, which incorrectly rounds the
   quotient to the nearest integer before truncation.
   Fix: use trunc(x / y) directly, since standard float division is already RNE.

3. Floor division:
   The common op uses floor(div_rn(x, y)), which has the same div_rn fallback issue.
   Fix: use floor(x / y) directly.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.div import true_divide as _common_true_divide
from flag_gems.ops.div import true_divide_ as _common_true_divide_
from flag_gems.ops.div import true_divide_out as _common_true_divide_out
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.triton_lang_extension import fmod, trunc

logger = logging.getLogger(__name__)


def _is_complex_real_div(A, B):
    """Check if this is a complex / real division that needs special handling."""
    if isinstance(A, torch.Tensor) and A.is_complex():
        if isinstance(B, torch.Tensor) and not B.is_complex():
            return True
        if isinstance(B, (int, float)):
            return True
    return False


def _complex_div_real(A, B):
    """
    Compute complex / real directly: (a+bi)/c = a/c + (b/c)i

    Avoids creating complex tensors from real inputs, which would trigger
    empty_kernel with complex64 output (unsupported by mthreads Triton).
    """
    result_dtype = A.dtype
    real = torch.view_as_real(A)  # shape (..., 2)
    ar = real[..., 0]
    ai = real[..., 1]

    if isinstance(B, torch.Tensor):
        B = B.to(ar.dtype)
        out_r = ar / B
        out_i = ai / B
    else:
        out_r = ar / B
        out_i = ai / B

    out = torch.stack((out_r, out_i), dim=-1)
    return torch.view_as_complex(out.contiguous()).to(result_dtype)


def true_divide(A, B):
    logger.debug("GEMS_MTHREADS TRUE_DIVIDE")
    if _is_complex_real_div(A, B):
        return _complex_div_real(A, B)
    return _common_true_divide(A, B)


def true_divide_out(A, B, out):
    logger.debug("GEMS_MTHREADS TRUE_DIVIDE OUT")
    if _is_complex_real_div(A, B):
        result = _complex_div_real(A, B)
        out.copy_(result)
        return out
    return _common_true_divide_out(A, B, out)


def true_divide_(A, B):
    logger.debug("GEMS_MTHREADS TRUE_DIVIDE_")
    if _is_complex_real_div(A, B):
        result = _complex_div_real(A, B)
        A.copy_(result)
        return A
    return _common_true_divide_(A, B)


# ---- Trunc division specialization ----
# The common op uses trunc(div_rn(x, y)), but mthreads lacks hardware div_rn,
# causing the fallback floor(x/y + 0.5) to incorrectly round the quotient to
# the nearest integer. Standard float `/` is already IEEE 754 RNE, so we use
# trunc(x / y) directly.


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func(x, y):
    return trunc(x / y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_tensor_scalar(x, y):
    return trunc(x / tl.cast(y, x.dtype))


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_scalar_tensor(x, y):
    return trunc(tl.cast(x, y.dtype) / y)


# Integer truncation division: Triton's // on integers is C-style (truncates toward zero)
@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_int_func(x, y):
    return x // y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_int_func_tensor_scalar(x, y):
    return x // y


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_int_func_scalar_tensor(x, y):
    return x // y


# ---- Floor division specialization ----
# The common op's _float_floordiv uses div_rn internally, which has the same
# fallback issue on mthreads. We reimplement the logic replacing div_rn with
# standard division (which is correct since (x - remainder) is an exact multiple
# of y, making standard float division give the exact integer quotient).


@triton.jit
def _mthreads_float_floordiv(x, y):
    # Cast to float32 for precision (same as common op)
    orig_dtype = x.dtype
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)

    # fmod's sign is the same as the dividend
    remainder = fmod(x_fp32, y_fp32)
    imperfect = remainder != 0.0
    different_sign = (x_fp32 < 0) ^ (y_fp32 < 0)

    # Use standard division instead of div_rn. Since (x - remainder) is an exact
    # multiple of y, standard float division gives the correct integer quotient.
    q = (x_fp32 - remainder) / y_fp32
    q = tl.where(imperfect & different_sign, q - 1, q)

    floor_q = tl.math.floor(q)
    c = q - floor_q > 0.5
    floor_q = tl.where(c, floor_q + 1.0, floor_q)

    q_is_zeros = q == 0.0
    floor_q = tl.where(q_is_zeros, tl.where(different_sign, -0.0, 0.0), floor_q)

    is_div_by_zero = y_fp32 == 0.0
    float_division = x_fp32 / y_fp32
    out = tl.where(is_div_by_zero, float_division, floor_q)
    return out.to(orig_dtype)


@triton.jit
def _mthreads_int_floordiv(x, y):
    # Same as common op: fix Triton's // behavior for negative values
    r = x % y
    c1 = r != 0
    c2 = (x < 0) ^ (y < 0)
    return tl.where(c1 & c2, x // y - 1, x // y)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _mthreads_int_floordiv(x, y)
    else:
        return _mthreads_float_floordiv(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func_tensor_scalar(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _mthreads_int_floordiv(x, y)
    else:
        return _mthreads_float_floordiv(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func_scalar_tensor(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _mthreads_int_floordiv(x, y)
    else:
        return _mthreads_float_floordiv(x, y)


def trunc_divide(A, B):
    logger.debug("GEMS_MTHREADS TRUNC_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.is_floating_point() or B.is_floating_point():
            return trunc_div_func(A, B)
        else:
            return trunc_div_int_func(A, B)
    elif isinstance(A, torch.Tensor):
        if A.is_floating_point() or isinstance(B, float):
            return trunc_div_func_tensor_scalar(A, B)
        else:
            return trunc_div_int_func_tensor_scalar(A, B)
    else:
        return trunc_div_func_scalar_tensor(A, B)


def trunc_divide_(A, B):
    logger.debug("GEMS_MTHREADS TRUNC_DIVIDE_")
    if isinstance(B, torch.Tensor):
        if A.is_floating_point() or B.is_floating_point():
            return trunc_div_func(A, B, out0=A)
        else:
            return trunc_div_int_func(A, B, out0=A)
    else:
        if A.is_floating_point() or isinstance(B, float):
            return trunc_div_func_tensor_scalar(A, B, out0=A)
        else:
            return trunc_div_int_func_tensor_scalar(A, B, out0=A)


def floor_divide(A, B):
    logger.debug("GEMS_MTHREADS FLOOR_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return floor_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return floor_div_func_tensor_scalar(A, B)
    else:
        return floor_div_func_scalar_tensor(A, B)


def floor_divide_(A, B):
    logger.debug("GEMS_MTHREADS FLOOR_DIVIDE_")
    if isinstance(B, torch.Tensor):
        return floor_div_func(A, B, out0=A)
    else:
        return floor_div_func_tensor_scalar(A, B, out0=A)


def div_mode(A, B, rounding_mode=None):
    logger.debug("GEMS_MTHREADS DIV_MODE")
    if rounding_mode is None:
        return true_divide(A, B)
    elif rounding_mode == "trunc":
        return trunc_divide(A, B)
    elif rounding_mode == "floor":
        return floor_divide(A, B)
    else:
        raise ValueError(f"Invalid rounding_mode: {rounding_mode}")


def div_mode_(A, B, rounding_mode=None):
    logger.debug("GEMS_MTHREADS DIV_MODE_")
    if rounding_mode is None:
        return true_divide_(A, B)
    elif rounding_mode == "trunc":
        return trunc_divide_(A, B)
    elif rounding_mode == "floor":
        return floor_divide_(A, B)
    else:
        raise ValueError(f"Invalid rounding_mode: {rounding_mode}")
