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
import numbers

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim
from flag_gems.utils.triton_lang_extension import div_rn, trunc

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# libdevice fma on gcu400; used to get the exact division residual.
fma = tl_extra_shim.fma


@triton.jit
def _rne_div(x, y):
    # gcu400's f32 `/` and libdevice div_rn are only accurate to ~1-2 ulp,
    # which flips trunc() when the exact quotient is within a few ulp of an
    # integer (e.g. 1.5639446 / 0.3909861 is exactly 4.0 but divides to
    # 3.9999998).  One Newton step on the exact FMA residual restores IEEE
    # round-to-nearest, matching PyTorch's trunc_divide semantics.
    q = div_rn(x, y)
    # Skip the residual for non-finite quotients (x/0 -> +-inf, 0/0 -> nan):
    # fma(-inf, 0, x) is nan and would clobber the correct inf/nan result.
    refined = q + fma(-q, y, x) / y
    return tl.where(q - q == 0.0, refined, q)


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func(x, y):
    return x / y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func_tensor_scalar(x, y):
    return x / y


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func_scalar_tensor(x, y):
    return x / y


def true_divide(A, B):
    logger.debug("GEMS_ENFLAME TRUE_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return true_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return true_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return true_div_func_scalar_tensor(A, B)
    else:
        # Both scalar
        return torch.tensor(A / B)


def true_divide_(A, B):
    logger.debug("GEMS_ENFLAME TRUE_DIVIDE_")
    if isinstance(B, torch.Tensor):
        return true_div_func(A, B, out0=A)
    else:
        return true_div_func_tensor_scalar(A, B, out0=A)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func(x, y):
    return trunc(_rne_div(x, y))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_tensor_scalar(x, y):
    return trunc(_rne_div(x, tl.cast(y, x.dtype)))


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_scalar_tensor(x, y):
    return trunc(_rne_div(tl.cast(x, y.dtype), y))


# Integer truncation division: Triton's // on integers is C-style (truncates
# toward zero), matching aten.
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


def trunc_divide(A, B):
    logger.debug("GEMS_ENFLAME TRUNC_DIVIDE")
    # Integer types: dedicated int kernels (Triton // is C-style truncation)
    if isinstance(A, torch.Tensor) and not A.is_floating_point():
        if isinstance(B, torch.Tensor):
            return trunc_div_int_func(A, B)
        else:
            return trunc_div_int_func_tensor_scalar(A, B)
    if isinstance(B, torch.Tensor) and not B.is_floating_point():
        return trunc_div_int_func_scalar_tensor(A, B)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return trunc_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return trunc_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return trunc_div_func_scalar_tensor(A, B)
    else:
        return _trunc_divide_scalars(A, B)


def _trunc_divide_scalars(A, B):
    """Both operands are Python/NumPy scalars: mirror aten's wrapped-number
    semantics for ``div(..., rounding_mode="trunc")``."""
    if isinstance(A, bool) or isinstance(B, bool):
        raise NotImplementedError("\"div_trunc_cpu\" not implemented for 'Bool'")
    if isinstance(A, numbers.Integral) and isinstance(B, numbers.Integral):
        if B == 0:
            raise RuntimeError("ZeroDivisionError")
        quot = abs(int(A)) // abs(int(B))
        return torch.tensor(-quot if (A < 0) != (B < 0) else quot)
    a = torch.tensor(float(A), dtype=torch.get_default_dtype())
    b = torch.tensor(float(B), dtype=torch.get_default_dtype())
    return torch.trunc(a / b)


def trunc_divide_(A, B):
    logger.debug("GEMS_ENFLAME TRUNC_DIVIDE_")
    # Integer types: dedicated int kernels (Triton // is C-style truncation)
    if not A.is_floating_point():
        if isinstance(B, torch.Tensor):
            return trunc_div_int_func(A, B, out0=A)
        else:
            return trunc_div_int_func_tensor_scalar(A, B, out0=A)
    if isinstance(B, torch.Tensor):
        return trunc_div_func(A, B, out0=A)
    else:
        return trunc_div_func_tensor_scalar(A, B, out0=A)


@triton.jit
def _int_floordiv(x, y):
    # TODO: request Triton to add an integer remainder builtin
    # The semantic of Triton floordiv differs from Pytorch/Numpy
    # Triton floordiv equates to
    #     (x - np.fmod(x, y)) / y
    # whereas Pytorch floordiv is
    #     (x - np.remainder(x, y)) y
    # The results show a one off difference when
    #     C1) x and y have opposite signs
    # and C2) x is not multiples of y.
    # Apart from the above, there's an erroneous case x // 0 returns -1
    # whereas in Pytorch x // 0 returns -1 if x >=0 and -2 if x < 0
    # but this special case is coalesced into the c1 and c2 check so
    # there's extra handling.
    r = x % y
    c1 = r != 0
    c2 = (x < 0) ^ (y < 0)
    return tl.where(c1 & c2, x // y - 1, x // y)


@triton.jit
def _trunc_to_int(q):
    return tl.where(q >= 0.0, tl.math.floor(q), tl.math.ceil(q))


@triton.jit
def _exact_fmod(x, y):
    """IEEE-exact fmod built from ``fma``.

    gcu400's libdevice ``fmod`` is not exact once x/y is large: for
    ``x=1.47, y=1.39e-7`` it returns -7.7e-7 instead of 1.09e-7 (and even
    violates ``|r| < |y|``), which moves ``floor(x/y)`` by several units.
    ``fma`` is exact on this device (verified bit-for-bit against an fp64
    reference), so recover the integral quotient ``k`` one step at a time and
    keep ``r = x - k*y`` exact via fma.  Each reduction removes an approximate
    quotient <= a few ulp, so two passes plus a magnitude safety net leave
    ``|r| < |y|``.
    """
    k = _trunc_to_int(div_rn(x, y))
    r = fma(-k, y, x)
    k = _trunc_to_int(div_rn(r, y))
    r = fma(-k, y, r)
    k = _trunc_to_int(div_rn(r, y))
    r = fma(-k, y, r)

    ay = tl.abs(y)
    ar = tl.abs(r)
    adj = tl.where(ar >= ay, tl.where(r >= 0.0, ay, -ay), 0.0)
    r = r - adj
    ar = tl.abs(r)
    adj = tl.where(ar >= ay, tl.where(r >= 0.0, ay, -ay), 0.0)
    r = r - adj

    # fmod's result carries the dividend's sign.  When the true quotient sits
    # just below an integer, div_rn can round up and leave a tiny r of the
    # wrong sign; the canonical remainder is then r + sign(x)*|y|.
    wrong_sign = (r != 0.0) & ((r < 0.0) != (x < 0.0))
    return tl.where(wrong_sign, r + tl.where(x < 0.0, -ay, ay), r)


# TO be consistent with python, numpy and torch, we have to implement it in the
# following way.
# CPython
# https://github.com/python/cpython/blob/ace008c531dd685a30c1dd68f9b5ba35f20171cf/Objects/floatobject.c#L636
# numpy
# https://github.com/numpy/numpy/blob/a4ad142aa1282a77bbb05acd706cb57c9cc29846/numpy/_core/src/npymath/npy_math_internal.h.src#L532
# torch
# https://github.com/pytorch/pytorch/blob/d6d9183456cd07ca0b361a194b98c2fb196e7c36/c10/util/generic_math.h#L23
@triton.jit
def _float_floordiv(x, y):
    # gcu400: triton_gcu's fmod/div_rn only expose signatures for matching
    # fp32/fp64 operands.  A tensor+scalar pair can arrive as (fp16, fp32)
    # and used to raise KeyError in libdevice's arg_type_symbol_dict.
    # Cast both operands to fp32 first, then restore the input dtype.
    orig_dtype = x.dtype
    x = x.to(tl.float32)
    y = y.to(tl.float32)
    remainder = _exact_fmod(x, y)
    imperfect = remainder != 0.0
    different_sign = (x < 0) ^ (y < 0)

    # (x - remainder) / y is exact in real arithmetic; one Newton step on the
    # fma residual turns the ~1-2 ulp div_rn into the correctly rounded
    # division the reference uses.
    q = _rne_div(x - remainder, y)
    q = tl.where(imperfect & different_sign, q - 1, q)

    floor_q = tl.math.floor(q)
    c = q - floor_q > 0.5
    floor_q = tl.where(c, floor_q + 1.0, floor_q)

    q_is_zeros = q == 0.0
    floor_q = tl.where(q_is_zeros, tl.where(different_sign, -0.0, 0.0), floor_q)

    is_div_by_zero = y == 0.0
    float_division = x / y
    out = tl.where(is_div_by_zero, float_division, floor_q)
    return out.to(orig_dtype)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func_tensor_scalar(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func_scalar_tensor(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


def floor_divide(A, B):
    logger.debug("GEMS_ENFLAME FLOOR_DIVIDE")
    # gcu400: pointwise_dynamic promotes two tensors but leaves a scalar or a
    # 0-d tensor at its own dtype.  A mixed int/float pair then reaches the
    # kernel as (int32, float32), where _int_floordiv's `x // y` raises
    # "unexpected type fp32" and _float_floordiv's fmod has no mixed-type
    # signature.  Normalise with torch's promotion rules before dispatch.
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.dtype != B.dtype:
            dtype = torch.promote_types(A.dtype, B.dtype)
            if A.dtype != dtype:
                A = A.to(dtype)
            if B.dtype != dtype:
                B = B.to(dtype)
        return floor_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return floor_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        if isinstance(A, int) and B.dtype.is_floating_point:
            A = float(A)
        return floor_div_func_scalar_tensor(A, B)
    else:
        # Both scalar
        return torch.tensor(A // B)


def floor_divide_(A, B):
    logger.debug("GEMS_ENFLAME FLOOR_DIVIDE_")
    if isinstance(B, torch.Tensor):
        return floor_div_func(A, B, out0=A)
    else:
        return floor_div_func_tensor_scalar(A, B, out0=A)


def div_mode(A, B, rounding_mode=None):
    # Only trunc needs the gcu400-specific IEEE division above; floor/None
    # keep the generic implementation (this is also what the direct-call API
    # used before this override was exported).
    if rounding_mode == "trunc":
        return trunc_divide(A, B)
    from flag_gems.ops.div import div_mode as _generic_div_mode

    return _generic_div_mode(A, B, rounding_mode)


def div_mode_(A, B, rounding_mode=None):
    if rounding_mode == "trunc":
        return trunc_divide_(A, B)
    from flag_gems.ops.div import div_mode_ as _generic_div_mode_

    return _generic_div_mode_(A, B, rounding_mode)


@triton.jit
def _remainder(x, y):
    r = x % y
    c1 = r != 0
    c2 = (x < 0) ^ (y < 0)
    return tl.where(c1 & c2, r + y, r)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_tt(x, y):
    return _remainder(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_ts(x, y):
    return _remainder(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_st(x, y):
    return _remainder(x, y)


def remainder(A, B):
    logger.debug("GEMS_ENFLAME FLOOR_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return rem_tt(A, B)
    elif isinstance(A, torch.Tensor):
        return rem_ts(A, B)
    elif isinstance(B, torch.Tensor):
        return rem_st(A, B)
    else:
        # Both scalar
        return torch.tensor(A % B)


def remainder_(A, B):
    logger.debug("GEMS_ENFLAME REMAINDER_")
    if isinstance(B, torch.Tensor):
        return rem_tt(A, B, out0=A)
    else:
        return rem_ts(A, B, out0=A)
