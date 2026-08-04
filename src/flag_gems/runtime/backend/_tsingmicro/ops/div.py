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

from flag_gems.utils.triton_lang_extension import div_rn, div_rz, fmod, trunc

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")]
)
@triton.jit
def true_div_func(x, y, inplace):
    return x / y


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")]
)
@triton.jit
def true_div_func_tensor_scalar(x, y, inplace):
    y = y.to(x.dtype)
    return x / y


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")]
)
@triton.jit
def true_div_func_scalar_tensor(x, y, inplace):
    x = x.to(y.dtype)
    return x / y


# Complex true-division. Triton cannot take complex pointers, so we split the
# operands into their real/imag channels via view_as_real and run the division
# in real space. The cross-term form (a/b = a*conj(b)/|b|^2) is computed with
# Smith's method to avoid overflow when one component dominates.
@pointwise_dynamic(
    is_tensor=[True, True, True, True, False],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, 3, "INT_TO_FLOAT"),
        (0, 1, 2, 3, "INT_TO_FLOAT"),
    ],
)
@triton.jit
def div_complex_kernel(ar, ai, br, bi, inplace):
    # Smith's method: divide by the larger component to avoid overflow.
    abs_br = tl.abs(br)
    abs_bi = tl.abs(bi)
    use_br = abs_br >= abs_bi

    # When |br| >= |bi|: ratio = bi/br, denom = br + bi*ratio
    ratio1 = tl.where(br == 0, 0.0, bi / br)
    denom1 = br + bi * ratio1
    real1 = (ar + ai * ratio1) / denom1
    imag1 = (ai - ar * ratio1) / denom1

    # When |bi| > |br|: ratio = br/bi, denom = bi + br*ratio
    ratio2 = tl.where(bi == 0, 0.0, br / bi)
    denom2 = bi + br * ratio2
    real2 = (ar * ratio2 + ai) / denom2
    imag2 = (ai * ratio2 - ar) / denom2

    real = tl.where(use_br, real1, real2)
    imag = tl.where(use_br, imag1, imag2)
    return real, imag


def _is_complex(x):
    return (isinstance(x, torch.Tensor) and x.is_complex()) or isinstance(x, complex)


def _true_divide_complex(A, B, out=None):
    # Promote both operands to a common complex tensor on the same device.
    result_dtype = torch.result_type(A, B)
    device = A.device if isinstance(A, torch.Tensor) else B.device

    def to_complex_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=result_dtype)
        return torch.tensor(x, dtype=result_dtype, device=device)

    A, B = to_complex_tensor(A), to_complex_tensor(B)
    A, B = torch.broadcast_tensors(A, B)

    Ar = torch.view_as_real(A)
    Br = torch.view_as_real(B)
    ar, ai = Ar[..., 0], Ar[..., 1]
    br, bi = Br[..., 0], Br[..., 1]

    common_dtype = torch.promote_types(ar.dtype, br.dtype)
    ar, ai = ar.to(common_dtype), ai.to(common_dtype)
    br, bi = br.to(common_dtype), bi.to(common_dtype)

    real, imag = div_complex_kernel(ar, ai, br, bi, False)
    res = torch.view_as_complex(torch.stack((real, imag), dim=-1).contiguous()).to(
        result_dtype
    )
    if out is not None:
        out.copy_(res)
        return out
    return res


def true_divide(A, B):
    logger.debug("GEMS_TSINGMICRO TRUE_DIVIDE")
    if _is_complex(A) or _is_complex(B):
        return _true_divide_complex(A, B)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return true_div_func(A, B, False)
    elif isinstance(A, torch.Tensor):
        return true_div_func_tensor_scalar(A, B, False)
    elif isinstance(B, torch.Tensor):
        return true_div_func_scalar_tensor(A, B, False)
    else:
        # Both scalar
        return torch.tensor(A / B)


def true_divide_out(A, B, out):
    logger.debug("GEMS_TSINGMICRO TRUE_DIVIDE OUT")
    if _is_complex(A) or _is_complex(B):
        return _true_divide_complex(A, B, out=out)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return true_div_func(A, B, False, out0=out)
    elif isinstance(A, torch.Tensor):
        return true_div_func_tensor_scalar(A, B, False, out0=out)
    elif isinstance(B, torch.Tensor):
        return true_div_func_scalar_tensor(A, B, False, out0=out)
    else:
        # Both scalar
        return torch.tensor(A / B) if out is None else out.fill_(A / B)


def true_divide_(A, B):
    logger.debug("GEMS_TSINGMICRO TRUE_DIVIDE_")
    if _is_complex(A) or _is_complex(B):
        return _true_divide_complex(A, B, out=A)
    if isinstance(B, torch.Tensor):
        return true_div_func(A, B, True, out0=A)
    else:
        return true_div_func_tensor_scalar(A, B, True, out0=A)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func(x, y, inplace):
    return trunc(div_rz(x, y))


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def trunc_div_func_tensor_scalar(x, y, inplace):
    return trunc(div_rz(x, tl.cast(y, x.dtype)))


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def trunc_div_func_scalar_tensor(x, y, inplace):
    return trunc(div_rz(tl.cast(x, y.dtype), y))


# Integer truncation division: Triton's // on integers is C-style (truncates toward zero)
@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_int_func(x, y, inplace):
    return x // y


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def trunc_div_int_func_tensor_scalar(x, y, inplace):
    return x // y


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def trunc_div_int_func_scalar_tensor(x, y, inplace):
    return x // y


def trunc_divide(A, B):
    logger.debug("GEMS_TSINGMICRO TRUNC_DIVIDE")
    # Integer types: use dedicated int kernels (Triton // is C-style truncation)
    if isinstance(A, torch.Tensor) and not A.is_floating_point():
        if isinstance(B, torch.Tensor):
            return trunc_div_int_func(A, B, False)
        else:
            return trunc_div_int_func_tensor_scalar(A, B, False)
    if isinstance(B, torch.Tensor) and not B.is_floating_point():
        return trunc_div_int_func_scalar_tensor(A, B, False)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return trunc_div_func(A, B, False)
    elif isinstance(A, torch.Tensor):
        return trunc_div_func_tensor_scalar(A, B, False)
    elif isinstance(B, torch.Tensor):
        return trunc_div_func_scalar_tensor(A, B, False)
    else:
        # Both scalar
        return torch.tensor(A / B)


def trunc_divide_(A, B):
    logger.debug("GEMS_TSINGMICRO TRUNC_DIVIDE_")
    # Integer types: use dedicated int kernels (Triton // is C-style truncation)
    if not A.is_floating_point():
        if isinstance(B, torch.Tensor):
            return trunc_div_int_func(A, B, True, out0=A)
        else:
            return trunc_div_int_func_tensor_scalar(A, B, True, out0=A)
    if isinstance(B, torch.Tensor):
        return trunc_div_func(A, B, True, out0=A)
    else:
        return trunc_div_func_tensor_scalar(A, B, True, out0=A)


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
    c3 = (x < 0) & (y == 0)
    c = c1 & c2
    if x.dtype == tl.int16:
        if y.dtype == tl.int16:
            return (x.to(tl.int32) // y.to(tl.int32)).cast(tl.int16) - c - c3
    return x // y - c - c3


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
    # NOTE: fmod's sign is the same as the dividend
    if y.type.scalar.is_int():
        y = y.to(tl.float32)
    remainder = fmod(x, y)
    imperfect = remainder != 0.0
    different_sign = (x < 0) ^ (y < 0)

    # NOTE: we have to use div_rn explicitly here
    q = div_rn(x - remainder, y)
    q = tl.where(imperfect & different_sign, q - 1, q)

    floor_q = tl.math.floor(q)
    c = q - floor_q > 0.5
    floor_q = tl.where(c, floor_q + 1.0, floor_q)

    q_is_zeros = q == 0.0
    floor_q = tl.where(q_is_zeros, tl.where(different_sign, -0.0, 0.0), floor_q)

    is_div_by_zero = y == 0.0
    float_division = x / y
    out = tl.where(is_div_by_zero, float_division, floor_q)
    return out


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func(x, y, inplace):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def floor_div_func_tensor_scalar(x, y, inplace):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def floor_div_func_scalar_tensor(x, y, inplace):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


def floor_divide(A, B):
    logger.debug("GEMS_TSINGMICRO FLOOR_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return floor_div_func(A, B, False)
    elif isinstance(A, torch.Tensor):
        return floor_div_func_tensor_scalar(A, B, False)
    elif isinstance(B, torch.Tensor):
        return floor_div_func_scalar_tensor(A, B, False)
    else:
        # Both scalar
        return torch.tensor(A // B)


def floor_divide_(A, B):
    logger.debug("GEMS_TSINGMICRO FLOOR_DIVIDE_")
    if isinstance(B, torch.Tensor):
        return floor_div_func(A, B, True, out0=A)
    else:
        return floor_div_func_tensor_scalar(A, B, True, out0=A)


def div_mode(A, B, rounding_mode=None):
    logger.debug("GEMS_TSINGMICRO DIV_MODE")
    if rounding_mode is None:
        return true_divide(A, B)
    elif rounding_mode == "trunc":
        return trunc_divide(A, B)
    elif rounding_mode == "floor":
        return floor_divide(A, B)
    else:
        msg = f"div expected rounding_mode to be one of None, 'trunc', or 'floor' but found {rounding_mode}."
        raise ValueError(msg)


def div_mode_(A, B, rounding_mode=None):
    logger.debug("GEMS_TSINGMICRO DIV_MODE_")
    if rounding_mode is None:
        return true_divide_(A, B)
    elif rounding_mode == "trunc":
        return trunc_divide_(A, B)
    elif rounding_mode == "floor":
        return floor_divide_(A, B)
    else:
        msg = f"div expected rounding_mode to be one of None, 'trunc', or 'floor' but found {rounding_mode}."
        raise ValueError(msg)


@triton.jit
def _remainder(x, y):
    r = x % y
    c1 = r != 0
    c2 = (x < 0) ^ (y < 0)
    return tl.where(c1 & c2, r + y, r)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_tt(x, y, inplace):
    return _remainder(x, y)


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def rem_ts(x, y, inplace):
    return _remainder(x, y)


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def rem_st(x, y, inplace):
    return _remainder(x, y)


def remainder(A, B):
    logger.debug("GEMS_TSINGMICRO REMAINDER")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return rem_tt(A, B, False)
    elif isinstance(A, torch.Tensor):
        return rem_ts(A, B, False)
    elif isinstance(B, torch.Tensor):
        return rem_st(A, B, False)
    else:
        # Both scalar
        return torch.tensor(A % B)


def remainder_(A, B):
    logger.debug("GEMS_TSINGMICRO REMAINDER_")
    if isinstance(B, torch.Tensor):
        return rem_tt(A, B, True, out0=A)
    else:
        return rem_ts(A, B, True, out0=A)
