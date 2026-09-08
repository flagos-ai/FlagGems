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

from flag_gems.utils import pointwise_dynamic, tl_extra_shim
from flag_gems.utils.pointwise_dynamic import ComplexMode

logger = logging.getLogger(__name__)
fmod = tl_extra_shim.fmod
trunc = tl_extra_shim.trunc


@pointwise_dynamic(
    is_tensor=[True, True, True, True],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, 3, "INT_TO_FLOAT"),
        (0, 1, 2, 3, "INT_TO_FLOAT"),
    ],
)
@triton.jit
def div_complex_kernel_hygon_fp64(ar, ai, br, bi):
    # Smith's method avoids overflow by dividing by the larger component.
    ar = ar.to(tl.float64)
    ai = ai.to(tl.float64)
    br = br.to(tl.float64)
    bi = bi.to(tl.float64)

    abs_br = tl.abs(br)
    abs_bi = tl.abs(bi)
    use_br = abs_br >= abs_bi

    safe_br = tl.where(br == 0, 1.0, br)
    safe_bi = tl.where(bi == 0, 1.0, bi)

    ratio1 = tl.where(br == 0, 0.0, bi / safe_br)
    denom1 = br + bi * ratio1
    selected_denom1 = tl.where(use_br, denom1, 1.0)
    real1 = (ar + ai * ratio1) / selected_denom1
    imag1 = (ai - ar * ratio1) / selected_denom1

    ratio2 = tl.where(bi == 0, 0.0, br / safe_bi)
    denom2 = bi + br * ratio2
    selected_denom2 = tl.where(use_br, 1.0, denom2)
    real2 = (ar * ratio2 + ai) / selected_denom2
    imag2 = (ai * ratio2 - ar) / selected_denom2

    return tl.where(use_br, real1, real2), tl.where(use_br, imag1, imag2)


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func(x, y):
    return x / y


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def complex_real_div_fp64_func(x, y):
    is_zero = y == 0.0
    safe_y = tl.where(is_zero, 1.0, y)
    quotient = x.to(tl.float64) / safe_y.to(tl.float64)
    return tl.where(is_zero, float("nan"), quotient)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func_tensor_scalar(x, y):
    return x / y


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func_scalar_tensor(x, y):
    return x / y


true_div_func.register_complex(
    mode=ComplexMode.CROSS, cross_kernel=div_complex_kernel_hygon_fp64
)
true_div_func_tensor_scalar.register_complex(
    mode=ComplexMode.CROSS, tensorize_scalars=True, fallback_target=true_div_func
)
true_div_func_scalar_tensor.register_complex(
    mode=ComplexMode.CROSS, tensorize_scalars=True, fallback_target=true_div_func
)


def true_divide(A, B):
    logger.debug("GEMS_HYGON TRUE_DIVIDE")
    if (
        isinstance(A, torch.Tensor)
        and A.is_complex()
        and isinstance(B, torch.Tensor)
        and not B.is_complex()
    ):
        return torch.complex(
            complex_real_div_fp64_func(A.real, B),
            complex_real_div_fp64_func(A.imag, B),
        )
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return true_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return true_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return true_div_func_scalar_tensor(A, B)
    else:
        # Both scalar
        return torch.tensor(A / B)


def true_divide_out(A, B, out):
    logger.debug("GEMS_HYGON TRUE_DIVIDE OUT")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return true_div_func(A, B, out0=out)
    elif isinstance(A, torch.Tensor):
        return true_div_func_tensor_scalar(A, B, out0=out)
    elif isinstance(B, torch.Tensor):
        return true_div_func_scalar_tensor(A, B, out0=out)
    else:
        return torch.tensor(A / B) if out is None else out.fill_(A / B)


def true_divide_(A, B):
    logger.debug("GEMS_HYGON TRUE_DIVIDE_")
    if isinstance(B, torch.Tensor):
        return true_div_func(A, B, out0=A)
    else:
        return true_div_func_tensor_scalar(A, B, out0=A)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func(x, y):
    x = x.to(tl.float64)
    y = y.to(tl.float64)
    return trunc((x / y))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_tensor_scalar(x, y):
    return trunc((x / y))


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_scalar_tensor(x, y):
    return trunc((x / y))


def trunc_divide(A, B):
    logger.debug("GEMS_HYGON TRUNC_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return trunc_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return trunc_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return trunc_div_func_scalar_tensor(A, B)
    else:
        # Both scalar
        return torch.tensor(A / B)


def trunc_divide_(A, B):
    logger.debug("GEMS_HYGON TRUNC_DIVIDE_")
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
    is_zero = y == 0
    safe_y = tl.where(is_zero, 1, y)
    r = x % safe_y
    c1 = r != 0
    c2 = (x < 0) ^ (safe_y < 0)
    quotient = tl.where(c1 & c2, x // safe_y - 1, x // safe_y)
    zero_quotient = tl.where(x < 0, x - 2, tl.where(x == 0, 2, x + 1))
    return tl.where(is_zero, zero_quotient, quotient)


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
    # Hygon's libdevice fmod only accepts matching fp32/fp64 operands.
    # Pointwise scalar promotion can otherwise produce (fp16/bf16, fp32).
    orig_dtype = x.dtype
    x = x.to(tl.float32)
    y = y.to(tl.float32)

    # NOTE: fmod's sign is the same as the dividend
    remainder = fmod(x, y)
    imperfect = remainder != 0.0
    different_sign = (x < 0) ^ (y < 0)

    # NOTE: we have to use div_rn explicitly here
    # q = div_rn(x - remainder, y)
    q = (x - remainder) / y
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
        if x.type.scalar.is_int16():
            return _int_floordiv(x.to(tl.int32), y.to(tl.int32))
        elif x.type.scalar.is_uint16():
            return _int_floordiv(x.to(tl.uint32), y.to(tl.uint32))
        else:
            return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func_tensor_scalar(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        if x.type.scalar.is_int16():
            return _int_floordiv(x.to(tl.int32), y.to(tl.int32))
        elif x.type.scalar.is_uint16():
            return _int_floordiv(x.to(tl.uint32), y.to(tl.uint32))
        else:
            return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func_scalar_tensor(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        if x.type.scalar.is_int16():
            return _int_floordiv(x.to(tl.int32), y.to(tl.int32))
        elif x.type.scalar.is_uint16():
            return _int_floordiv(x.to(tl.uint32), y.to(tl.uint32))
        else:
            return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


def floor_divide(A, B):
    logger.debug("GEMS_HYGON FLOOR_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return floor_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return floor_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return floor_div_func_scalar_tensor(A, B)
    else:
        # Both scalar
        return torch.tensor(A // B)


def floor_divide_(A, B):
    logger.debug("GEMS_HYGON FLOOR_DIVIDE_")
    if isinstance(B, torch.Tensor):
        return floor_div_func(A, B, out0=A)
    else:
        return floor_div_func_tensor_scalar(A, B, out0=A)


def div_mode(A, B, rounding_mode=None):
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
    logger.debug("GEMS_HYGON FLOOR_DIVIDE")
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
    logger.debug("GEMS_HYGON REMAINDER_")
    if isinstance(B, torch.Tensor):
        return rem_tt(A, B, out0=A)
    else:
        return rem_ts(A, B, out0=A)
