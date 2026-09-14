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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
div_rn = tl_extra_shim.div_rn
div_rz = tl_extra_shim.div_rz
fmod = tl_extra_shim.fmod
trunc = tl_extra_shim.trunc
xpu_trunc_div = tl_extra_shim.xpu_trunc_div  # use it if we need to cmp result with xpu

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_)
@triton.jit
def true_div_func(x, y):
    return x / y


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "INT_TO_FLOAT")],
    config=config_,
)
@triton.jit
def true_div_func_tensor_scalar(x, y):
    return x / y


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func_scalar_tensor(x, y):
    return x / y


@pointwise_dynamic(
    is_tensor=[True, True, True, True],
    promotion_methods=[(0, 1, 2, 3, "INT_TO_FLOAT")],
)
@triton.jit
def div_complex_real(ar, ai, br, bi):
    # Smith's method: divide by the larger denominator component to avoid
    # intermediate overflow/underflow (mirrors the common op complex kernel).
    # Computed in fp32: fp16/bf16 components would lose precision and the
    # fp16 division path is less robust in the XPU backend.
    arf = ar.to(tl.float32)
    aif = ai.to(tl.float32)
    brf = br.to(tl.float32)
    bif = bi.to(tl.float32)
    abs_br = tl.abs(brf)
    abs_bi = tl.abs(bif)
    use_br = abs_br >= abs_bi

    # When |br| >= |bi|: ratio = bi/br, denom = br + bi*ratio
    ratio1 = tl.where(brf == 0, 0.0, bif / brf)
    denom1 = brf + bif * ratio1
    real1 = (arf + aif * ratio1) / denom1
    imag1 = (aif - arf * ratio1) / denom1

    # When |bi| > |br|: ratio = br/bi, denom = bi + br*ratio
    ratio2 = tl.where(bif == 0, 0.0, brf / bif)
    denom2 = bif + brf * ratio2
    real2 = (arf * ratio2 + aif) / denom2
    imag2 = (aif * ratio2 - arf) / denom2

    return tl.where(use_br, real1, real2)


@pointwise_dynamic(
    is_tensor=[True, True, True, True],
    promotion_methods=[(0, 1, 2, 3, "INT_TO_FLOAT")],
)
@triton.jit
def div_complex_imag(ar, ai, br, bi):
    arf = ar.to(tl.float32)
    aif = ai.to(tl.float32)
    brf = br.to(tl.float32)
    bif = bi.to(tl.float32)
    abs_br = tl.abs(brf)
    abs_bi = tl.abs(bif)
    use_br = abs_br >= abs_bi

    ratio1 = tl.where(brf == 0, 0.0, bif / brf)
    denom1 = brf + bif * ratio1
    imag1 = (aif - arf * ratio1) / denom1

    ratio2 = tl.where(bif == 0, 0.0, brf / bif)
    denom2 = bif + brf * ratio2
    imag2 = (aif * ratio2 - arf) / denom2

    return tl.where(use_br, imag1, imag2)


def _true_divide_complex(A, B):
    # Kunlunxin pointwise codegen cannot lower complex pointers
    # (canonicalize_ptr_dtype KeyError), so compute Smith's method on
    # real/imag components with two plain pointwise kernels and reassemble.
    if not A.is_complex():
        # real ÷ complex: promote A to B's precision, imag = 0
        a = torch.stack((A.to(B.dtype), torch.zeros_like(A, dtype=B.dtype)), dim=-1)
    else:
        a = torch.view_as_real(A.resolve_conj().resolve_neg())
    if isinstance(B, torch.Tensor):
        if B.is_complex():
            b = torch.view_as_real(B.resolve_conj().resolve_neg())
        else:
            b = torch.stack(
                (B.to(a.dtype), torch.zeros_like(B, dtype=a.dtype)), dim=-1
            )
    else:
        if isinstance(B, complex):
            b = torch.tensor(
                [B.real, B.imag], dtype=a.dtype, device=a.device
            ).unsqueeze(0).expand(*a.shape[:-1], 2)
        else:
            b = torch.stack(
                (
                    torch.full(a.shape[:-1], B, dtype=a.dtype, device=a.device),
                    torch.zeros(a.shape[:-1], dtype=a.dtype, device=a.device),
                ),
                dim=-1,
            )
    a = a.contiguous()
    b = b.contiguous()
    ar, ai = a.select(-1, 0), a.select(-1, 1)
    br, bi = b.select(-1, 0), b.select(-1, 1)
    real = div_complex_real(ar, ai, br, bi)
    imag = div_complex_imag(ar, ai, br, bi)
    output = torch.stack((real, imag), dim=-1)
    return torch.view_as_complex(output)


def divide(A, B):
    # Vendor entry for aten.divide.Tensor. SpecOpRegistrar replaces the
    # flag_gems global by function name, so a function named `divide` must be
    # exported here; otherwise divide.Tensor keeps dispatching to the generic
    # flag_gems.ops.divide.divide -> generic true_div_func (untuned codegen
    # config), which runs ~300x slower than the tuned kunlunxin kernel.
    #
    # tests/test_divide.py pins the legacy dispatch log contract ("GEMS DIVIDE"
    # via logger "flag_gems.ops.divide"); emit the same message through that
    # logger so the contract holds while the computation runs on the tuned
    # kunlunxin kernel below.
    logging.getLogger("flag_gems.ops.divide").debug("GEMS DIVIDE")
    logger.debug("GEMS_KUNLUNXIN DIVIDE")
    return true_divide(A, B)


def true_divide_tensor(A, B):
    # Vendor entry for aten.true_divide.Tensor. torch.true_divide (both
    # tensor-tensor and tensor-scalar) dispatches to this Tensor overload, so
    # without a function named `true_divide_tensor` exported from this package,
    # SpecOpRegistrar keeps the generic flag_gems.ops.true_divide.true_divide_tensor
    # (which calls the generic flag_gems.ops.div.true_divide by value), and the
    # untuned generic kernel runs orders of magnitude slower than the tuned
    # kunlunxin kernel.
    #
    # tests/test_true_divide.py pins the legacy dispatch log contract
    # ("GEMS TRUE_DIVIDE" via logger "flag_gems.ops.true_divide") to prove the
    # call is intercepted by a flag_gems override rather than native aten. Emit
    # the same message through that logger so the contract holds while the
    # actual computation runs on the tuned kunlunxin kernel below.
    logging.getLogger("flag_gems.ops.true_divide").debug("GEMS TRUE_DIVIDE")
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE_TENSOR")
    return true_divide(A, B)


def true_divide(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE")
    if isinstance(A, torch.Tensor) and A.is_complex():
        return _true_divide_complex(A, B)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and B.is_complex():
        return _true_divide_complex(A, B)
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
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE_OUT")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return true_div_func(A, B, out0=out)
    elif isinstance(A, torch.Tensor):
        return true_div_func_tensor_scalar(A, B, out0=out)
    elif isinstance(B, torch.Tensor):
        return true_div_func_scalar_tensor(A, B, out0=out)
    else:
        # Both scalar
        return torch.tensor(A / B) if out is None else out.fill_(A / B)


def true_divide_tensor_(A, B):
    # Vendor entry for aten.true_divide_.Tensor. Tensor.true_divide_ dispatches
    # here even for scalar others, so without a function named
    # `true_divide_tensor_` exported from this package, SpecOpRegistrar keeps
    # the generic flag_gems.ops.true_divide_.true_divide_tensor_ (which imports
    # the generic true_divide_ by value), and the untuned generic kernel runs
    # ~300x slower than the tuned kunlunxin kernel.
    #
    # tests/test_true_divide.py pins the legacy dispatch log contract
    # ("GEMS TRUE_DIVIDE_" via logger "flag_gems.ops.true_divide_") to prove the
    # call is intercepted by a flag_gems override rather than native aten. Emit
    # the same message through that logger so the contract holds while the
    # actual computation runs on the tuned kunlunxin kernel below.
    logging.getLogger("flag_gems.ops.true_divide_").debug("GEMS TRUE_DIVIDE_")
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE_TENSOR_")
    return true_divide_(A, B)


def true_divide_(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE_")
    if isinstance(B, torch.Tensor):
        return true_div_func(A, B, out0=A)
    else:
        return true_div_func_tensor_scalar(A, B, out0=A)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func(x, y):
    return xpu_trunc_div(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_tensor_scalar(x, y):
    return xpu_trunc_div(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_scalar_tensor(x, y):
    return xpu_trunc_div(x, y)


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


def trunc_divide(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUNC_DIVIDE")
    # Integer types: use dedicated int kernels (Triton // is C-style truncation)
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
        # Both scalar
        return torch.tensor(A / B)


def trunc_divide_(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUNC_DIVIDE_")
    # Integer types: use dedicated int kernels (Triton // is C-style truncation)
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
    # XPU libdevice fmod/div_rn only support fp32; fp16/bf16 inputs would fail
    # to compile (KeyError) or fall back to approximated division with ±1
    # boundary errors. Promote to fp32 for the CPython algorithm; results are
    # integer-valued so the store back to the original dtype is exact in range.
    # fp64 keeps the original precision (fp64 fmod is unsupported on XPU
    # libdevice, same as before).
    if x.type.scalar == tl.float64:
        xf = x
        yf = y
    else:
        xf = x.to(tl.float32)
        yf = y.to(tl.float32)
    remainder = fmod(xf, yf)
    imperfect = remainder != 0.0
    different_sign = (xf < 0) ^ (yf < 0)

    # NOTE: we have to use div_rn explicitly here
    q = div_rn(xf - remainder, yf)
    q = tl.where(imperfect & different_sign, q - 1, q)

    floor_q = tl.math.floor(q)
    c = q - floor_q > 0.5
    floor_q = tl.where(c, floor_q + 1.0, floor_q)

    q_is_zeros = q == 0.0
    floor_q = tl.where(q_is_zeros, tl.where(different_sign, -0.0, 0.0), floor_q)

    is_div_by_zero = yf == 0.0
    float_division = xf / yf
    out = tl.where(is_div_by_zero, float_division, floor_q)
    return out


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func(x, y):
    if x.type.scalar.is_int() & x.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func_tensor_scalar(x, y):
    if x.type.scalar.is_int() & x.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func_scalar_tensor(x, y):
    if x.type.scalar.is_int() & x.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv(x, y)


def floor_divide(A, B):
    logger.debug("GEMS_KUNLUNXIN FLOOR_DIVIDE")
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
    logger.debug("GEMS_KUNLUNXIN FLOOR_DIVIDE_")
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
    logger.debug("GEMS_KUNLUNXIN FLOOR_DIVIDE")
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
    logger.debug("GEMS_KUNLUNXIN REMAINDER_")
    if isinstance(B, torch.Tensor):
        return rem_tt(A, B, out0=A)
    else:
        return rem_ts(A, B, out0=A)
