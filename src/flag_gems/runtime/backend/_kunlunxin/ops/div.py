import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

# Generic pointwise codegen -- used only on the complex path because the
# kunlunxin codegen has no register_complex hook (Triton 3.0 cannot mangle
# complex64 directly, so complex tensors must be split into real/imag and
# fed through a 4-in/2-out cross kernel).
from flag_gems.utils.pointwise_dynamic import ComplexMode
from flag_gems.utils.pointwise_dynamic import (
    pointwise_dynamic as generic_pointwise_dynamic,
)

# Kunlunxin-tuned pointwise codegen for real dtypes (preserves P800 tile/
# buffer/unroll tuning).
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


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func_tensor_scalar(x, y):
    return x / y


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func_scalar_tensor(x, y):
    return x / y


# ---------------------------------------------------------------------------
# Complex true_divide path.
#
# The kunlunxin pointwise codegen does NOT expose register_complex, and Triton
# 3.0's mangle_type does not know about complex64 -- so complex tensors must
# be routed through the GENERIC codegen, which splits a complex op into a
# 4-input / 2-output real cross-kernel using Smith's method.
# ---------------------------------------------------------------------------
@generic_pointwise_dynamic(
    is_tensor=[True, True, True, True],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, 3, "INT_TO_FLOAT"),
        (0, 1, 2, 3, "INT_TO_FLOAT"),
    ],
)
@triton.jit
def _div_complex_cross_kernel(ar, ai, br, bi):
    # Smith's method -- avoids fp overflow when |c| and |d| differ greatly.
    abs_br = tl.abs(br)
    abs_bi = tl.abs(bi)
    use_br = abs_br >= abs_bi

    r1 = tl.where(br == 0, 0.0, bi / br)
    d1 = br + bi * r1
    real1 = (ar + ai * r1) / d1
    imag1 = (ai - ar * r1) / d1

    r2 = tl.where(bi == 0, 0.0, br / bi)
    d2 = bi + br * r2
    real2 = (ar * r2 + ai) / d2
    imag2 = (ai * r2 - ar) / d2

    return tl.where(use_br, real1, real2), tl.where(use_br, imag1, imag2)


@generic_pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def _true_div_func_complex(x, y):
    return x / y


@generic_pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")]
)
@triton.jit
def _true_div_func_ts_complex(x, y):
    return x / y


@generic_pointwise_dynamic(
    is_tensor=[False, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")]
)
@triton.jit
def _true_div_func_st_complex(x, y):
    return x / y


_true_div_func_complex.register_complex(
    mode=ComplexMode.CROSS,
    cross_kernel=_div_complex_cross_kernel,
)
_true_div_func_ts_complex.register_complex(
    mode=ComplexMode.CROSS,
    tensorize_scalars=True,
    fallback_target=_true_div_func_complex,
)
_true_div_func_st_complex.register_complex(
    mode=ComplexMode.CROSS,
    tensorize_scalars=True,
    fallback_target=_true_div_func_complex,
)


def _is_complex_arg(x):
    return (isinstance(x, torch.Tensor) and x.is_complex()) or isinstance(x, complex)


def _has_complex(*xs):
    return any(_is_complex_arg(x) for x in xs)


def true_divide(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE")
    if _has_complex(A, B):
        # Complex inputs must use the generic codegen (register_complex path).
        if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
            return _true_div_func_complex(A, B)
        elif isinstance(A, torch.Tensor):
            return _true_div_func_ts_complex(A, B)
        elif isinstance(B, torch.Tensor):
            return _true_div_func_st_complex(A, B)
        else:
            return torch.tensor(A / B)
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
    if _has_complex(A, B):
        if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
            return _true_div_func_complex(A, B, out0=out)
        elif isinstance(A, torch.Tensor):
            return _true_div_func_ts_complex(A, B, out0=out)
        elif isinstance(B, torch.Tensor):
            return _true_div_func_st_complex(A, B, out0=out)
        else:
            return torch.tensor(A / B) if out is None else out.fill_(A / B)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return true_div_func(A, B, out0=out)
    elif isinstance(A, torch.Tensor):
        return true_div_func_tensor_scalar(A, B, out0=out)
    elif isinstance(B, torch.Tensor):
        return true_div_func_scalar_tensor(A, B, out0=out)
    else:
        # Both scalar
        return torch.tensor(A / B) if out is None else out.fill_(A / B)


def true_divide_(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE_")
    if _has_complex(A, B):
        if isinstance(B, torch.Tensor):
            return _true_div_func_complex(A, B, out0=A)
        else:
            return _true_div_func_ts_complex(A, B, out0=A)
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
