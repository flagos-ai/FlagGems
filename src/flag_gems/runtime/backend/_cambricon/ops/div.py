import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
div_rn = tl_extra_shim.div_rn
div_rz = tl_extra_shim.div_rz
fmod = tl_extra_shim.fmod
trunc = tl_extra_shim.trunc


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func(x, y):
    return x / y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func_tensor_scalar(x, y):
    y = y.to(x.dtype)
    return x / y


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func_scalar_tensor(x, y):
    x = x.to(y.dtype)
    return x / y


def true_divide(A, B):
    logger.debug("GEMS_CAMBRICON TRUE_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.shape != B.shape:
            A, B = torch.broadcast_tensors(A, B)
            A = A.clone()
            B = B.clone()
        return true_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return true_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return true_div_func_scalar_tensor(A, B)
    else:
        # Both scalar
        return torch.tensor(A / B)


def true_divide_(A, B):
    logger.debug("GEMS_CAMBRICON TRUE_DIVIDE_")
    if isinstance(B, torch.Tensor):
        if A.shape != B.shape:
            A, B = torch.broadcast_tensors(A, B)
            A = A.clone()
            B = B.clone()
        return true_div_func(A, B, out0=A)
    else:
        return true_div_func_tensor_scalar(A, B, out0=A)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func(x, y):
    return trunc(div_rn(x, y))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_tensor_scalar(x, y):
    return trunc(div_rn(x, y))


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_scalar_tensor(x, y):
    return trunc(div_rn(x, y))


def trunc_divide(A, B):
    logger.debug("GEMS_CAMBRICON TRUNC_DIVIDE")
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
    logger.debug("GEMS_CAMBRICON TRUNC_DIVIDE_")
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
    c3 = (x < 0) & (y == 0)
    c = c1 & c2
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
    logger.debug("GEMS_CAMBRICON FLOOR_DIVIDE")
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
    logger.debug("GEMS_CAMBRICON FLOOR_DIVIDE_")
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
    logger.debug("GEMS_CAMBRICON FLOOR_DIVIDE")
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
    logger.debug("GEMS_CAMBRICON REMAINDER_")
    if isinstance(B, torch.Tensor):
        return rem_tt(A, B, out0=A)
    else:
        return rem_ts(A, B, out0=A)
