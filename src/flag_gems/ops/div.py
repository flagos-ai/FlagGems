import logging

import torch
import triton
import triton.language as tl

from ..utils import pointwise_dynamic

try:
    from triton.language.extra.cuda.libdevice import div_rd, div_rz, trunc
except ImportError:
    try:
        from triton.language.math import div_rd, div_rz, trunc
    except ImportError:
        from triton.language.libdevice import div_rd, div_rz, trunc


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
    logging.debug("GEMS TRUE_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return true_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return true_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return true_div_func_scalar_tensor(A, B)
    else:
        # Both scalar
        return A / B


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func(x, y):
    return trunc(div_rz(x, y))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_tensor_scalar(x, y):
    return trunc(div_rz(x, y))


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_scalar_tensor(x, y):
    return trunc(div_rz(x, y))


def trunc_divide(A, B):
    logging.debug("GEMS TRUNC_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return trunc_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return trunc_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return trunc_div_func_scalar_tensor(A, B)
    else:
        # Both scalar
        return A / B


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func(x, y):
    return tl.math.floor(div_rd(x, y))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func_tensor_scalar(x, y):
    return tl.math.floor(div_rd(x, y))


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func_scalar_tensor(x, y):
    return tl.math.floor(div_rd(x, y))


def floor_divide(A, B):
    logging.debug("GEMS FLOOR_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return floor_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return floor_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return floor_div_func_scalar_tensor(A, B)
    else:
        # Both scalar
        return A // B


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
