import logging

import torch
import triton

from ..utils import pointwise_dynamic


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
    return triton.div_rz(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_tensor_scalar(x, y):
    return triton.div_rz(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_scalar_tensor(x, y):
    return triton.div_rz(x, y)


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
    return x // y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func_tensor_scalar(x, y):
    return x // y


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func_scalar_tensor(x, y):
    return x // y


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


def div(A, B, rounding_mode=None):
    if rounding_mode is None:
        return true_divide(A, B)
    elif rounding_mode == "trunc":
        return trunc_divide(A, B)
    elif rounding_mode == "floor":
        return floor_divide(A, B)
    else:
        msg = f"div expected rounding_mode to be one of None, 'trunc', or 'floor' but found {rounding_mode}."
        raise ValueError(msg)
