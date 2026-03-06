import logging

import torch
import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_left_shift_func(x, y):
    return x << y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_left_shift_func_scalar(x, y):
    return x << y


def bitwise_left_shift(A, B):
    logger.debug("GEMS BITWISE_LEFT_SHIFT")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return bitwise_left_shift_func(A, B)
    elif isinstance(A, torch.Tensor):
        return bitwise_left_shift_func_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return bitwise_left_shift_func_scalar(B, A)
    else:
        # Both scalar
        return torch.tensor(A * B)


def bitwise_left_shift_(A, B):
    logger.debug("GEMS BITWISE_LEFT_SHIFT")
    if isinstance(B, torch.Tensor):
        return bitwise_left_shift_func(A, B, out0=A)
    else:
        return bitwise_left_shift_func_scalar(A, B, out0=A)
