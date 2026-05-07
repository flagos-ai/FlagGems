import logging

import torch
import triton

from flag_gems.utils import pointwise_dynamic

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
    return torch.tensor(A << B)


def bitwise_left_shift_out(A, B, out):
    logger.debug("GEMS BITWISE_LEFT_SHIFT OUT")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return bitwise_left_shift_func(A, B, out0=out)
    elif isinstance(A, torch.Tensor):
        return bitwise_left_shift_func_scalar(A, B, out0=out)
    elif isinstance(B, torch.Tensor):
        return bitwise_left_shift_func_scalar(B, A, out0=out)
    return out.fill_(A << B)


def bitwise_left_shift_(A, B):
    logger.debug("GEMS BITWISE_LEFT_SHIFT_")
    if isinstance(B, torch.Tensor):
        return bitwise_left_shift_func(A, B, out0=A)
    return bitwise_left_shift_func_scalar(A, B, out0=A)
