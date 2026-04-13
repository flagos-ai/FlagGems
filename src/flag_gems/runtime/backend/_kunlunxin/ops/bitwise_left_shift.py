import logging

import torch
import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_left_shift_kernel(a, b):
    return a << b


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_left_shift_tensor_scalar(a, b):
    return a << b


def bitwise_left_shift(self, other, *, out=None):
    logger.debug("GEMS BITWISE_LEFT_SHIFT")
    if isinstance(self, torch.Tensor) and isinstance(other, torch.Tensor):
        return bitwise_left_shift_kernel(self, other, out=out)
    elif isinstance(self, torch.Tensor):
        return bitwise_left_shift_tensor_scalar(self, other, out=out)
    else:
        return bitwise_left_shift_kernel(self, other, out=out)
