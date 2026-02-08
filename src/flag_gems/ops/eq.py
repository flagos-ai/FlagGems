import logging

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import device
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)
device = device.name


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def eq_func(x, y):
    return x.to(tl.float32) == y.to(tl.float32)


def eq(A, B):
    if A.device != B.device:
        if A.device.type == device:
            B = B.to(A.device)
        else:
            A = A.to(B.device)
    logger.debug("GEMS EQ")
    return eq_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def eq_func_scalar(x, y):
    return x.to(tl.float32) == y.to(tl.float32)


def eq_scalar(A, B):
    logger.debug("GEMS EQ SCALAR")
    return eq_func_scalar(A, B)


def equal(x: torch.Tensor, y: torch.Tensor) -> bool:
    logger.debug("GEMS EQUAL")
    if x.shape != y.shape:
        return False
    eq_tensor = eq(x, y)
    return bool(flag_gems.all(eq_tensor).item())
