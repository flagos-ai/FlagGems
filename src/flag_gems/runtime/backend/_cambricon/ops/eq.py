import logging

import triton

from flag_gems.runtime import device

from ..utils.pointwise_dynamic import pointwise_dynamic

device = device.name


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def eq_func(x, y):
    return x == y


def eq(A, B):
    if A.device != B.device:
        if A.device.type == device:
            B = B.to(A.device)
        else:
            A = A.to(B.device)
    logging.debug("GEMS_CAMBRICON EQ")
    return eq_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def eq_func_scalar(x, y):
    return x == y


def eq_scalar(A, B):
    logging.debug("GEMS_CAMBRICON EQ SCALAR")
    return eq_func_scalar(A, B)
