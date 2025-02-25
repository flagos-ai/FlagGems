import logging

import triton

from ..utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def ge_func(x, y):
    return x >= y


def ge(A, B):
    logging.debug("GEMS_CAMBRICON GE")
    return ge_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def ge_func_scalar(x, y):
    return x >= y


def ge_scalar(A, B):
    logging.debug("GEMS_CAMBRICON GE SCALAR")
    return ge_func_scalar(A, B)
