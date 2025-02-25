import logging

import triton

from ..utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def ne_func(x, y):
    return x != y


def ne(A, B):
    logging.debug("GEMS_CAMBRICON NE")
    return ne_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def ne_func_scalar(x, y):
    return x != y


def ne_scalar(A, B):
    logging.debug("GEMS_CAMBRICON NE SCALAR")
    return ne_func_scalar(A, B)
