import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[[0, 1, "ALWAYS_BOOL"]])
@triton.jit
def eq_func(x, y):
    return x.to(tl.float32) == y.to(tl.float32)


def eq(A, B):
    logging.debug("GEMS EQ")
    return eq_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[[0, 1, "ALWAYS_BOOL"]])
@triton.jit
def eq_func_scalar(x, y):
    return x.to(tl.float32) == y.to(tl.float32)


def eq_scalar(A, B):
    logging.debug("GEMS EQ SCALAR")
    return eq_func_scalar(A, B)
