import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[[0, "INT_TO_FLOAT"]])
@triton.jit
def exp_func(x):
    return tl.exp(x.to(tl.float32))


def exp(A):
    logging.debug("GEMS EXP")
    return exp_func(A)
