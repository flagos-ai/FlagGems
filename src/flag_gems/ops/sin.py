import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[[0, "INT_TO_FLOAT"]])
@triton.jit
def sin_func(x):
    return tl.sin(x.to(tl.float32))


def sin(A):
    logging.debug("GEMS SIN")
    return sin_func(A)
