import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def reciprocal_func(x):
    return 1.0 / x.to(tl.float32)


def reciprocal(A):
    logging.debug("GEMS RECIPROCAL")
    return reciprocal_func(A)
