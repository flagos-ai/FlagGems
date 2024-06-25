import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[[0, "ALWAYS_BOOL"]])
@triton.jit
def isinf_func(x):
    return tl.math.isinf(x.to(tl.float32))


def isinf(A):
    logging.debug("GEMS ISINF")
    return isinf_func(A)
