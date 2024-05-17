import triton
import logging
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def neg_func(x):
    return -x


def neg(A):
    logging.debug("GEMS NEG")
    O = neg_func(A)
    return O
