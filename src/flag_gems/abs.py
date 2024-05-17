import triton
import triton.language as tl
import logging
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def abs_func(x):
    return tl.abs(x)


def abs(A):
    logging.debug("GEMS ABS")
    O = abs_func(A)
    return O
