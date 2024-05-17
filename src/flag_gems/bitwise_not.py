import triton
import logging
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def bitwise_not_func(x):
    return ~x


def bitwise_not(A):
    logging.debug("GEMS BITWISE NOT")
    O = bitwise_not_func(A)
    return O
