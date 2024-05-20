import triton
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def bitwise_and_func(x, y):
    return x & y


def bitwise_and_tensor(A, B):
    logging.debug("GEMS BITWISE AND")
    O = bitwise_and_func(A, B)
    return O
