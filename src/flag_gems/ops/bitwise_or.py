import triton
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def bitwise_or_func(x, y):
    return x | y


def bitwise_or_tensor(A, B):
    logging.debug("GEMS BITWISE OR")
    O = bitwise_or_func(A, B)
    return O
