import triton
import triton.language as tl
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def sin_func(x):
    return tl.sin(x.to(tl.float32))


def sin(A):
    logging.debug("GEMS SIN")
    O = sin_func(A)
    return O
