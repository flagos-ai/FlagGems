import triton
import triton.language as tl
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def isinf_func(x):
    return tl.math.isinf(x.to(tl.float32))


def isinf(A):
    logging.debug("GEMS ISINF")
    O = isinf_func(A)
    return O
