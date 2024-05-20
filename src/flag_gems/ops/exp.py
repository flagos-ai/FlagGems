import triton
import triton.language as tl
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def exp_func(x):
    return tl.exp(x.to(tl.float32))


def exp(A):
    logging.debug("GEMS EXP")
    O = exp_func(A)
    return O
