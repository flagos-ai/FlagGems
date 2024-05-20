import triton
import triton.language as tl
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def cos_func(x):
    return tl.cos(x.to(tl.float32))


def cos(A):
    logging.debug("GEMS COS")
    O = cos_func(A)
    return O
