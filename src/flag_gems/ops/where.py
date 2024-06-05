import triton
import triton.language as tl
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def where_func(condition, x, y):
    return tl.where(condition, x, y)


def where(condition, x, y):
    logging.debug("GEMS WHERE")
    O = where_func(condition, x, y)
    return O
