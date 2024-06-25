import logging

import triton

from ..utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[[0, "DEFAULT"]])
@triton.jit
def bitwise_not_func(x):
    return ~x


def bitwise_not(A):
    logging.debug("GEMS BITWISE NOT")
    return bitwise_not_func(A)
