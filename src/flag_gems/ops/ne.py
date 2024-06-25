import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[[0, 1, "ALWAYS_BOOL"]])
@triton.jit
def ne_func(x, y):
    return x.to(tl.float32) != y.to(tl.float32)


def ne(A, B):
    logging.debug("GEMS NE")
    return ne_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[[0, 1, "ALWAYS_BOOL"]])
@triton.jit
def ne_func_scalar(x, y):
    return x.to(tl.float32) != y.to(tl.float32)


def ne_scalar(A, B):
    logging.debug("GEMS NE SCALAR")
    return ne_func_scalar(A, B)
