import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[[0, 1, "ALWAYS_BOOL"]])
@triton.jit
def gt_func(x, y):
    return x.to(tl.float32) > y


def gt(A, B):
    logging.debug("GEMS GT")
    return gt_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[[0, 1, "ALWAYS_BOOL"]])
@triton.jit
def gt_func_scalar(x, y):
    return x.to(tl.float32) > y


def gt_scalar(A, B):
    logging.debug("GEMS GT SCALAR")
    return gt_func_scalar(A, B)
