import logging

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def abs_func(x):
    return tl.abs(x)


def abs(A):
    logger.debug("GEMS ABS")
    return_type = A.dtype
    if A.dtype == torch.int64:
        A = A.to(torch.int32)
    return abs_func(A).to(return_type)


def abs_(A):
    logger.debug("GEMS ABS_")
    abs_func(A, out0=A)
    return_type = A.dtype
    if A.dtype == torch.int64:
        A = A.to(torch.int32)
    return A.to(return_type)
