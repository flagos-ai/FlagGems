import logging

import torch
import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def neg_func(x):
    return -x


def neg(A):
    logger.debug("GEMS NEG")
    return_dtype = A.dtype
    if A.dtype == torch.int64:
        A = A.to(torch.int32)
    return neg_func(A).to(return_dtype)


def neg_(A):
    logger.debug("GEMS NEG_")
    return_dtype = A.dtype
    if A.dtype == torch.int64:
        A = A.to(torch.int32)
    return neg_func(A, out0=A).to(return_dtype)
