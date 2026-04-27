import logging

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def gt_func(x, y):
    return x.to(tl.float32) > y


def gt(A, B):
    logger.debug("GEMS GT")
    return gt_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def gt_func_scalar(x, y):
    return x.to(tl.float32) > y


def gt_scalar(A, B):
    if A.dtype == torch.int64:
        A = A.to(torch.int32)
    if A.dtype == torch.float64:
        A = A.to(torch.float32)
    logger.debug("GEMS GT SCALAR")
    return gt_func_scalar(A, B)
