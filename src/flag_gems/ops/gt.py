import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

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
    logger.debug("GEMS GT SCALAR")
    return gt_func_scalar(A, B)
