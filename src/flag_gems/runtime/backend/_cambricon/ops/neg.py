import logging

import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def neg_func(x):
    return -x


def neg(A):
    logger.debug("GEMS_CAMBRICON NEG")
    return neg_func(A)
