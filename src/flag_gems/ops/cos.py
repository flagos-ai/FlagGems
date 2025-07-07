import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def cos_func(x):
    return tl.cos(x.to(tl.float32))


def cos(A):
    logger.debug("GEMS COS")
    return cos_func(A)


def cos_(A):
    logger.debug("GEMS COS_")
    cos_func(A, out0=A)
    return A
