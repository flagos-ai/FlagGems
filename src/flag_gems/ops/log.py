import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def log_func(x):
    return tl.log(x.to(tl.float32))


def log(A):
    logger.debug("GEMS LOG")
    return log_func(A)
