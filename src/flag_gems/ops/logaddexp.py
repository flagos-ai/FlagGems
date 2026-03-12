import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def logaddexp_kernel(x, y):
    x = x.to(tl.float32)
    y = y.to(tl.float32)
    m = tl.maximum(x, y)
    return m + tl.log(1.0 + tl.exp(-tl.abs(x - y)))


def logaddexp(A, B):
    logger.debug("GEMS LOGADDEXP")
    return logaddexp_kernel(A, B)
