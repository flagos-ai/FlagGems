import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def logaddexp_kernel(X, Y):
    X = X.to(tl.float32)
    Y = Y.to(tl.float32)

    max_val = tl.maximum(X, Y)
    min_val = tl.minimum(X, Y)
    return max_val + tl.log(1.0 + tl.exp(min_val - max_val))


def logaddexp(X, Y):
    logger.debug("GEMS LOGADDEXP")
    return logaddexp_kernel(X, Y)