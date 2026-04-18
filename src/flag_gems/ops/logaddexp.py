import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def logaddexp_func(x, y):
    # Numerically stable: log(exp(x) + exp(y)) = max + log(1 + exp(min - max))
    x_f = x.to(tl.float32)
    y_f = y.to(tl.float32)
    max_val = tl.maximum(x_f, y_f)
    min_val = tl.minimum(x_f, y_f)
    # Handle -inf: if max is -inf, result is -inf
    diff = min_val - max_val  # always <= 0
    return max_val + tl.log1p(tl.exp(diff))


def logaddexp(A, B):
    logger.debug("GEMS LOGADDEXP")
    return logaddexp_func(A, B)
