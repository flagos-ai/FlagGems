import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def logaddexp_func(x, y):
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    delta = tl.abs(x_fp32 - y_fp32)
    m = tl.maximum(x_fp32, y_fp32)
    return m + tl.log(1.0 + tl.exp(-delta))


def logaddexp(A, B):
    logger.debug("GEMS LOGADDEXP")
    return logaddexp_func(A, B)
