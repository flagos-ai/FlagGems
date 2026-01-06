import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def asinh_func(x):
    # asinh(x) = log(x + sqrt(x^2 + 1))
    x_f32 = x.to(tl.float32)
    return tl.log(x_f32 + tl.sqrt(x_f32 * x_f32 + 1.0))


def asinh(A):
    logger.debug("GEMS ASINH")
    return asinh_func(A)
