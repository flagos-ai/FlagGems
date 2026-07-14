import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


# asinh(x) = sign(x) * log(|x| + sqrt(x^2 + 1))
# Uses float32 intermediate for numerical precision.
@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def asinh_inplace_func(x):
    x_fp32 = x.to(tl.float32)
    abs_x = tl.abs(x_fp32)
    y = tl.log(abs_x + tl.sqrt(abs_x * abs_x + 1.0))
    return tl.where(x_fp32 < 0.0, -y, y)


def asinh_(A):
    logger.debug("GEMS ASINH_")
    asinh_inplace_func(A, out0=A)
    return A
