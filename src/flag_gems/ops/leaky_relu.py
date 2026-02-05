import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def leaky_relu_func(x, negative_slope: tl.constexpr):
    x_f32 = x.to(tl.float32)
    # y = x si x>=0, si no y = negative_slope*x
    y_f32 = tl.where(x_f32 >= 0.0, x_f32, x_f32 * negative_slope)
    return y_f32.to(x.dtype)


def leaky_relu(A, negative_slope=0.01):
    logger.debug("GEMS LEAKY_RELU")
    return leaky_relu_func(A, negative_slope=negative_slope)