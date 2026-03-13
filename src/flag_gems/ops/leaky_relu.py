import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def leaky_relu_kernel(x, negative_slope):
    x_fp32 = x.to(tl.float32)
    return tl.where(x_fp32 > 0, x_fp32, x_fp32 * negative_slope)


def leaky_relu(A, negative_slope=0.01):
    logger.debug("GEMS LEAKY_RELU")
    return leaky_relu_kernel(A, negative_slope)


def leaky_relu_(A, negative_slope=0.01):
    logger.debug("GEMS LEAKY_RELU_")
    return leaky_relu_kernel(A, negative_slope, out0=A)
