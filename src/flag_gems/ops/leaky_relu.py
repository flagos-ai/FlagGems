import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], is_tensor=[True, False])
@triton.jit
def leaky_relu_func(x, negative_slope):
    x_f = x.to(tl.float32)
    return tl.where(x_f >= 0, x_f, x_f * negative_slope)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], is_tensor=[True, False])
@triton.jit
def leaky_relu_backward_func(grad_output, x, negative_slope):
    x_f = x.to(tl.float32)
    grad_f = grad_output.to(tl.float32)
    return tl.where(x_f >= 0, grad_f, grad_f * negative_slope)


def leaky_relu(A, negative_slope: float = 0.01):
    logger.debug("GEMS LEAKY_RELU")
    return leaky_relu_func(A, negative_slope)


def leaky_relu_(A, negative_slope: float = 0.01):
    logger.debug("GEMS LEAKY_RELU_")
    leaky_relu_func(A, negative_slope, out0=A)
    return A


def leaky_relu_backward(grad_output, A, negative_slope: float = 0.01):
    logger.debug("GEMS LEAKY_RELU BACKWARD")
    return leaky_relu_backward_func(grad_output, A, negative_slope)
