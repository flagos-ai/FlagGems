import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


# LeakyReLU forward: f(x) = x if x > 0, else negative_slope * x
# Uses float32 intermediate for numerical precision.
@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def leaky_relu_forward_kernel(x, negative_slope):
    x_fp32 = x.to(tl.float32)
    return tl.where(x_fp32 > 0, x_fp32, x_fp32 * negative_slope)


# LeakyReLU backward: grad_input = grad if self > 0, else grad * negative_slope
@pointwise_dynamic(is_tensor=[True, False, True], promotion_methods=[(0, 2, "DEFAULT")])
@triton.jit
def leaky_relu_backward_kernel(grad_output, negative_slope, self):
    grad_fp32 = grad_output.to(tl.float32)
    return tl.where(self > 0, grad_fp32, grad_fp32 * negative_slope)


def leaky_relu(A, negative_slope=0.01):
    logger.debug("GEMS LEAKY_RELU")
    return leaky_relu_forward_kernel(A, negative_slope)


def leaky_relu_(A, negative_slope=0.01):
    logger.debug("GEMS LEAKY_RELU_")
    return leaky_relu_forward_kernel(A, negative_slope, out0=A)


def leaky_relu_backward(grad_output, self, negative_slope, self_is_result):
    logger.debug("GEMS LEAKY_RELU_BACKWARD")
    return leaky_relu_backward_kernel(grad_output, negative_slope, self)
