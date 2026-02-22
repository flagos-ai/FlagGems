import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def leaky_relu_forward(x, negative_slope):
    return tl.where(x > 0, x, x * negative_slope)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def leaky_relu_backward_kernel(grad_output, self_or_result, negative_slope):
    return tl.where(self_or_result > 0, grad_output, grad_output * negative_slope)


def leaky_relu(self, negative_slope=0.01):
    logger.debug("GEMS LEAKY_RELU FORWARD")
    return leaky_relu_forward(self, negative_slope)


def leaky_relu_(self, negative_slope=0.01):
    logger.debug("GEMS LEAKY_RELU_ FORWARD")
    return leaky_relu_forward(self, negative_slope, out0=self)


def leaky_relu_backward(grad_output, self_or_result, negative_slope, self_is_result):
    logger.debug("GEMS LEAKY_RELU BACKWARD")
    return leaky_relu_backward_kernel(grad_output, self_or_result, negative_slope)
