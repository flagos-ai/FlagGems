import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(is_tensor=[True, False, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def threshold_kernel(self, threshold, value):
    return tl.where(self > threshold, self, value)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def threshold_backward_kernel(grad_output, self, threshold):
    return tl.where(self > threshold, grad_output, 0)


def threshold(self, threshold, value):
    logger.debug("GEMS THRESHOLD FORWARD")
    output = threshold_kernel(self, threshold, value)
    return output


def threshold_backward(grad_output, self, threshold):
    logger.debug("GEMS THRESHOLD BACKWARD")
    grad_input = threshold_backward_kernel(grad_output, self, threshold)
    return grad_input
