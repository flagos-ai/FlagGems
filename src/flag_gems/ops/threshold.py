import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, False, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def threshold_kernel(self, threshold, value):
    return tl.where(self > threshold, self, value)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def threshold_backward_kernel(grad_output, self, threshold):
    return tl.where(self > threshold, grad_output, 0)


def threshold(self, threshold, value):
    logging.debug("GEMS THRESHOLD FORWARD")
    output = threshold_kernel(self, threshold, value)
    return output


def threshold_backward(grad_output, self, threshold):
    logging.debug("GEMS THRESHOLD BACKWARD")
    grad_input = threshold_backward_kernel(grad_output, self, threshold)
    return grad_input
