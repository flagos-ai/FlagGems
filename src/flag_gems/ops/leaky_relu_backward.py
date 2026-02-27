import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True, False, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def leaky_relu_backward_kernel(grad_output, self, negative_slope, self_is_result):
    grad_output_fp32 = grad_output.to(tl.float32)
    self_fp32 = self.to(tl.float32)
    # For leaky_relu_backward:
    # - If self > 0: gradient = grad_output * 1.0
    # - If self <= 0: gradient = grad_output * negative_slope
    # The self_is_result flag indicates whether self is the output of forward pass.
    # The logic is the same since we check > 0 in both cases.
    return tl.where(self_fp32 > 0, grad_output_fp32, grad_output_fp32 * negative_slope)


def leaky_relu_backward(grad_output, self, negative_slope, self_is_result):
    logger.debug("GEMS LEAKY_RELU_BACKWARD")
    grad_input = leaky_relu_backward_kernel(grad_output, self, negative_slope, self_is_result)
    return grad_input
