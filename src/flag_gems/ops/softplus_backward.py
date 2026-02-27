import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True, False, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def softplus_backward_kernel(grad_output, self, beta, threshold):
    grad_output_fp = grad_output.to(tl.float32)
    self_fp = self.to(tl.float32)
    z = self_fp * beta
    # sigmoid(z) = 1 / (1 + exp(-z))
    sigmoid_z = 1.0 / (1.0 + tl.exp(-z))
    grad_input = tl.where(z > threshold, grad_output_fp, grad_output_fp * sigmoid_z)
    return grad_input.to(grad_output.dtype)


def softplus_backward(grad_output, self, beta, threshold):
    logger.debug("GEMS SOFTPLUS_BACKWARD")
    return softplus_backward_kernel(grad_output, self, beta, threshold)
