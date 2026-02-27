import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def hardsigmoid_backward_kernel(grad_output, self):
    # hardsigmoid(x) = clamp(x/6 + 0.5, 0, 1)
    # derivative: 0 if self <= -3 or self >= 3, else 1/6
    self_fp32 = self.to(tl.float32)
    grad_output_fp32 = grad_output.to(tl.float32)

    # Compute mask: True if -3 < self < 3
    in_range = (self_fp32 > -3.0) & (self_fp32 < 3.0)

    # Apply gradient: grad_output / 6 if in range, else 0
    grad_input = tl.where(in_range, grad_output_fp32 / 6.0, 0.0)

    return grad_input


def hardsigmoid_backward(grad_output, self):
    logger.debug("GEMS HARDSIGMOID_BACKWARD")
    return hardsigmoid_backward_kernel(grad_output, self)
