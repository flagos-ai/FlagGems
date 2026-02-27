import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def hardswish_backward_kernel(grad_output, self):
    # hardswish(x) = x * clamp(x + 3, 0, 6) / 6
    # gradient (using strict inequalities for boundary continuity):
    #   if x < -3: 0
    #   if x > 3:  grad_output
    #   else:      grad_output * (2*x + 3) / 6
    grad_fp32 = grad_output.to(tl.float32)
    x_fp32 = self.to(tl.float32)

    lower = x_fp32 < -3.0
    upper = x_fp32 > 3.0

    grad_input = tl.where(
        lower,
        0.0,
        tl.where(upper, grad_fp32, grad_fp32 * (2.0 * x_fp32 + 3.0) / 6.0),
    )
    return grad_input


def hardswish_backward(grad_output, self):
    logger.debug("GEMS HARDSWISH_BACKWARD")
    return hardswish_backward_kernel(grad_output, self)
