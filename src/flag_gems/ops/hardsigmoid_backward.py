import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def hardsigmoid_backward_kernel(grad_output, x):
    x_f = x.to(tl.float32)
    # hardsigmoid(x) = clamp(x/6 + 0.5, 0, 1)
    # derivative = 1/6 when 0 < x/6 + 0.5 < 1, else 0
    in_range = (x_f > -3.0) & (x_f < 3.0)
    grad = tl.where(in_range, grad_output / 6.0, 0.0)
    return grad.to(grad_output.dtype)


def hardsigmoid_backward(grad_output, x):
    logger.debug("GEMS HARDSIGMOID BACKWARD")
    return hardsigmoid_backward_kernel(grad_output, x)
