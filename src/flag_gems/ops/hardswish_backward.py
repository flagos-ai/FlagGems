import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def hardswish_backward_kernel(grad_output, x):
    x_f = x.to(tl.float32)
    # hardswish(x) = x * hardsigmoid(x) = x * clamp(x/6 + 0.5, 0, 1)
    # derivative = hardsigmoid(x) + x * hardsigmoid'(x)
    in_range = (x_f > -3.0) & (x_f < 3.0)
    hardsigmoid_val = tl.minimum(tl.maximum(x_f / 6.0 + 0.5, 0.0), 1.0)
    hardsigmoid_grad = tl.where(in_range, 1.0 / 6.0, 0.0)
    grad = grad_output * (hardsigmoid_val + x_f * hardsigmoid_grad)
    return grad.to(grad_output.dtype)


def hardswish_backward(grad_output, x):
    logger.debug("GEMS HARDSWISH BACKWARD")
    return hardswish_backward_kernel(grad_output, x)
