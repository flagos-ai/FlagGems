import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True, False])
@triton.jit
def softshrink_backward_kernel(grad_output, x, lambd):
    # grad_input = grad_output * (1 if |x| > lambd else 0)
    mask = tl.abs(x) > lambd
    return tl.where(mask, grad_output, 0.0)


def softshrink_backward(grad_output, x, lambd=0.5):
    logger.debug("GEMS SOFTSHRINK_BACKWARD")
    return softshrink_backward_kernel(grad_output, x, lambd)
