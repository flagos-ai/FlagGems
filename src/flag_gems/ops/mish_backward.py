import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

tanh = tl_extra_shim.tanh

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def mish_backward_kernel(x, dy):
    # mish(x) = x * tanh(softplus(x))
    # mish'(x) = tanh(softplus(x)) + x * sigmoid(x) * (1 - tanh^2(softplus(x)))
    x_fp32 = x.to(tl.float32)
    dy_fp32 = dy.to(tl.float32)

    # softplus(x) = ln(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(x_fp32))

    # tanh(softplus(x))
    t = tanh(sp)

    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x_fp32))

    # mish'(x) = t + x * sig * (1 - t^2)
    grad = t + x_fp32 * sig * (1.0 - t * t)

    dx = dy_fp32 * grad
    return dx


def mish_backward(grad_output, self):
    logger.debug("GEMS MISH_BACKWARD")
    grad_input = mish_backward_kernel(self, grad_output)
    return grad_input
