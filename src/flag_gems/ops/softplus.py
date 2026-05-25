import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, False, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def softplus_forward(x, beta, threshold):
    x_fp = x.to(tl.float32)
    z = x_fp * beta
    soft_z = tl.where(z > threshold, z, tl.log(1 + tl.exp(z)))
    out = (soft_z / beta).to(x.dtype)
    return out


@pointwise_dynamic(is_tensor=[True, True, False, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def softplus_backward_kernel(x, dy, beta, threshold):
    x_fp = x.to(tl.float32)
    dy_fp = dy.to(tl.float32)
    z = x_fp * beta
    # sigmoid(z) = 1/(1+exp(-z))
    sig = tl.where(z > threshold, 1.0, 1.0 / (1.0 + tl.exp(-z)))
    return (dy_fp * sig).to(x.dtype)


def softplus(self, beta=1.0, threshold=20.0):
    logger.debug("GEMS SOFTPLUS FORWARD")
    output = softplus_forward(self, beta, threshold)
    return output


def softplus_backward(grad_output, self, beta=1.0, threshold=20.0):
    logger.debug("GEMS SOFTPLUS BACKWARD")
    grad_input = softplus_backward_kernel(self, grad_output, beta, threshold)
    return grad_input
