import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def log_sigmoid_forward(x):
    return tl.minimum(x, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(x).to(tl.float32)))


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def log_sigmoid_backward_kernel(dy, x):
    dy_f32 = dy.to(tl.float32)
    x_f32 = x.to(tl.float32)
    # d/dx log_sigmoid(x) = 1 - sigmoid(x) = sigmoid(-x)
    sig_neg = 1.0 / (1.0 + tl.exp(x_f32))
    return (dy_f32 * sig_neg).to(x.dtype)


def log_sigmoid(x):
    logger.debug("GEMS LOG_SIGMOID FORWARD")

    return log_sigmoid_forward(x)


def log_sigmoid_backward(grad_output, self, buffer=None):
    logger.debug("GEMS LOG_SIGMOID BACKWARD")
    grad_input = log_sigmoid_backward_kernel(grad_output, self)
    return grad_input
