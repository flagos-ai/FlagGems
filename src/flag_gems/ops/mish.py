import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def mish_kernel(x):
    x_fp = x.to(tl.float32)
    softplus = tl.log(1 + tl.exp(x_fp))
    tanh_sp = tl.tanh(softplus)
    out = x_fp * tanh_sp
    return out.to(x.dtype)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def mish_backward_kernel(grad_output, x):
    x_fp = x.to(tl.float32)
    softplus = tl.log(1 + tl.exp(x_fp))
    tanh_sp = tl.tanh(softplus)
    sech_sq = 1 - tanh_sp * tanh_sp
    sigmoid = tl.exp(x_fp) / (1 + tl.exp(x_fp))
    grad_input = grad_output * (tanh_sp + x_fp * sech_sq * sigmoid)
    return grad_input.to(grad_output.dtype)


def mish(x):
    logger.debug("GEMS MISH")
    return mish_kernel(x)


def mish_backward(grad_output, x):
    logger.debug("GEMS MISH BACKWARD")
    return mish_backward_kernel(grad_output, x)
