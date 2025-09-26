import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(
    is_tensor=[True, False, False, False], promotion_methods=[(0, "DEFAULT")]
)
@triton.jit
def elu_forward_kernel(x, alpha, scale, input_scale):
    return tl.where(
        x.to(tl.float32) > 0,
        scale * input_scale * x.to(tl.float32),
        scale * alpha * (tl.exp(x.to(tl.float32) * input_scale) - 1),
    )


@pointwise_dynamic(
    is_tensor=[True, True, False, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def elu_backward_kernel(grad_output, x, alpha, scale, input_scale):
    x_fp32 = x.to(tl.float32)
    grad_pos = grad_output * scale * input_scale
    grad_neg = grad_output * scale * alpha * input_scale * tl.exp(x_fp32 * input_scale)

    return tl.where(x_fp32 > 0, grad_pos, grad_neg)


def elu(A, alpha=1.0, scale=1.0, input_scale=1.0):
    logger.debug("GEMS ELU")
    return elu_forward_kernel(A, alpha, scale, input_scale)


def elu_(A, alpha=1.0, scale=1.0, input_scale=1.0):
    logger.debug("GEMS ELU_")
    return elu_forward_kernel(A, alpha, scale, input_scale, out0=A)


def elu_backward(grad_output, alpha, scale, input_scale, is_result, self_or_result):
    logger.debug("GEMS ELU BACKWARD")
    grad_input = elu_backward_kernel(
        grad_output, self_or_result, alpha, scale, input_scale
    )
    return grad_input
