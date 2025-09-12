import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, False, False, False], promotion_methods=[(0, "DEFAULT")]
)
@triton.jit
def elu_forward_kernel(x, alpha, scale, input_scale):
    return tl.where(
        x > 0,
        scale * input_scale * x,
        scale * alpha * (tl.exp(x.to(tl.float32) * input_scale) - 1),
    )


@pointwise_dynamic(
    is_tensor=[True, False, False, False, True], promotion_methods=[(0, 4, "DEFAULT")]
)
@triton.jit
def elu_backward_kernel_with_self(grad_output, alpha, scale, input_scale, x):
    x_fp32 = x.to(tl.float32)
    grad_input = tl.where(
        x > 0,
        grad_output * scale * input_scale,
        grad_output * (scale * alpha * tl.exp(x_fp32 * input_scale) * input_scale),
    )
    return grad_input


@pointwise_dynamic(
    is_tensor=[True, False, False, False, True], promotion_methods=[(0, 4, "DEFAULT")]
)
@triton.jit
def elu_backward_kernel_with_result(grad_output, alpha, scale, input_scale, y):
    grad_input = tl.where(
        y > 0,
        grad_output * scale * input_scale,
        grad_output * ((y + scale * alpha) * input_scale),
    )
    return grad_input


def elu(A, alpha=1.0, scale=1.0, input_scale=1.0):
    logger.debug("GEMS ELU")
    return elu_forward_kernel(A, alpha, scale, input_scale)


def elu_(A, alpha=1.0, scale=1.0, input_scale=1.0):
    logger.debug("GEMS ELU_")
    return elu_forward_kernel(A, alpha, scale, input_scale, out0=A)


def elu_backward(grad_output, alpha, scale, input_scale, is_result, self_or_result):
    logger.debug("GEMS ELU BACKWARD")
    if is_result:
        return elu_backward_kernel_with_result(
            grad_output, alpha, scale, input_scale, self_or_result
        )
    else:
        return elu_backward_kernel_with_self(
            grad_output, alpha, scale, input_scale, self_or_result
        )
