import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic, tl_extra_shim

div_rn = tl_extra_shim.div_rn


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def silu_forward(x):
    x_fp32 = x.to(tl.float32)
    y = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    return y


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def silu_backward_kernel(x, dy):
    dy_fp32 = dy.to(tl.float32)
    x_fp32 = x.to(tl.float32)
    sigma = div_rn(1.0, 1.0 + tl.exp(-x_fp32))
    dx = dy_fp32 * sigma * (1.0 + x_fp32 * (1.0 - sigma))
    return dx


def silu(self):
    logging.debug("GEMS SILU FORWARD")
    output = silu_forward(self)
    return output


def silu_backward(grad_output, self):
    logging.debug("GEMS SILU BACKWARD")
    grad_input = silu_backward_kernel(self, grad_output)
    return grad_input


def silu_(A):
    logging.debug("GEMS SILU_ FORWARD")
    out = silu_forward(A, out0=A)
    return out
