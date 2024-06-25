import logging

import torch
import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[[0, "DEFAULT"]])
@triton.jit
def silu_forward(x):
    x_fp32 = x.to(tl.float32)
    y = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    return y


@pointwise_dynamic(promotion_methods=[[0, "DEFAULT"]])
@triton.jit
def silu_backward(x, dy):
    dy_fp32 = dy.to(tl.float32)
    x_fp32 = x.to(tl.float32)
    sigma = tl.math.div_rn(1.0, 1.0 + tl.exp(-x_fp32))
    dx = dy_fp32 * sigma * (1.0 + x_fp32 * (1.0 - sigma))
    return dx


class Silu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logging.debug("GEMS SILU FORWARD")
        out = silu_forward(A)
        ctx.save_for_backward(A)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS SILU BACKWARD")
        (inp,) = ctx.saved_tensors
        in_grad = silu_backward(inp, out_grad)
        return in_grad


def silu(A):
    return Silu.apply(A)
