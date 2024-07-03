import logging

import torch
import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_forward(x):
    return tl.math.tanh(x.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_backward(y, dy):
    return dy * (1.0 - tl.math.pow(y.to(tl.float32), 2))


class Tanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logging.debug("GEMS TANH FORWARD")
        if A.requires_grad is True:
            out = tanh_forward(A.to(torch.float32))
            ctx.save_for_backward(out)
            return out.to(A.dtype)
        else:
            out = tanh_forward(A)
            return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS TANH BACKWARD")
        (out,) = ctx.saved_tensors
        in_grad = tanh_backward(out, out_grad)
        return in_grad


def tanh(A):
    return Tanh.apply(A)
