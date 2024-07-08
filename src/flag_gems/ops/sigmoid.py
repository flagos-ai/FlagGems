import logging
import math

import torch
import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sigmoid_forward(x):
    log2e: tl.constexpr = math.log2(math.e)
    return 1 / (1 + tl.math.exp2(-x.to(tl.float32) * log2e))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sigmoid_backward(y, dy):
    y_f32 = y.to(tl.float32)
    dy_f32 = dy.to(tl.float32)
    return dy_f32 * (1.0 - y_f32) * y_f32


class Sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logging.debug("GEMS SIGMOID FORWARD")
        if A.requires_grad is True:
            out = sigmoid_forward(A.to(torch.float32))
            ctx.save_for_backward(out)
            return out.to(A.dtype)
        else:
            out = sigmoid_forward(A)
            return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS SIGMOID BACKWARD")
        (out,) = ctx.saved_tensors
        in_grad = sigmoid_backward(out, out_grad)
        return in_grad


def sigmoid(A):
    return Sigmoid.apply(A)
