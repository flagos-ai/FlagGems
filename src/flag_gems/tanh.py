import torch
import triton
import triton.language as tl
import logging
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def tanh_forward(x):
    return tl.math.tanh(x.to(tl.float32))


@pointwise_dynamic
@triton.jit
def tanh_backward(y, dy):
    return dy * (1.0 - tl.math.pow(y.to(tl.float32), 2))


class Tanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logging.debug("GEMS TANH FORWARD")
        O = tanh_forward(A)
        ctx.save_for_backward(O)
        return O

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS TANH BACKWARD")
        (out,) = ctx.saved_tensors
        in_grad = tanh_backward(out, out_grad)
        return in_grad


def tanh(A):
    return Tanh.apply(A)
