import math
import torch
import triton
import triton.language as tl
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic

@pointwise_dynamic
@triton.jit
def sigmoid_forward(x):
    log2e: tl.constexpr = math.log2(math.e)
    return 1 / (1 + tl.math.exp2(-x.to(tl.float32) * log2e))


@pointwise_dynamic
@triton.jit
def sigmoid_backward(y, dy):
    y_f32 = y.to(tl.float32)
    dy_f32 = dy.to(tl.float32)
    return dy_f32 * (1.0 - y_f32) * y_f32


class Sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        if __debug__:
            print("GEMS SIGMOID FORWARD")
        O = sigmoid_forward(A)
        ctx.save_for_backward(O)
        return O

    @staticmethod
    def backward(ctx, out_grad):
        if __debug__:
            print("GEMS SIGMOID BACKWARD")
        (out,) = ctx.saved_tensors
        in_grad = sigmoid_backward(out, out_grad)
        return in_grad


def sigmoid(A):
    return Sigmoid.apply(A)
