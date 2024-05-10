import torch
import triton
import triton.language as tl
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def relu_forward(x):
    return tl.where(x > 0, x, 0)

@pointwise_dynamic
@triton.jit
def relu_backward(x, dy):
    return tl.where(x > 0, dy, 0)

class Relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        if __debug__:
            print("GEMS RELU FORWARD")
        O = relu_forward(A)
        ctx.save_for_backward(A)
        return O

    @staticmethod
    def backward(ctx, out_grad):
        if __debug__:
            print("GEMS RELU BACKWARD")
        (inp,) = ctx.saved_tensors
        in_grad = relu_backward(inp, out_grad)
        return in_grad


def relu(A):
    return Relu.apply(A)
