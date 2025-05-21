import logging

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def relu_forward(x):
    return tl.where(x > 0, x, 0)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def relu_backward(x, dy):
    return tl.where(x > 0, dy, 0)


class Relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logger.debug("GEMS_CAMBRICON RELU FORWARD")
        out = relu_forward(A)
        ctx.save_for_backward(A)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS_CAMBRICON RELU BACKWARD")
        out_grad = out_grad.contiguous()
        (inp,) = ctx.saved_tensors
        in_grad = relu_backward(inp, out_grad)
        return in_grad


def relu(A):
    return Relu.apply(A)
