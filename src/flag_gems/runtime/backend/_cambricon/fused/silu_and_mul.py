import logging

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def silu_and_mul_kernel(x, y):
    x_fp32 = x.to(tl.float32)
    x_silu = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    return x_silu * y


@pointwise_dynamic(
    promotion_methods=[(0, 1, 2, "DEFAULT"), (0, 1, 2, "DEFAULT")], num_outputs=2
)
@triton.jit
def silu_and_mul_grad_kernel(x, y, dgrad):
    x_fp32 = x.to(tl.float32)
    sig = tl.extra.mlu.libdevice.fast_sigmoid(x_fp32)
    x_silu = x_fp32 * sig
    d_x_silu = sig * (1 + x_fp32 * (1 - sig))
    dx = d_x_silu * dgrad * y
    dy = dgrad * x_silu
    return dx, dy


class SiluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        logger.debug("GEMS_CAMBRICON SILU AND MUL FORWARD")
        return silu_and_mul_kernel(A, B)

    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A, grad_B = silu_and_mul_grad_kernel(A, B, grad_output)
        return grad_A, grad_B


def silu_and_mul(A, B):
    return SiluAndMul.apply(A, B)
