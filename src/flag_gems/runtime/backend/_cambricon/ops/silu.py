import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
div_rn = tl_extra_shim.div_rn


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def silu_forward(x):
    x_fp32 = x.to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-x_fp32)) * x_fp32
    return y


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def silu_backward(x, dy):
    dy_fp32 = dy.to(tl.float32)
    x_fp32 = x.to(tl.float32)
    sigma = 1.0 / (1.0 + tl.exp(-x_fp32))
    dx = dy_fp32 * sigma * (1.0 + x_fp32 * (1.0 - sigma))
    return dx


class Silu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logger.debug("GEMS_CAMBRICON SILU FORWARD")
        out = silu_forward(A)
        ctx.save_for_backward(A)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS_CAMBRICON SILU BACKWARD")
        out_grad = out_grad.contiguous()
        (inp,) = ctx.saved_tensors
        in_grad = silu_backward(inp, out_grad)
        return in_grad


def silu(A):
    return Silu.apply(A)
