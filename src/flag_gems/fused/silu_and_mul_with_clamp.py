import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, 2, "DEFAULT")])
@triton.jit
def silu_and_mul_with_clamp_kernel(x, y, limit):
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    limit_fp32 = limit.to(tl.float32)

    gate = tl.minimum(x_fp32, limit_fp32)
    up = tl.minimum(tl.maximum(y_fp32, -limit_fp32), limit_fp32)
    gate_silu = tl.fdiv(gate, (1.0 + tl.exp(-gate)))

    return gate_silu * up


@pointwise_dynamic(
    promotion_methods=[
        (0, 1, 2, 3, "DEFAULT"),
        (0, 1, 2, 3, "DEFAULT"),
    ],
    num_outputs=2,
)
@triton.jit
def silu_and_mul_with_clamp_grad_kernel(x, y, dgrad, limit):
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    dgrad_fp32 = dgrad.to(tl.float32)
    limit_fp32 = limit.to(tl.float32)

    gate = tl.minimum(x_fp32, limit_fp32)
    up = tl.minimum(tl.maximum(y_fp32, -limit_fp32), limit_fp32)

    sig = 1 / (1 + tl.exp(-gate))
    gate_silu = gate * sig
    d_gate_silu = sig * (1 + gate * (1 - sig))

    gate_mask = x_fp32 <= limit_fp32
    up_mask = (y_fp32 >= -limit_fp32) & (y_fp32 <= limit_fp32)

    dx = dgrad_fp32 * up * d_gate_silu * gate_mask.to(tl.float32)
    dy = dgrad_fp32 * gate_silu * up_mask.to(tl.float32)

    return dx, dy


class SiluAndMulWithClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, limit):
        limit_tensor = torch.tensor(limit, device=x.device, dtype=x.dtype)
        ctx.save_for_backward(x, y, limit_tensor)
        logger.debug("GEMS SILU_AND_MUL_WITH_CLAMP_FORWARD")
        return silu_and_mul_with_clamp_kernel(x, y, limit_tensor)

    @staticmethod
    def backward(ctx, dgrad):
        x, y, limit_tensor = ctx.saved_tensors
        logger.debug("GEMS SILU_AND_MUL_WITH_CLAMP_BACKWARD")
        dx, dy = silu_and_mul_with_clamp_grad_kernel(x, y, dgrad, limit_tensor)
        return dx, dy, None


def silu_and_mul_with_clamp(x, y, limit):
    return SiluAndMulWithClamp.apply(x, y, limit)


def silu_and_mul_with_clamp_out(x, y, out, limit):
    logger.debug("GEMS SILU_AND_MUL_WITH_CLAMP_OUT")
    limit_tensor = torch.tensor(limit, device=x.device, dtype=x.dtype)
    silu_and_mul_with_clamp_kernel(x, y, limit_tensor, out0=out)
    return out
