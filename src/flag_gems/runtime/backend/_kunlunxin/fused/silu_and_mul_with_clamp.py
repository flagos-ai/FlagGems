import logging

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# The generic fused/silu_and_mul_with_clamp.py passes `limit` as a 0-d tensor
# (extra broadcast input) AND uses the SHARED flag_gems.utils.pointwise_dynamic
# (nvidia-style codegen). On XPU this kills the contiguous 1D fast path -> the
# huge-N shapes explode at compile time (>350s) and run at ~0.003 speedup.
# Fix: route through the kunlunxin pointwise_dynamic (same as silu_and_mul) and
# feed `limit` as a plain scalar (is_tensor last arg = False). That restores the
# pure 2-tensor stride-1 path and lifts speedup to ~0.5-0.68.


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit
def silu_and_mul_with_clamp_kernel(x, y, limit):
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)

    gate = tl.minimum(x_fp32, limit)
    up = tl.minimum(tl.maximum(y_fp32, -limit), limit)
    gate_silu = tl.fdiv(gate, (1.0 + tl.exp(-gate)))

    return gate_silu * up


@pointwise_dynamic(
    is_tensor=[True, True, True, False],
    promotion_methods=[
        (0, 1, 2, "DEFAULT"),
        (0, 1, 2, "DEFAULT"),
    ],
    num_outputs=2,
)
@triton.jit
def silu_and_mul_with_clamp_grad_kernel(x, y, dgrad, limit):
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    dgrad_fp32 = dgrad.to(tl.float32)

    gate = tl.minimum(x_fp32, limit)
    up = tl.minimum(tl.maximum(y_fp32, -limit), limit)

    sig = 1 / (1 + tl.exp(-gate))
    gate_silu = gate * sig
    d_gate_silu = sig * (1 + gate * (1 - sig))

    gate_mask = x_fp32 <= limit
    up_mask = (y_fp32 >= -limit) & (y_fp32 <= limit)

    dx = dgrad_fp32 * up * d_gate_silu * gate_mask.to(tl.float32)
    dy = dgrad_fp32 * gate_silu * up_mask.to(tl.float32)

    return dx, dy


class SiluAndMulWithClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, limit):
        ctx.save_for_backward(x, y)
        ctx.limit = limit
        logger.debug("GEMS_KUNLUNXIN SILU_AND_MUL_WITH_CLAMP_FORWARD")
        return silu_and_mul_with_clamp_kernel(x, y, limit)

    @staticmethod
    def backward(ctx, dgrad):
        x, y = ctx.saved_tensors
        logger.debug("GEMS_KUNLUNXIN SILU_AND_MUL_WITH_CLAMP_BACKWARD")
        dx, dy = silu_and_mul_with_clamp_grad_kernel(x, y, dgrad, ctx.limit)
        return dx, dy, None


def silu_and_mul_with_clamp(x, y, limit):
    return SiluAndMulWithClamp.apply(x, y, limit)


def silu_and_mul_with_clamp_out(x, y, out, limit):
    logger.debug("GEMS_KUNLUNXIN SILU_AND_MUL_WITH_CLAMP_OUT")
    silu_and_mul_with_clamp_kernel(x, y, limit, out0=out)
    return out
