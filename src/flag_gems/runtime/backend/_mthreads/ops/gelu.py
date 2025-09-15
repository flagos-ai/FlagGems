# This custom op requires musa device capability >= 31.
# We determine whether to enable this op by distinguish the op registration for different arch.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)
erf = tl_extra_shim.erf
exp = tl_extra_shim.exp
pow = tl_extra_shim.pow
fast_tanh = tl_extra_shim.fast_tanh
fast_gelu = tl_extra_shim.fast_gelu


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_none(x):
    return fast_gelu(x.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_tanh(x):
    output = (
        0.5
        * x
        * (1 + fast_tanh(x * 0.79788456 * (1 + 0.044715 * pow(x.to(tl.float32), 2))))
    )
    return output


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gelu_backward_none(x, dy):
    scale1: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    scale2: tl.constexpr = 0.3989422803  # 1 / math.sqrt(2 * math.pi)
    x_fp32 = x.to(tl.float32)
    dydx = (
        scale2 * x_fp32 * exp(-pow(scale1 * x_fp32, 2))
        + 0.5 * erf(scale1 * x_fp32)
        + 0.5
    )
    dx = dydx * dy
    return dx


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gelu_backward_tanh(x, dy):
    x_fp32 = x.to(tl.float32)
    # 0.79788456 = math.sqrt(2 / math.pi)
    tanh_out = fast_tanh(0.79788456 * x * (1 + 0.044715 * pow(x_fp32, 2)))
    dydx = 0.5 * x * (
        (1 - pow(tanh_out, 2)) * (0.79788456 + 0.1070322243 * pow(x_fp32, 2))
    ) + 0.5 * (1 + tanh_out)
    dx = dydx * dy
    return dx


class Gelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, approximate):
        logger.debug("GEMS_MTHREADS GELU FORWARD")
        if approximate == "tanh":
            out = gelu_tanh(A)
        else:
            out = gelu_none(A)
        ctx.save_for_backward(A)
        ctx.approximate = approximate
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS_MTHREADS GELU BACKWARD")
        (inp,) = ctx.saved_tensors
        approximate = ctx.approximate
        if approximate == "tanh":
            in_grad = gelu_backward_tanh(inp, out_grad)
        else:
            in_grad = gelu_backward_none(inp, out_grad)
        return in_grad, None


def gelu(A, *, approximate="none"):
    return Gelu.apply(A, approximate)
