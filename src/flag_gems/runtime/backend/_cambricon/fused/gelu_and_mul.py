import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
fast_erf = tl_extra_shim.fast_erf
fast_tanh = tl_extra_shim.fast_tanh


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gelu_none_and_mul_kernel(x, y):
    x_fp32 = x.to(tl.float32)
    x_gelu = 0.5 * x_fp32 * (1 + fast_erf(x_fp32 * 0.7071067811))
    return x_gelu * y


@pointwise_dynamic(
    promotion_methods=[(0, 1, 2, "DEFAULT"), (0, 1, 2, "DEFAULT")], num_outputs=2
)
@triton.jit
def gelu_none_and_mul_grad_kernel(x, y, dgrad):
    RCP_SQRT_2: tl.constexpr = 0.7071067811
    COEFF: tl.constexpr = 0.7978845608028654

    x_fp32 = x.to(tl.float32)
    x_gelu = 0.5 * x_fp32 * (1 + fast_erf(x_fp32 * RCP_SQRT_2))

    d_gelu = dgrad * y
    dx = (
        d_gelu
        * 0.5
        * (
            1.0
            + fast_erf(x_fp32 * RCP_SQRT_2)
            + x_fp32 * COEFF * tl.exp(-0.5 * x_fp32 * x_fp32)
        )
    )

    dy = dgrad * x_gelu

    return dx, dy


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gelu_tanh_and_mul_kernel(x, y):
    x_fp32 = x.to(tl.float32)
    x_gelu = (
        0.5
        * x_fp32
        * (1 + fast_tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * x_fp32 * x_fp32)))
    )
    return x_gelu * y


@pointwise_dynamic(
    promotion_methods=[(0, 1, 2, "DEFAULT"), (0, 1, 2, "DEFAULT")], num_outputs=2
)
@triton.jit
def gelu_tanh_and_mul_grad_kernel(x, y, dgrad):
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)

    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    a_cubed = x_fp32 * x_fp32 * x_fp32
    tanh_arg = sqrt_2_over_pi * (x_fp32 + 0.044715 * a_cubed)
    tanh_result = fast_tanh(tanh_arg)
    geglu_a = 0.5 * x_fp32 * (1 + tanh_result)
    dy = geglu_a * dgrad

    term1 = 0.5 * (1 + tanh_result)
    tanh_sq = tanh_result * tanh_result
    term2 = (
        0.5
        * x_fp32
        * (1 - tanh_sq)
        * (sqrt_2_over_pi * (1 + 3 * 0.044715 * x_fp32 * x_fp32))
    )
    dx = dgrad * y_fp32 * (term1 + term2)

    return dx, dy


class GeluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, approximate="none"):
        logger.debug("GEMS_CAMBRICON GELU AND MUL FORWARD")
        ctx.save_for_backward(x, y)
        ctx.approximate = approximate
        if approximate == "none":
            return gelu_none_and_mul_kernel(x, y)
        elif approximate == "tanh":
            return gelu_tanh_and_mul_kernel(x, y)
        else:
            raise ValueError(f"Invalid approximate value: {approximate}")

    @staticmethod
    def backward(ctx, dgrad):
        logger.debug("GEMS_CAMBRICON GELU AND MUL BACKWARD")
        x, y = ctx.saved_tensors
        if ctx.approximate == "none":
            dx, dy = gelu_none_and_mul_grad_kernel(x, y, dgrad)
        else:
            dx, dy = gelu_tanh_and_mul_grad_kernel(x, y, dgrad)
        return dx, dy, None


def gelu_and_mul(A, B, approximate="none"):
    return GeluAndMul.apply(A, B, approximate)
