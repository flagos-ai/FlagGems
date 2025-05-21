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


class GeluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, approximate="none"):
        logger.debug("GEMS_CAMBRICON GELU AND MUL FORWARD")
        if approximate == "none":
            return gelu_none_and_mul_kernel(A, B)
        elif approximate == "tanh":
            return gelu_tanh_and_mul_kernel(A, B)
        else:
            raise ValueError(f"Invalid approximate value: {approximate}")


def gelu_and_mul(A, B, approximate="none"):
    return GeluAndMul.apply(A, B, approximate)
