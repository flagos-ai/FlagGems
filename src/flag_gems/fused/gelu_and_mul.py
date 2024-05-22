import torch
import triton
import triton.language as tl
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def gelu_none_and_mul_kernel(x, y):
    x_fp32 = x.to(tl.float32)
    x_gelu = 0.5 * x_fp32 * (1 + tl.math.erf(x_fp32 * 0.7071067811))
    return x_gelu * y


@pointwise_dynamic
@triton.jit
def gelu_tanh_and_mul_kernel(x, y):
    x_fp32 = x.to(tl.float32)
    x_gelu = (
        0.5
        * x_fp32
        * (
            1
            + tl.math.tanh(
                x_fp32
                * 0.79788456
                * (1 + 0.044715 * tl.math.pow(x_fp32.to(tl.float32), 2))
            )
        )
    )
    return x_gelu * y


class GeluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, approximate="none"):
        logging.debug("GEMS GELU AND MUL FORWARD")
        if approximate == "none":
            O = gelu_none_and_mul_kernel(A, B)
        elif approximate == "tanh":
            O = gelu_tanh_and_mul_kernel(A, B)
        else:
            raise ValueError(f"Invalid approximate value: {approximate}")
        return O


def gelu_and_mul(A, B, approximate="none"):
    return GeluAndMul.apply(A, B, approximate)
