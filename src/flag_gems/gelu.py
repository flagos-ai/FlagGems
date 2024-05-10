import torch
import triton
import triton.language as tl

from flag_gems.utils.pointwise_dynamic import pointwise_dynamic

@pointwise_dynamic
@triton.jit
def gelu_none(x):
    scale = 0.7071067811
    output = 0.5 * x * (1 + tl.math.erf(x * scale))
    return output

@pointwise_dynamic
@triton.jit
def gelu_tanh(x):
    output = (
        0.5
        * x
        * (
            1
            + tl.math.tanh(
                x * 0.79788456 * (1 + 0.044715 * tl.math.pow(x.to(tl.float32), 2))
            )
        )
    )
    return output


def gelu(A, *, approximate="none"):
    if __debug__:
        print("GEMS GELU")

    if approximate == "tanh":
        return gelu_tanh(A)
    else:
        return gelu_none(A)

