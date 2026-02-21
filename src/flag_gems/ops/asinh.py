import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def asinh_kernel(x):
    # 数学公式: asinh(x) = ln(x + sqrt(x^2 + 1))
    x_f32 = x.to(tl.float32)
    abs_x = tl.abs(x_f32)
    res = tl.log(abs_x + tl.sqrt(abs_x * abs_x + 1.0))
    return tl.where(x_f32 < 0, -res, res)


def asinh(input):
    return asinh_kernel(input)


def asinh_out(input, out):
    asinh_kernel(input, out0=out)
    return out


def asinh_(input):
    asinh_kernel(input, out0=input)
    return input
