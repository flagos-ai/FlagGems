import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(is_tensor=[True, True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def lerp_tensor_kernel(input, end, weight):
    input32 = input.to(tl.float32)
    end32 = end.to(tl.float32)
    weight32 = weight.to(tl.float32)
    res32 = tl.where(
        tl.abs(weight32) < 0.5,
        input32 + weight32 * (end32 - input32),
        end32 - (end32 - input32) * (1 - weight32),
    )
    return res32.to(input.dtype)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    dtypes=[None, None, float],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit(do_not_specialize=["weight"])
def lerp_scalar_kernel_head(input, end, weight):
    input32 = input.to(tl.float32)
    end32 = end.to(tl.float32)
    weight32 = weight.to(tl.float32)
    return (input32 + weight32 * (end32 - input32)).to(input.dtype)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    dtypes=[None, None, float],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit(do_not_specialize=["weight"])
def lerp_scalar_kernel_tail(input, end, weight):
    input32 = input.to(tl.float32)
    end32 = end.to(tl.float32)
    weight32 = weight.to(tl.float32)
    return (end32 - (end32 - input32) * (1 - weight32)).to(input.dtype)


def lerp_tensor(input, end, weight):
    logger.debug("GEMS LERP TENSOR")
    out = lerp_tensor_kernel(input, end, weight)
    return out


def lerp_tensor_(input, end, weight):
    logger.debug("GEMS LERP INPLACE TENSOR")
    return lerp_tensor_kernel(input, end, weight, out0=input)


def lerp_scalar(input, end, weight):
    logger.debug("GEMS LERP TENSOR")
    if weight < 0.5:
        out = lerp_scalar_kernel_head(input, end, weight)
    else:
        out = lerp_scalar_kernel_tail(input, end, weight)
    return out


def lerp_scalar_(input, end, weight):
    logger.debug("GEMS LERP INPLACE TENSOR")
    if weight < 0.5:
        return lerp_scalar_kernel_head(input, end, weight, out0=input)
    else:
        return lerp_scalar_kernel_tail(input, end, weight, out0=input)
