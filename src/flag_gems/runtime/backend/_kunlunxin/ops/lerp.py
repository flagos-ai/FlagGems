import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(is_tensor=[True, True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def lerp_tensor_kernel(input, end, weight):
    return tl.where(
        tl.abs(weight) < 0.5,
        input + weight * (end - input),
        end - (end - input) * (1 - weight),
    )


@pointwise_dynamic(
    is_tensor=[True, True, False],
    dtypes=[None, None, float],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit(do_not_specialize=["weight"])
def lerp_scalar_kernel_head(input, end, weight):
    return input + weight * (end - input)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    dtypes=[None, None, float],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit(do_not_specialize=["weight"])
def lerp_scalar_kernel_tail(input, end, weight):
    return end - (end - input) * (1 - weight)


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
