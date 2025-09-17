import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def relu_forward(x):
    return tl.where(x > 0, x, 0)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def relu_backward(x, dy):
    return tl.where(x > 0, dy, 0)


def relu(self):
    logger.debug("GEMS_CAMBRICON RELU FORWARD")
    output = relu_forward(self)
    return output


def relu_(A):
    logger.debug("GEMS_CAMBRICON RELU_ FORWARD")
    out = relu_forward(A, out0=A)
    return out
