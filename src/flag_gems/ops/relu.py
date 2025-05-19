import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def relu_forward(x):
    return tl.where(x > 0, x, 0)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def relu_backward(x, dy):
    return tl.where(x > 0, dy, 0)


def relu(self):
    logging.debug("GEMS RELU FORWARD")
    output = relu_forward(self)
    return output


def relu_(A):
    logging.debug("GEMS RELU_ FORWARD")
    out = relu_forward(A, out0=A)
    return out
