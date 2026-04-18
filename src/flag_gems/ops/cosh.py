import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def cosh_func(x):
    # cosh(x) = (exp(x) + exp(-x)) / 2
    # Use float32 for intermediate computation to avoid overflow in fp16/bf16
    x_f = x.to(tl.float32)
    return (tl.exp(x_f) + tl.exp(-x_f)) * 0.5


def cosh(A):
    logger.debug("GEMS COSH")
    return cosh_func(A)


def cosh_(A):
    logger.debug("GEMS COSH_")
    cosh_func(A, out0=A)
    return A


def cosh_out(A, *, out=None):
    logger.debug("GEMS COSH OUT")
    return cosh_func(A, out0=out)
