import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def expm1_func(x):
    return tl.exp(x.to(tl.float32)) - 1.0


def expm1(A):
    logger.debug("GEMS EXPM1")
    return expm1_func(A)


def expm1_(A):
    logger.debug("GEMS EXPM1_")
    return expm1_func(A, out0=A)


# expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def expm1_out(A, out):
    logger.debug("GEMS EXPM1_OUT")
    return expm1_func(A, out0=out)
