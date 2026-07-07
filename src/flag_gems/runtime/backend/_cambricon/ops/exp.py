import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def exp_func(x, inplace):
    return tl.exp(x.to(tl.float32))


def exp(A):
    logger.debug("GEMS_CAMBRICON EXP")
    return exp_func(A, False)


def exp_(A):
    logger.debug("GEMS_CAMBRICON EXP_")
    return exp_func(A, True, out0=A)


# exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def exp_out(A, out):
    logger.debug("GEMS_CAMBRICON EXP_OUT")
    return exp_func(A, True, out0=out)
