import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def exp_func(x):
    return tl.exp(x.to(tl.float32))


def exp(A):
    logger.debug("GEMS EXP")
    return exp_func(A)


def exp_(A):
    logger.debug("GEMS EXP_")
    return exp_func(A, out0=A)


# exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def exp_out(A, out):
    logger.debug("GEMS EXP_OUT")
    return exp_func(A, out0=out)
