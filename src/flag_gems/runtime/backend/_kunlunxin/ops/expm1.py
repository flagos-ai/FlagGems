import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

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
def expm1_func(x):
    return tl.exp(x.to(tl.float32)) - 1.0


def expm1(A):
    logger.debug("GEMS_KUNLUNXIN EXPM1")
    return expm1_func(A)


def expm1_(A):
    logger.debug("GEMS_KUNLUNXIN EXPM1_")
    return expm1_func(A, out0=A)


# expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def expm1_out(A, out):
    logger.debug("GEMS_KUNLUNXIN EXPM1_OUT")
    return expm1_func(A, out0=out)
