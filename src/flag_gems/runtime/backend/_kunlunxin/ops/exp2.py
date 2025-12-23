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
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def exp2_func(x):
    return tl.exp2(x.to(tl.float32))


def exp2(A):
    logger.debug("GEMS EXP")
    return exp2_func(A)


def exp2_(A):
    logger.debug("GEMS EXP_")
    return exp2_func(A, out0=A)
