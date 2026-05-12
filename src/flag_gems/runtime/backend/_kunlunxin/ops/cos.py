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
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def cos_func(x):
    return tl.cos(x.to(tl.float32))


def cos(A):
    logger.debug("GEMS_KUNLUNXIN COS")
    return cos_func(A)


def cos_(A):
    logger.debug("GEMS_KUNLUNXIN COS_")
    cos_func(A, out0=A)
    return A
