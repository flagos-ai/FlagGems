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
def special_log1p_out_func(x):
    return tl.log(1.0 + x.to(tl.float32)).to(x.dtype)


def special_log1p_out(A, out):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_LOG1P_OUT")
    return special_log1p_out_func(A, out0=out)


def special_log1p(A):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_LOG1P")
    return special_log1p_out_func(A)
