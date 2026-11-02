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
def asinh__func(x):
    x_fp32 = x.to(tl.float32)
    abs_x = tl.abs(x_fp32)
    r = tl.where(
        abs_x > 1e16,
        abs_x + abs_x,
        abs_x + tl.sqrt(abs_x * abs_x + 1.0),
    )
    y = tl.log(r)
    result = tl.where(x_fp32.to(tl.int32, bitcast=True) < 0, -y, y)
    return result.to(x.dtype)


def asinh_(A):
    logger.debug("GEMS_KUNLUNXIN ASINH_")
    asinh__func(A, out0=A)
    return A
