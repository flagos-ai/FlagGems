import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

config_ = CodeGenConfig(
    1024,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def floor_func(x):
    x_fp32 = x.to(tl.float32)
    r = (x_fp32 + 12582912.0) - 12582912.0
    d = tl.minimum(tl.maximum((r - x_fp32) * 1e38, 0.0), 1.0)
    return (r - d).to(x.dtype)


def _floor_impl(A, out=None):
    if out is None:
        return floor_func(A)
    floor_func(A, out0=out)
    return out


def floor(A):
    logger.debug("GEMS_KUNLUNXIN FLOOR")
    return _floor_impl(A)


def floor_out(A, *, out=None):
    logger.debug("GEMS_KUNLUNXIN FLOOR_OUT")
    return _floor_impl(A, out)


def floor_(A):
    logger.debug("GEMS_KUNLUNXIN FLOOR_")
    return _floor_impl(A, A)
