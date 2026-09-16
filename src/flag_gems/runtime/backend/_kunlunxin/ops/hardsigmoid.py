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


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def hardsigmoid_func(x):
    xf = x.to(tl.float32)
    y = xf * (1.0 / 6.0) + 0.5
    y = tl.minimum(tl.maximum(y, 0.0), 1.0)
    return y.to(x.dtype)


def hardsigmoid(x):
    logger.debug("GEMS_KUNLUNXIN HARDSIGMOID")
    return hardsigmoid_func(x)


def hardsigmoid_out(x, out):
    logger.debug("GEMS_KUNLUNXIN HARDSIGMOID_OUT")
    assert x.numel() == out.numel()
    hardsigmoid_func(x, out0=out)
    return out
