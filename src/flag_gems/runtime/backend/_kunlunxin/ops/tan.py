import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    unroll_num=4,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def tan_func(x):
    y = tl_extra_shim.tan(x.to(tl.float32))
    return y


def tan(A):
    logger.debug("GEMS TAN")
    return tan_func(A)


def tan_(A):
    logger.debug("GEMS TAN_")
    tan_func(A, out0=A)
    return A
