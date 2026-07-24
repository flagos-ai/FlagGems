import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

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
def trunc_func(x):
    x_fp32 = x.to(tl.float32)
    return tl_extra_shim.trunc(x_fp32).to(x.dtype)


def trunc(A):
    logger.debug("GEMS_KUNLUNXIN TRUNC")
    return trunc_func(A)


def trunc_(A):
    logger.debug("GEMS_KUNLUNXIN TRUNC_")
    trunc_func(A, out0=A)
    return A
