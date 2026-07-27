import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

_ATAN2 = tl_extra_shim.atan2
logger = logging.getLogger(__name__)

# The XPU asin intrinsic is not accurate enough for the pointwise tolerance.
# Use a float32 atan2/sqrt identity and disable vectorization for this compound
# expression; all three public arcsin variants share this body.
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def arcsin_precise_func(x):
    x = x.to(tl.float32)
    result = _ATAN2(x, tl.sqrt((1.0 - x) * (1.0 + x)))
    return tl.where(tl.abs(x) > 1.0, float("nan"), result)


def arcsin(x, *, out=None):
    logger.debug("GEMS_KUNLUNXIN ARCSIN FORWARD")
    if out is None:
        return arcsin_precise_func(x)
    arcsin_precise_func(x, out0=out)
    return out


def arcsin_(x):
    logger.debug("GEMS_KUNLUNXIN ARCSIN INPLACE")
    arcsin_precise_func(x, out0=x)
    return x


def arcsin_out(x, *, out=None):
    logger.debug("GEMS_KUNLUNXIN ARCSIN OUT")
    return arcsin(x, out=out)
