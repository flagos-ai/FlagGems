import logging
import math

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_SCALE = tl.constexpr(180.0 / math.pi)

# rad2deg = x * (180/pi), a pure scaled-copy unary map. Neither rad2deg nor
# rad2deg_ was overridden by kunlunxin, so both fell to the generic
# ops/rad2deg.py bare pointwise_dynamic (no CodeGenConfig) -> launch-bound
# narrow-DMA slow path (IR ir-rad2deg-dev5.log / ir-rad2deg_-dev5.log).
# Fix: rewrite mirroring sibling deg2rad.py (unary INT_TO_FLOAT + tuned
# config_). vec-OPEN (isCloseVectorization=False) like the scaled-copy family
# (alias_copy/cos/deg2rad) -> contiguous block-DMA tiles.
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
def rad2deg_func(x):
    return x.to(tl.float32) * _SCALE


def rad2deg(A):
    logger.debug("GEMS_KUNLUNXIN RAD2DEG")
    return rad2deg_func(A)


def rad2deg_(A):
    logger.debug("GEMS_KUNLUNXIN RAD2DEG_")
    return rad2deg_func(A, out0=A)
