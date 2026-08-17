import logging
import math

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_SCALE = tl.constexpr(math.pi / 180.0)

# deg2rad = x * (pi/180), a pure scaled-copy unary map. Neither deg2rad,
# deg2rad_ nor deg2rad.out was overridden by kunlunxin, so all three fell to
# the generic ops/deg2rad.py hand-written kernel (hard-coded BLOCK_SIZE=1024,
# grid=cdiv(n,1024), no CodeGenConfig) → launch-bound narrow-DMA slow path.
# Baseline (IR ir-deg2rad*-dev*): large shapes ~0.03-0.09 ([1024,65536] gems
# ~4.4ms, [64,64] first-block 243ms do_bench warmup).
# Fix: rewrite as pointwise_dynamic mirroring sibling exp.py (unary INT_TO_FLOAT
# + tuned config_). vec-OPEN (isCloseVectorization=False) like the scaled-copy
# family (alias_copy/cos) → contiguous block-DMA tiles.
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
def deg2rad_func(x):
    return x.to(tl.float32) * _SCALE


def deg2rad(A):
    logger.debug("GEMS_KUNLUNXIN DEG2RAD")
    return deg2rad_func(A)


def deg2rad_(A):
    logger.debug("GEMS_KUNLUNXIN DEG2RAD_")
    return deg2rad_func(A, out0=A)


# deg2rad.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def deg2rad_out(A, out):
    logger.debug("GEMS_KUNLUNXIN DEG2RAD_OUT")
    return deg2rad_func(A, out0=out)
