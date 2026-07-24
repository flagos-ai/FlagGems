import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

_atan = tl_extra_shim.atan
logger = logging.getLogger(__name__)

# arctan is an alias of atan (out = atan(x)). Neither arctan nor arctan_ was
# overridden by kunlunxin, so both fell to the generic ops/arctan_.py bare
# `@pointwise_dynamic` (NO CodeGenConfig, default buffer_size_limit=2048, no
# XPU tuning) → discrete/launch-bound slow path. Baseline (IR
# ir-arctan-dev5.log / ir-arctan_-dev6.log): large shapes ~0.012-0.013
# ([1024,65536] gems ~155-158ms), all shapes 0.012-0.5.
# Fix: mirror the sibling unary op cos.py — same INT_TO_FLOAT promotion + tuned
# config_ (vec-OPEN + kunlunAutoGrid + unroll8) → contiguous block-DMA tiles.
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
def arctan_func(x):
    return _atan(x.to(tl.float32))


def arctan(A):
    logger.debug("GEMS_KUNLUNXIN ARCTAN")
    return arctan_func(A)


def arctan_(A):
    logger.debug("GEMS_KUNLUNXIN ARCTAN_")
    arctan_func(A, out0=A)
    return A
