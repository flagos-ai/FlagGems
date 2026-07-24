import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

_ASIN = tl_extra_shim.asin
logger = logging.getLogger(__name__)

# arcsin is an alias of asin (out = asin(x)). arcsin/arcsin_/arcsin_out were not
# overridden by kunlunxin, so they fell to the generic ops/arcsin.py bare
# `@pointwise_dynamic` (NO CodeGenConfig) → per-shape recompile IR explosion
# (ir-arcsin-dev5.log: 163K lines / 14M, arcsin_kernel recompiled ~22503x) plus
# a discrete/launch-bound slow path (ir-arcsin_-dev0.log: [1024,65536] gems
# ~157ms, large shapes speedup ~0.007).
# Fix: mirror the sibling unary op cos.py/arctan.py — same INT_TO_FLOAT
# promotion + tuned config_ (vec-OPEN + kunlunAutoGrid + unroll8) → the kernel
# is shape-independent (compiles ONCE) + contiguous block-DMA tiles.
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
def arcsin_func(x):
    return _ASIN(x.to(tl.float32))


def arcsin(x, *, out=None):
    logger.debug("GEMS_KUNLUNXIN ARCSIN FORWARD")
    if out is None:
        return arcsin_func(x)
    arcsin_func(x, out0=out)
    return out


def arcsin_(x):
    logger.debug("GEMS_KUNLUNXIN ARCSIN INPLACE")
    arcsin_func(x, out0=x)
    return x


def arcsin_out(x, *, out=None):
    logger.debug("GEMS_KUNLUNXIN ARCSIN OUT")
    return arcsin(x, out=out)
