import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

_acos = tl_extra_shim.acos
_atan2 = tl_extra_shim.atan2
logger = logging.getLogger(__name__)

# arccos is an alias of acos (out = acos(x)). Neither arccos nor arccos_ was
# overridden by kunlunxin, so both fell to the generic ops/arccos.py bare
# `@pointwise_dynamic` (NO CodeGenConfig) → per-shape recompile IR explosion
# (ir-arccos_-dev4.log: 163K lines / 14M, arccos_kernel recompiled ~22503x)
# + discrete/launch-bound slow path.
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
def arccos_func(x):
    x_f32 = x.to(tl.float32)
    in_domain = tl.abs(x_f32) <= 1.0
    radicand = tl.maximum(1.0 - x_f32 * x_f32, 0.0)
    stable = _atan2(tl.sqrt(radicand), x_f32)
    return tl.where(in_domain, stable, _acos(x_f32))


def arccos(A):
    logger.debug("GEMS_KUNLUNXIN ARCCOS")
    return arccos_func(A)


def arccos_(A):
    logger.debug("GEMS_KUNLUNXIN ARCCOS_")
    arccos_func(A, out0=A)
    return A
