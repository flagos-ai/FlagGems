import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


_atan2 = tl_extra_shim.atan2

# Without an explicit CodeGenConfig, pointwise_dynamic specializes the kernel
# per input shape on XPU -> per-shape recompile -> IR explosion, and the default
# tiny tile<256> no-unroll codegen underutilizes the XPU badly
# (baseline ~0.007-0.45x torch; see ir-asin_-dev4.log, 163k-line IR dump).
# kunlunAutoGrid=True + prefer_1d_tile + bounded tile makes the kernel
# shape-independent so it compiles ONCE and covers large tensors. Mirrors acos.
#
# Close XPU vectorization for the atan2-based formula to avoid large-shape
# intrinsic precision regressions.
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit()
def asin_kernel(x):
    x_f32 = x.to(tl.float32)
    inside = (x_f32 >= -1.0) & (x_f32 <= 1.0)
    cosine = tl.sqrt(tl.maximum(1.0 - x_f32 * x_f32, 0.0))
    result = _atan2(x_f32, cosine)
    return tl.where(inside, result, float("nan"))


def asin(x):
    logger.debug("GEMS_KUNLUNXIN ASIN")
    y = asin_kernel(x)
    return y


def asin_(x):
    logger.debug("GEMS_KUNLUNXIN ASIN_")
    asin_kernel(x, out0=x)
    return x
