import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


_asin = tl_extra_shim.asin

# Without an explicit CodeGenConfig, pointwise_dynamic specializes the kernel
# per input shape on XPU -> per-shape recompile -> IR explosion, and the default
# tiny tile<256> no-unroll codegen underutilizes the XPU badly
# (baseline ~0.007-0.45x torch; see ir-asin_-dev4.log, 163k-line IR dump).
# kunlunAutoGrid=True + prefer_1d_tile + bounded tile makes the kernel
# shape-independent so it compiles ONCE and covers large tensors. Mirrors acos.
#
# isCloseVectorization MUST stay False (vec OPEN) for the _asin transcendental:
# flipping it to True is catastrophic here (measured [1024,65536] fp32 211ms /
# [4096,4096] 55ms, avg collapses back to ~0.069 == baseline). This is the
# OPPOSITE of silu (where vec OPEN spiked); tune vectorization per-op, not by
# copying a sibling's flag.
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
@triton.jit()
def asin_kernel(x):
    return _asin(x.to(tl.float32))


def asin(x):
    logger.debug("GEMS_KUNLUNXIN ASIN")
    y = asin_kernel(x)
    return y


def asin_(x):
    logger.debug("GEMS_KUNLUNXIN ASIN_")
    asin_kernel(x, out0=x)
    return x
