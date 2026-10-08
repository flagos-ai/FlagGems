# Kunlunxin (XPU) override of log2 / log2_.
#
# log2 was NOT overridden by kunlunxin, so it fell to the generic KernelGen
# ops/log2.py, whose kernel body is `tl.log2(x.to(tl.float32))`. On triton-XPU
# the tl.log2 intrinsic is mis-lowered and the log2 conversion factor gets
# dropped: the kernel computes natural log instead of log2,
#     log2(2.0) -> 0.693147 (ln 2) instead of 1.0,
# i.e. every result is off by exactly ln(2) (res/log(x) == 1.0, verified for
# fp32/fp16/bf16 on [1024], [37,91], [2,19,7]; 36/36 test_log2 cases failed
# with relative error ~0.307). Special values themselves are right
# (0 -> -inf, -0 -> -inf, x<0 -> NaN, +inf -> +inf, NaN -> NaN).
#
# Fix: compute log2 in terms of the (correct) tl.log intrinsic with an explicit
# literal multiplier:
#     log2(x) = log(x) * (1/ln 2) = log(x) * 1.4426950408889634
# (exact same recipe as the sibling log1p.py: tuned CodeGenConfig +
# isCloseVectorization=True, which is REQUIRED for the log family -- with
# isCloseVectorization=False the vectorized log miscompiles bf16, ~1.6% of
# elements off by exactly +ln(2), see log1p.py).
#
# Both ("log2", log2) and ("log2_", log2_) are already present in the generic
# _FULL_CONFIG, so registering these two functions here (and exporting them
# from ops/__init__.py) is enough for SpecOpRegistrar to replace the generic
# ones -- no _install_register_config_patch entry needed (unlike atanh_).
import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# 1 / ln(2) = 1.4426950408889634 (fp32 rel. error ~1.2e-7, far below the test
# rtol of 1.3e-6 for fp32 / 1e-3 for fp16/bf16). Inlined literal: triton jit
# only allows constexpr globals.

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


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")], config=config_)
@triton.jit
def log2_func(x):
    # log2(x) = ln(x) * (1/ln 2), computed in fp32 for precision.
    # NOTE: no trailing .to(x.dtype) on purpose -- the output tensor store
    # performs the final conversion (sibling exp/arcsinh/atanh follow the
    # same pattern).
    return tl.log(x.to(tl.float32)) * 1.4426950408889634


def log2(A, *, out=None):
    logger.debug("GEMS_KUNLUNXIN LOG2")
    if out is None:
        return log2_func(A)
    log2_func(A, out0=out)
    return out


def log2_(A):
    logger.debug("GEMS_KUNLUNXIN LOG2_")
    log2_func(A, out0=A)
    return A
