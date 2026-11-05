# Kunlunxin (XPU) override of log10_ (in-place) and log10_out (out variant).
#
# log10_ was NOT overridden by kunlunxin, so it fell to the generic
# `flag_gems/ops/log10.py` which uses the bare `pointwise_dynamic` (no
# CodeGenConfig) -> per-shape recompile / IR explosion and discrete access on
# XPU -> catastrophic latency on large shapes.
#
# Fix: reuse the proven memory-bound unary recipe (acosh.py / log1p.py): a
# tuned CodeGenConfig (prefer_1d_tile, buffer_size_limit=4096,
# kunlunAutoGrid=True, unroll_num=8) on the pointwise_dynamic kernel; log10_
# shares the kernel via out0=A.
#
# isCloseVectorization=False (vectorization OPEN, matching acosh.py). Unlike
# log1p (log(1+x)), log10 = log(x)*log10(e) does NOT hit the bf16 vectorized
# tl.log miscompile (the +ln(2) error is triggered by log1p's `1.0 + x`
# addend, absent here). Verified: bf16 log10_ with vectorization OPEN stays
# within bf16 quantization error (max rel ~0.4%) across [0,1), [1,10), [10,100)
# vs an fp32 reference; functional test 18/18 (incl. bf16) clean. OPEN gives
# ~1.07x dtype-equal speedup vs ~0.98x with CLOSED (isCloseVectorization=True),
# so OPEN is both correct and faster here.
import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

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
def log10_func(x):
    return tl.log(x.to(tl.float32)) * 0.4342944819032518


# Separate CLOSED-vectorization config for the out-of-place path: the OPEN
# config miscompiles int16-promoted inputs (ConvertTritonXPUToLLVM
# "size mismatch when packing elements for LLVM struct"), see
# test_log10_int_promotes_to_float.
config_closed_ = CodeGenConfig(
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


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_closed_)
@triton.jit
def log10_func_closed(x):
    return tl.log(x.to(tl.float32)) * 0.4342944819032518


def log10(A):
    logger.debug("GEMS_KUNLUNXIN LOG10")
    return log10_func_closed(A)


def log10_(A):
    logger.debug("GEMS_KUNLUNXIN LOG10_")
    log10_func(A, out0=A)
    return A


def log10_out(A, out):
    logger.debug("GEMS_KUNLUNXIN LOG10_OUT")
    log10_func(A, out0=out)
    return out
