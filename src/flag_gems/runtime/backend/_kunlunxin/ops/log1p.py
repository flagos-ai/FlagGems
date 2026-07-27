# Kunlunxin (XPU) override of log1p / log1p_.
#
# log1p was NOT overridden by kunlunxin, so it fell to the generic bare
# `pointwise_dynamic` (no CodeGenConfig) -> discrete access on XPU ->
# catastrophic latency (56ms for [4096,4096], gems speedup ~0.003 in
# harness/perf_ir_3/ir-log1p-dev3.log). log1p_ fell to the KernelGen
# hand-written fixed-BLOCK kernel (grid=cdiv(N,1024)), ~0.13-0.15 on large
# shapes (ir-log1p_-dev5.log).
#
# Fix: reuse the proven memory-bound unary recipe (like the sibling cos.py):
# the tuned CodeGenConfig (prefer_1d_tile, buffer_size_limit=4096,
# kunlunAutoGrid=True, unroll_num=8) on a pointwise_dynamic kernel; log1p_
# shares the kernel via out0=A. Kernel body (tl.log(1 + x_fp32)) unchanged.
#
# IMPORTANT (differs from cos.py): isCloseVectorization=True (vectorization
# CLOSED). With isCloseVectorization=False (vectorization OPEN, cos.py's value)
# the vectorized log MISCOMPILES bf16: ~1.6% of elements come out off by
# exactly +ln(2)=0.6931 -> 6 bf16 accuracy tests fail. Closing vectorization
# fixes correctness at negligible perf cost (log1p/log1p_ still ~0.87-0.89 avg,
# fp32 large shapes >=1.0). log(1+x) is more vectorization-sensitive than cos.
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
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def log1p_func(x):
    return tl.log(1.0 + x.to(tl.float32)).to(x.dtype)


def log1p(A):
    logger.debug("GEMS_KUNLUNXIN LOG1P")
    return log1p_func(A)


def log1p_(A):
    logger.debug("GEMS_KUNLUNXIN LOG1P_")
    log1p_func(A, out0=A)
    return A
