# Kunlunxin (XPU) override of lift_fresh_copy.
#
# lift_fresh_copy is semantically a bulk memory copy (fresh clone). It was NOT
# overridden by kunlunxin, so it fell to the generic KernelGen hand-written
# fixed-BLOCK=1024 copy kernel (grid=cdiv(N,1024)) -> launch-bound / untuned on
# XPU -> large-shape gems speedup ~0.04-0.13, several first-block anomalies
# (harness/perf_ir_3/ir-lift_fresh_copy-dev1.log).
#
# Fix: reuse the proven copy-family recipe (identical to the sibling
# alias_copy.py): a tuned pointwise_dynamic copy (`return x`) with the
# memory-bound CodeGenConfig (prefer_1d_tile, buffer_size_limit=4096,
# isCloseVectorization=False, kunlunAutoGrid=True, unroll_num=8). A pure copy
# has no math, so vectorization-open is safe (unlike log1p) and maximizes
# bandwidth.
import logging

import torch
import triton  # noqa: F401
import triton.language as tl  # noqa: F401
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

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


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def lift_fresh_copy_func(x):
    return x


def lift_fresh_copy(A):
    logger.debug("GEMS_KUNLUNXIN LIFT_FRESH_COPY")
    if A.numel() == 0:
        return torch.empty_like(A)
    return lift_fresh_copy_func(A)
