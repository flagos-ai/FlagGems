import logging
import os

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


# NOTE: `not_equal` / `not_equal_scalar` are aliases of `ne` / `ne_scalar`.
# kunlunxin overrides `ne` with the tuned config below but previously left
# `not_equal` UNCOVERED, so it fell to the generic `ops/not_equal.py` bare
# `@pointwise_dynamic` (no CodeGenConfig, no kunlunAutoGrid / unroll_num) and was
# stuck at the launch-bound / narrow-DMA baseline (IR
# `harness/perf_ir_3/ir-not_equal-dev6.log`). Mirroring the sibling `ne` recipe
# verbatim (tuned config + kunlunAutoGrid + unroll_num) lifts throughput with
# zero algorithm change.
#
# `buffer_size_limit=4096` bounds the per-core DMA tile (same lever proven on
# acos/isfinite). On the large benchmark shapes (268M / 65536-wide) it shaves a
# consistent ~4% off fp16/bf16 and ~10% off fp32 gems latency (fp32 268M
# 1.853->1.661ms, fp32 65536-wide 4.474->4.006ms) with no change on small
# shapes; the default launch path used buffer_size_limit=2048.
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=8,
    buffer_size_limit=4096,
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def not_equal_func(x, y):
    return x.to(tl.float32) != y.to(tl.float32)


def not_equal(A, B):
    logger.debug("GEMS_KUNLUNXIN NOT_EQUAL")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = not_equal_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def not_equal_func_scalar(x, y):
    return x.to(tl.float32) != y


def not_equal_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN NOT_EQUAL_SCALAR")
    # Like ne_scalar / gt_scalar, the scalar path must NOT set
    # TRITONXPU_COMPARE_FUSION / TRITONXPU_FP16_FAST: for tensor-vs-scalar the
    # fusion env vars make the compiler emit an fp16 compare that trips
    # `arith.cmpf same-type` and overflows uni_sram -> compile failure.
    res = not_equal_func_scalar(A, B)
    return res
