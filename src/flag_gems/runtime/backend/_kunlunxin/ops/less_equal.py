# Kunlunxin (XPU) override of less_equal / less_equal_scalar.
#
# `less_equal.Tensor` is functionally identical to `le.Tensor`, and kunlunxin
# already ships a tuned override for le (`_kunlunxin/ops/le.py`). But
# `less_equal` was NOT overridden, so it fell back to the generic bare
# `pointwise_dynamic` (no CodeGenConfig) -> discrete access on XPU ->
# catastrophic latency (see `harness/perf_ir_3/ir-less_equal-dev1.log`, the
# kernel is `less_equal_func_kernel` generated from `ops/less_equal.py`).
#
# Fix: reuse the exact le recipe -- same tuned CodeGenConfig
# (block=1024, unroll_num=8, kunlunAutoGrid=True, prefer_1d_tile=True) plus the
# TRITONXPU_COMPARE_FUSION / TRITONXPU_FP16_FAST launch env vars for the tensor
# path. Kernel body / algorithm unchanged (zero correctness risk).
import logging
import os

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


config_ = CodeGenConfig(
    1024,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)

# Scalar (tensor-vs-scalar) path config. Same shape recipe as config_, but
# with unroll_num=16 + buffer_size_limit=8192 (the greater.py config_scalar
# sweet spot). On the OLD fp32-promotion kernel body these knobs were inert
# (the kernel was stuck on the slow non-fused load+compare pipeline at
# ~135-260 GB/s regardless); once the body compares in the input dtype (see
# less_equal_func_scalar below) and the fusion env vars fire, the sweep shows
# unroll_num=16 + buffer_size_limit=8192 lifts the big shapes a further
# ~1.2-1.25x (fp32 [268435456] 1.17 -> 0.96 ms, ~1.4 TB/s, on par with the
# tensor-tensor kernel), while unroll 8 leaves ~20% on the table.
config_scalar = CodeGenConfig(
    1024,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=16,
    buffer_size_limit=8192,
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def less_equal_func(x, y):
    return x.to(tl.float32) <= y


def less_equal(A, B):
    logger.debug("GEMS_KUNLUNXIN LESS_EQUAL")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = less_equal_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_scalar,
)
@triton.jit
def less_equal_func_scalar(x, y):
    # Compare in the input dtype after casting the scalar, i.e. exactly ATen's
    # tensor-vs-scalar type promotion (a wrapped python scalar participates as
    # the tensor's dtype). This is ALSO what unlocks the XPU fast path: with
    # the previous `x.to(tl.float32) <= y` body the scalar kernel was stuck on
    # the slow non-fused load+compare pipeline (~135-260 GB/s, 4-11x slower
    # than the tensor-tensor kernel), because
    #   1. TRITONXPU_COMPARE_FUSION only fires for same-type operand pairs --
    #      with `val0` arriving as an fp32 kernel argument and the load side
    #      fp16/bf16, the TritonXPUDtypeConvert pass trips `arith.cmpf
    #      requires all operands to have the same type` and then
    #      `out of resource: uni_sram` for fp16 (measured, fresh cache);
    #      COMPARE_FUSION alone (no FP16_FAST) does nothing for the scalar
    #      kernel.
    #   2. Casting `y` to `x.dtype` makes both compare operands the same type,
    #      so the fusion env vars compile cleanly for fp16/bf16/fp32 and the
    #      kernel reaches ~0.9-1.45 TB/s (fp16 [268435456]: 5.59ms -> 0.68ms,
    #      then 0.56ms with config_scalar).
    # Semantics are unchanged vs ATen (CPU ref agrees on edge values
    # +-inf/NaN/-0/overflow and on randn matrices, all dtypes).
    return x <= y.to(x.dtype)


def less_equal_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN LESS_EQUAL_SCALAR")
    # NOTE (updated 2026-09-10): the scalar path DOES set the fusion env vars
    # now. The old comment claimed they must be omitted because fp16 hit
    # `out of resource: uni_sram`; that failure was caused by the mixed-type
    # fp32-compare body above, not by the env vars themselves. With the
    # dtype-cast body (`y.to(x.dtype)`) the same env vars as the tensor path
    # compile cleanly and are what unlocks the fast fused compare (see probe
    # history in harness/solution/performance/less_equal_perf_fix.md).
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = less_equal_func_scalar(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res
