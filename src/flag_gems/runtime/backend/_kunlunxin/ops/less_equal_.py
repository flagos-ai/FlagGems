# Kunlunxin (XPU) override of less_equal_ / less_equal_scalar_ (in-place).
#
# `less_equal_.Tensor` is the in-place variant of `less_equal.Tensor`, and
# kunlunxin already ships a tuned override for the out-of-place `less_equal`
# (`_kunlunxin/ops/less_equal.py`, itself reusing the `le.py` recipe). But
# `less_equal_` was NOT overridden, so it fell back to the generic bare
# `pointwise_dynamic` (no CodeGenConfig) -> discrete access on XPU ->
# catastrophic latency (4096x4096 fp32: gems 50.8ms vs torch 0.128ms,
# speedup 0.003; dtype-balanced 0.082, 2026-09-03 baseline).
#
# Fix: reuse the tuned CodeGenConfig (block=1024, unroll_num=8,
# kunlunAutoGrid=True, prefer_1d_tile=True) plus the
# TRITONXPU_COMPARE_FUSION / TRITONXPU_FP16_FAST launch env vars for the tensor
# path; in-place write-back via `out0=A`. Kernel body uses an arithmetic-only
# formulation of `x <= y` (see less_equal_func) because every TritonXPU
# select/compare path is capped at ~60 Gelem/s by the vselect mask assembly
# (dtype-balanced 0.082 -> 0.509 -> 0.949, 2026-09-03).
import logging
import os

import torch
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


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def less_equal_func(x, y):
    # x <= y <=> !(y - x < 0). Arithmetic form, avoiding the TritonXPU vselect mask
    # assembly: VSelectOpConversion assembles the predicate mask bit by bit (~16
    # select+or per 32 elements), so any select/compare path is bandwidth-capped at
    # ~60 Gelem/s (fp16 where 365GB/s, fp32 710GB/s, measured 2026-09-03); min/max/clip
    # are pure VALU instructions and run close to full copy speed (fp16 1588GB/s).
    # This formula uses only fast-class instructions:
    #   d   = y - x
    #   neg = min(d, 0)     # d when d<0, else 0
    #   mag = max(-neg, 0)  # -d when d<0, else 0
    #   r   = 1 - min(mag * 1e38, 1)
    # d<0 -> mag>0 -> mag*1e38 overflows to inf -> min yields 1 -> r=0; d>=0 -> r=1.
    # ±0 is correct (d=±0 -> mag=0 -> r=1). NaN behavior matches the where version
    # (for non 0/1 inputs). The 1e38 multiplier must stay f32 (under f16, 0*inf=NaN
    # breaks the d=0 branch).
    d = y - x
    neg = tl.minimum(d, 0.0)
    mag = tl.maximum(-neg, 0.0)
    return 1.0 - tl.minimum(mag * 1e38, 1.0)


# DEFAULT-promotion twin of less_equal_func for the in-place integral path:
# ALWAYS_BOOL would store a bool into int8/uint8 (miscomputes on uint8), and
# the int8 `y - x` cannot be selected by the XPU backend, so the integral path
# runs this on fp32 and casts the 0/1 result back to the tensor dtype.
@pointwise_dynamic(
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def less_equal_func_default(x, y):
    d = y - x
    neg = tl.minimum(d, 0.0)
    mag = tl.maximum(-neg, 0.0)
    return 1.0 - tl.minimum(mag * 1e38, 1.0)


def less_equal_(A, B):
    logger.debug("GEMS_KUNLUNXIN LESS_EQUAL_")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    try:
        if not A.dtype.is_floating_point:
            # torch keeps the tensor dtype for an in-place integral compare, so
            # the 0/1 result must land back in int8/uint8. Run the arithmetic on
            # fp32 (int8 `y - x` is not selectable on this backend) with the
            # DEFAULT-promotion twin, then cast back.
            orig_dtype = A.dtype
            a32 = A.to(torch.float32)
            b32 = B.to(torch.float32) if isinstance(B, torch.Tensor) else B
            less_equal_func_default(a32, b32, out0=a32)
            A.copy_(a32.to(orig_dtype))
            return A
        less_equal_func(A, B, out0=A)
    finally:
        del os.environ["TRITONXPU_COMPARE_FUSION"]
        del os.environ["TRITONXPU_FP16_FAST"]
    return A


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def less_equal_func_scalar(x, y):
    # Same arithmetic form as the tensor path (see the less_equal_func comment).
    d = y - x
    neg = tl.minimum(d, 0.0)
    mag = tl.maximum(-neg, 0.0)
    return 1.0 - tl.minimum(mag * 1e38, 1.0)


def less_equal_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN LESS_EQUAL_SCALAR_")
    # NOTE: unlike the tensor path, the scalar path must NOT set
    # TRITONXPU_COMPARE_FUSION / TRITONXPU_FP16_FAST. For tensor-vs-scalar
    # compare these fusion env vars make the compiler emit an fp16 compare that
    # trips `arith.cmpf requires all operands to have the same type` and blows
    # the uni_sram budget -> `out of resource: uni_sram` compile failure (fp16).
    # The sibling le_scalar / gt_scalar / less_equal_scalar deliberately omit
    # them for the same reason.
    less_equal_func_scalar(A, B, out0=A)
    return A
