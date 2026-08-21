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
#
# `less_equal_scalar` (2026-08-13): added the family two-stage fast path
# (saturating fp32 arithmetic + vendor fp32->bool `_copy_from`, no i1), same
# recipe as the closed `le_scalar` (le(x, s) == less_equal(x, s) == x <= s).
# See the fast-path block below for the M = 1e38 rationale (inverse of gt).
import logging
import math
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
    config=config_,
)
@triton.jit
def less_equal_func_scalar(x, y):
    return x.to(tl.float32) <= y


def less_equal_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN LESS_EQUAL_SCALAR")
    # Fast paths below (same two-stage recipe as the closed le_scalar family:
    # saturating fp32 store + vendor fp32->bool conversion, no i1 ever
    # materialized). Generic path otherwise, unchanged behavior.
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and float(B) == float(torch.tensor(float(B), dtype=dtype).item())
    ):
        if numel >= _LESS_EQUAL_SCALAR_FAST_TILE and numel % _LESS_EQUAL_SCALAR_FAST_TILE == 0:
            # exact-multiple flat tiles (grid = numel / TILE >= 1): no mask, no
            # i1 -- a saturating fp32 store + vendor bool conversion. Applies
            # to every tile-divisible size (grid >= 128 on the big benchmark
            # shapes, down to grid == 1 mid sizes like [10000,256] = 20 tiles).
            return _less_equal_scalar_fast(A, float(B), (numel // _LESS_EQUAL_SCALAR_FAST_TILE,))
        if numel >= _LESS_EQUAL_SCALAR_MASKED_MIN and numel % _LESS_EQUAL_SCALAR_FAST_TILE != 0:
            # non-multiple mid sizes (e.g. 2.56M+1): flat tiles with a real
            # tail mask. The mask is genuine (tail elements), so the
            # masked-memory path is the only penalty and the i1/bool-store
            # catastrophe is still avoided.
            return _less_equal_scalar_fast_masked(A, float(B), numel)
    res = less_equal_func_scalar(A, B)
    return res


# ---------------------------------------------------------------------------
# less_equal_scalar fast paths (fp16/fp32/bf16, contiguous).
#
# Why: the generic scalar-compare path (pointwise_dynamic 1d-tile codegen)
# materializes `arith.cmpf -> i1 -> bool store` per lane, which the XPU backend
# lowers to a ~4-11x slower path (measured on XPU 2, [10000,65536]: generic
# fp16 13.66 ms / fp32 12.69 ms / bf16 13.52 ms vs 4.1-4.8 ms for the pure fp32
# saturating store + vendor fp32->bool conversion; same root cause and recipe
# as the closed le/lt/gt/ge/greater family closures).
#
# NOTE: unlike the tensor path, the scalar fast path must NOT set
# TRITONXPU_COMPARE_FUSION / TRITONXPU_FP16_FAST. For tensor-vs-scalar
# compare these fusion env vars make the compiler emit an fp16 compare that
# trips `arith.cmpf requires all operands to have the same type` and blows
# the uni_sram budget -> `out of resource: uni_sram` compile failure (fp16).
# The sibling le_scalar / gt_scalar deliberately omit them for the same
# reason.
#
# less_equal(x, s) is the INVERSE of greater(x, s) (le = NOT gt), so the family
# saturating expression is negated exactly once:
#   1. t = (x - s) * 1e38; max(0,t); min(1,t)  -> 1.0 when x > s, else 0.0
#      (M = 1e38 saturates EVERY representable NORMAL fp32 gap: min normal
#      1.175e-38 * 1e38 >= 1.17 -> clamps to 1, so no (0,1) leakage; eq and
#      +-0 give +-0 -> 0; subnormal gaps FTZ-flush to 0 on this backend ->
#      t = 0 -> le = 1, matching device-native torch FTZ compare semantics)
#   2. le = 1.0 - t                            -> EXACTLY 1.0 when x <= s,
#      0.0 when x > s (fp32 values 0.0/1.0, then bool conversion)
#
# Why M = 1e38 instead of the family 1e30: less_equal inverts gt, so the output
# must be exactly {0, 1} -- with M = 1e30 a tiny normal gap (e.g. x = 1e-31,
# s = 0) gives t in (0, 1) and 1 - t in (0, 1) which converts to True
# (diverging from torch). Collapsing via tl.floor works numerically but
# lowers to a ~200ns/lane slow path on this backend (10x regression, measured
# [10000,65536] fp16: 13.7ms -> 154ms). M = 1e38 keeps everything in fast fp
# arithmetic: every NORMAL gap saturates to >= 1 -> t = 1 exactly, and every
# SUBNORMAL gap is flushed to 0 by the FPU -> t = 0 -> le = 1, which equals
# device-native le (native torch treats x = 1e-40, s = 0 as True; verified
# on device).
#   written into a fp32 buffer, then fp32 -> bool via
#   `torch.ops.aten._copy_from` (NOT registered by gems, so it reaches the
#   vendor's native conversion kernel under use_gems).
# NaN inputs: (x - s) = NaN -> max/min on this backend prefer the non-NaN
# operand -> t collapses to 0 -> le = 1, whereas torch NaN <= s == False. Same
# documented boundary as the closed ge_/ge_scalar/gt family (NaN is outside the
# randn test/benchmark matrix; corner 对拍 below records this as the only
# divergence). +-0, equality, +-inf, subnormal gaps verified exact vs
# device-native torch.
#
# The scalar gate `float(B) == float(torch.tensor(B, dtype=A.dtype).item())`
# only admits scalars exactly representable in A.dtype (benchmark scalar 0,
# test scalar 0): torch compares against the scalar rounded to the input
# dtype, and restricting to representable scalars keeps the fp32 compare
# bit-identical. Anything else keeps the generic path, unchanged behavior.
_LESS_EQUAL_SCALAR_FAST_TILE = 131072
_LESS_EQUAL_SCALAR_MASKED_MIN = 1 << 20


@triton.jit
def less_equal_scalar_fast_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    t = (x - scalar) * 1.0e38
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    # less_equal is the inverse of greater, so the final result must be
    # EXACTLY {0, 1}. The family M (1e30) leaves a (0, 1) leakage for tiny
    # normal gaps (e.g. x = 1e-31 with s = 0: 1e-30 * 1e30 = 0.1 -> 1 - t in
    # (0, 1)), which would convert to True. M = 1e38 saturates every
    # representable NORMAL fp32 gap (min normal 1.175e-38 * 1e38 >= 1.17 ->
    # clamps to 1), so 1 - t is exactly {0, 1}. Subnormal gaps are FTZ-flushed
    # by the XPU hardware (x - s -> +-0 when a subnormal is involved),
    # yielding t = 0 -> le = 1, which matches device-native torch (verified:
    # native le treats x = 1e-40, s = 0 as True). M = 1e30/floor(x) would need
    # a tl.floor, which lowers to a ~200ns/lane slow path on this backend
    # (~10x total); M = 1e38 keeps the whole body in fast fp arithmetic.
    tl.store(out_ptr + tid, 1.0 - t)


def _less_equal_scalar_fast(A, scalar, grid):
    out32 = torch.empty_like(A, dtype=torch.float32)
    less_equal_scalar_fast_kernel[grid](
        out32,
        A,
        scalar,
        TILE=_LESS_EQUAL_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out, False)
    return out


@triton.jit
def less_equal_scalar_fast_masked_kernel(out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float32)
    t = (x - scalar) * 1.0e38
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    # see less_equal_scalar_fast_kernel: M = 1e38 keeps 1 - t exactly in
    # {0, 1}.
    tl.store(out_ptr + tid, 1.0 - t, mask=mask)


def _less_equal_scalar_fast_masked(A, scalar, numel):
    out32 = torch.empty_like(A, dtype=torch.float32)
    grid = (math.ceil(numel / _LESS_EQUAL_SCALAR_FAST_TILE),)
    less_equal_scalar_fast_masked_kernel[grid](
        out32,
        A,
        scalar,
        numel,
        TILE=_LESS_EQUAL_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out, False)
    return out
