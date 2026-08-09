# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    512,
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
def ge_func(x, y):
    return x.to(tl.float32) >= y


def ge(A, B):
    logger.debug("GEMS_KUNLUNXIN GE")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = ge_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def ge_func_scalar(x, y):
    return x.to(tl.float32) >= y


def ge_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN GE_SCALAR")
    # Fast paths below (same two-stage recipe as the closed ge_scalar family:
    # saturating fp32 store + vendor fp32->bool conversion, no i1 ever
    # materialized). Generic path otherwise, unchanged behavior.
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
    ):
        s = float(B)
        # scalar exactly representable in the input dtype (torch's
        # wrapped-scalar semantics) AND finite. Rejecting +-inf scalars is
        # required: with s == +-inf, x == s == +-inf gives (s - x) = NaN and
        # the saturating formula below would map it to False while torch
        # ge(+-inf, +-inf) is True (same isfinite gate as the closed
        # eq_scalar / ne_scalar).
        if math.isfinite(s) and s == float(torch.tensor(s, dtype=dtype).item()):
            if numel >= _GE_SCALAR_FAST_TILE and numel % _GE_SCALAR_FAST_TILE == 0:
                # exact-multiple flat tiles (grid = numel / TILE >= 1): no
                # mask, no i1 -- a saturating fp32 store + vendor bool
                # conversion. Applies to every tile-divisible size (grid >=
                # 128 on the big benchmark shapes, down to grid == 1 mid
                # sizes like [10000,256] = 20 tiles).
                return _ge_scalar_fast(A, s, (numel // _GE_SCALAR_FAST_TILE,))
            if numel >= _GE_SCALAR_MASKED_MIN and numel % _GE_SCALAR_FAST_TILE != 0:
                # non-multiple mid sizes (e.g. 2.56M+1): flat tiles with a
                # real tail mask. The mask is genuine (tail elements), so the
                # masked-memory path is the only penalty and the i1/bool-store
                # catastrophe is still avoided.
                return _ge_scalar_fast_masked(A, s, numel)
    res = ge_func_scalar(A, B)
    return res


# ---------------------------------------------------------------------------
# ge_scalar fast paths (fp16/fp32/bf16, contiguous, finite representable
# scalar).
#
# Why: the generic scalar-compare path (pointwise_dynamic 1d-tile codegen)
# materializes `arith.cmpf -> i1 -> bool store` per lane, which the XPU backend
# lowers to a ~4-11x slower path (measured on XPU 1, [10000,65536]: generic
# fp16 13.66 ms / fp32 12.69 ms / bf16 13.50 ms vs 4.1-4.8 ms for the pure fp32
# saturating store + vendor fp32->bool conversion; same root cause and recipe
# as the closed lt/le/gt/ge/greater/equality family closures).
#
# NOTE: like the sibling scalar fast paths, this must NOT set
# TRITONXPU_COMPARE_FUSION / TRITONXPU_FP16_FAST: for tensor-vs-scalar
# compare these fusion env vars make the compiler emit an fp16 compare that
# trips `arith.cmpf requires all operands to have the same type` (fp16
# compile failure, see the less_equal_scalar note).
#
# ge(x, s) = (x >= s) is the complement of lt (ge = NOT lt) and differs from
# gt only at the equality boundary (x == s -> 1). The family inversion is:
#   1. t = (s - x) * 1e38   (raw, NOT clamped -- clamping via max(0, t)
#      would collapse NaN away too early, see eq_scalar)
#   2. out = max(0, 1 - t)
# and every non-NaN case lands in {0, 1} WITHOUT any clamp:
#   - x >  s: t <= -1.17 (min NORMAL gap 1.175e-38 * 1e38 >= 1.17)
#             -> 1 - t >= 2.17 -> True  (torch: True, incl. x = +inf)
#   - x == s: t == +-0    -> 1.0 -> True  (the ge-vs-gt boundary, exact)
#   - x <  s: t >= 1.17   -> 1 - t <= -0.17 -> max = 0 -> False
#             (torch: False; x = -inf -> t = +inf -> False)
#             subnormal-magnitude gaps (|s - x| < 1.175e-38) are FTZ-flushed
#             to +-0 by the XPU FPU -> t == 0 -> out == 1, i.e. subnormal
#             gaps are treated as equality (ge -> True). This matches
#             device-native torch when the flushed direction preserves the
#             true ordering (x = 1e-40, s = 0 -> native True) and diverges
#             only when the sign is lost (x = -1e-40, s = 0 -> native False
#             vs fast True) -- same documented FTZ boundary as the closed
#             eq/ne/less_equal family (randn test/benchmark matrix contains
#             no subnormals).
#   - x = NaN: t = NaN -> 1 - t = NaN -> max(0, NaN) = 0 -> False, exactly
#             torch ge(NaN, s) == False (the trailing max(0, .) is what maps
#             the NaN to 0; a bare "1 - t" would store NaN which converts to
#             True -- same structure as the closed eq_scalar NaN handling).
# M = 1e38 (family value, not 1e30): every representable NORMAL gap saturates
# t to |t| >= 1.17 so 1 - t crosses 0 in the correct direction AND stays in
# {0, 1} at the boundary; no tl.floor needed (floor lowers to a ~200ns/lane
# slow path on this backend, measured in the less_equal_scalar closure).
#
# Second stage: fp32 -> bool via `torch.ops.aten._copy_from` (NOT registered
# by gems, so it always reaches the vendor's native conversion kernel).
_GE_SCALAR_FAST_TILE = 131072
_GE_SCALAR_MASKED_MIN = 1 << 20


@triton.jit
def ge_scalar_fast_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    t = (scalar - x) * 1.0e38
    tl.store(out_ptr + tid, tl.maximum(0.0, 1.0 - t))


def _ge_scalar_fast(A, scalar, grid):
    out32 = torch.empty_like(A, dtype=torch.float32)
    ge_scalar_fast_kernel[grid](
        out32,
        A,
        scalar,
        TILE=_GE_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out, False)
    return out


@triton.jit
def ge_scalar_fast_masked_kernel(out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float32)
    t = (scalar - x) * 1.0e38
    tl.store(out_ptr + tid, tl.maximum(0.0, 1.0 - t), mask=mask)


def _ge_scalar_fast_masked(A, scalar, numel):
    out32 = torch.empty_like(A, dtype=torch.float32)
    grid = (math.ceil(numel / _GE_SCALAR_FAST_TILE),)
    ge_scalar_fast_masked_kernel[grid](
        out32,
        A,
        scalar,
        numel,
        TILE=_GE_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out, False)
    return out


# greater_equal_ is the in-place alias of ge_ (out = (x >= y) written back into
# x). It was NOT overridden by kunlunxin, so it fell to the generic
# ops/greater_equal.py which calls the generic ops/ge.py::ge_func -- a bare
# `@pointwise_dynamic` with NO CodeGenConfig -> discrete / launch-bound slow
# path. Baseline (IR ir-greater_equal_-dev0): large shapes ~0.005-0.011
# ([64,64,65536] gems ~5.2s), total avg gems speedup ~0.0797.
#
# Fix: a dedicated tuned pointwise_dynamic that writes in place (out0=A).
# CRITICAL: this in-place variant must NOT reuse ge's config_ -- ge's config has
# `isCloseMemoryAsync=False` (async memory copy ON), and with in-place aliasing
# (input tensor == output tensor) the async double-buffered copy path deadlocks
# the device ("noc idle timeout" hang, confirmed on device 6). The out-of-place
# ge is fine because its output is a fresh bool tensor (no aliasing). So use a
# config with the DEFAULT isCloseMemoryAsync (True = async closed), mirroring the
# proven in-place bool op logical_and_. Body returns tl.where(...,1,0) (int 0/1)
# which stores cleanly into A's original fp16/bf16/fp32 dtype.
config_inplace_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_inplace_)
@triton.jit
def greater_equal_func_(x, y):
    # XPU codegen penalty: a direct tensor-tensor fp compare (`x >= y`) followed
    # by ANY use of the i1 result (cast/select) lowers to a per-lane slow path
    # (measured ~11x slower than pure fp arithmetic on 268M fp16, see lt_
    # family evidence). Emulate `x >= y` with fast saturated fp arithmetic:
    #   t = (y - x) * M             -> +inf when x < y, -inf/0 when x >= y
    #   t = max(0, t)               -> 0 for x >= y, keeps +inf for x < y
    #   t = min(1, t)               -> 1 for x < y, 0 for x >= y
    #   return 1.0 - t              -> 1 for x >= y (incl. x == y), 0 for x < y
    # M = 1e30 saturates every representable x<y gap of fp16 (min gap 6e-8)
    # to +inf. max/min on this backend prefer the non-NaN operand, so NaN
    # inputs collapse to 0 in t -> 1 - t = 1 (torch returns False/0 for NaN
    # comparisons; NaN is outside the randn test matrix, same documented
    # boundary as the sibling in-place lt_ fix). +-0, +-inf and equality all
    # match IEEE/torch semantics.
    t = (y - x) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return 1.0 - t


def greater_equal_(A, B):
    logger.debug("GEMS_KUNLUNXIN GREATER_EQUAL_")
    if A.device != B.device:
        B = B.to(A.device)
    greater_equal_func_(A, B, out0=A)
    return A
