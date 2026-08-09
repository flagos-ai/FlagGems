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
def gt_func(x, y):
    return x.to(tl.float32) > y


def gt(A, B):
    logger.debug("GEMS_KUNLUNXIN GT")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = gt_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def gt_func_scalar(x, y):
    return x.to(tl.float32) > y


def gt_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN GT_SCALAR")
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and float(B) == float(torch.tensor(float(B), dtype=dtype).item())
    ):
        if numel >= _GT_SCALAR_FAST_TILE * _GT_SCALAR_MIN_GRID and numel % _GT_SCALAR_FAST_TILE == 0:
            # exact-multiple flat tiles (grid = numel / TILE >= MIN_GRID): no
            # mask, no i1 -- a saturating fp32 store + vendor bool conversion.
            return _gt_scalar_fast(A, float(B), (numel // _GT_SCALAR_FAST_TILE,))
        if numel >= _GT_SCALAR_MASKED_MIN and numel % _GT_SCALAR_FAST_TILE != 0:
            # non-multiple mid sizes (e.g. 2.56M): flat tiles with a real tail
            # mask. The mask is genuine (tail elements), so the masked-memory
            # path is the only penalty and the i1/bool-store catastrophe is
            # still avoided.
            return _gt_scalar_fast_masked(A, float(B), numel)
    res = gt_func_scalar(A, B)
    return res


# ---------------------------------------------------------------------------
# gt_scalar fast paths (fp16/fp32/bf16, contiguous).
#
# Why: the generic scalar-compare path (pointwise_dynamic 1d-tile codegen)
# always emits `mask = tid < num_tasks` and materializes
# `arith.cmpf -> i1 -> bool store` per lane. On XPU both are slow: the
# always-true runtime mask goes through the masked-memory path, and the
# i1/bool store lowers to a per-lane slow path (~10x). Measured on XPU 5
# ([10000, 65536]): generic fp16 13.67 ms / fp32 12.69 ms / bf16 13.51 ms;
# a pure fp32 batched-store of the saturating result is only 2.16-3.04 ms.
#
# Strategy -- same two-stage recipe as the greater_scalar/lt_scalar family;
# no i1 is ever materialized in Triton:
#   1. saturating fp arithmetic on the fp32-upcast value:
#      t = (x - s) * 1e30; max(0,t); min(1,t) -> exactly 0.0/1.0 written into a
#      fp32 buffer (M = 1e30 saturates every representable x != s gap of
#      fp16/bf16/fp32 to +-inf; eq/gap exact). NaN -> 0 (max/min prefer the
#      non-NaN operand; torch: NaN > s == False). +-0, +-inf, subnormal gaps
#      are exact.
#   2. fp32 -> bool via `torch.ops.aten._copy_from` (NOT registered by gems,
#      so it always reaches the vendor's native conversion kernel; measured
#      ~1.97 ms on [10000,65536] fp16, vs the generic path's 13.67 ms).
#
# The scalar gate `float(B) == float(torch.tensor(B, dtype=A.dtype).item())`
# only admits scalars exactly representable in A.dtype: torch compares against
# the scalar rounded to the input dtype (wrapped scalar), and restricting to
# representable scalars (benchmark 0.5, test 0) makes the fp32 compare
# bit-identical to torch semantics. Anything else (e.g. fp16 scalar 1e-10,
# 0.1 non-representable) keeps the generic path, unchanged behavior.
#
# fp16/bf16 intermediate buffers were probed and rejected: the vendor's
# fp16/bf16 -> bool conversion is ~3x slower than fp32 -> bool, so the fp32
# intermediate wins despite the extra bytes.
_GT_SCALAR_FAST_TILE = 131072
_GT_SCALAR_MIN_GRID = 128
_GT_SCALAR_MASKED_MIN = 1 << 20


@triton.jit
def gt_scalar_fast_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    t = (x - scalar) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(out_ptr + tid, t)


def _gt_scalar_fast(A, scalar, grid):
    out32 = torch.empty_like(A, dtype=torch.float32)
    gt_scalar_fast_kernel[grid](
        out32,
        A,
        scalar,
        TILE=_GT_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out, False)
    return out


@triton.jit
def gt_scalar_fast_masked_kernel(out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float32)
    t = (x - scalar) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(out_ptr + tid, t, mask=mask)


def _gt_scalar_fast_masked(A, scalar, numel):
    out32 = torch.empty_like(A, dtype=torch.float32)
    grid = (math.ceil(numel / _GT_SCALAR_FAST_TILE),)
    gt_scalar_fast_masked_kernel[grid](
        out32,
        A,
        scalar,
        numel,
        TILE=_GT_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out, False)
    return out


# ---------------------------------------------------------------------------
# gt_scalar_ (in-place alias of gt.Scalar, e.g. `x.gt_(0.5)` on a float
# tensor). torch keeps the input dtype and stores 0.0/1.0 back into x.
#
# Baseline (XPU 2, 2026-08-16): the generic `gt_func_scalar(A, B, out0=A)`
# path (ALWAYS_BOOL; `cmf -> i1 -> bool` per-lane) measured 0.2676x
# dtype-equal-weight (fp16 0.2382 / fp32 0.3099 / bf16 0.2548); big shapes
# were catastrophic ([10000,65536] fp16 0.066x, [268435456] fp16 0.067x).
#
# Fix (same recipe as the sibling in-place lt_/ge_ family, 0.2683x -> 0.9012x):
#   1. generic in-place kernel with DEFAULT promotion + saturating fp
#      arithmetic (no i1 materialized), under the in-place-safe CodeGenConfig
#      below (config_ would open the async-copy path that deadlocks with
#      in-place aliasing, see lt.py config_inplace_ note).
#   2. unmasked flat-tile in-place fast kernel for fp16/fp32 contiguous
#      tensors whose numel is an exact multiple of TILE (grid >= MIN_GRID),
#      restoring the fast DMA path the always-true runtime mask blocks.
#   3. both bodies use the saturating identity `gt(x, s) == min(1, max(0,
#      (x - s) * K))` with K = 1e32 * 1e32 = 1e64, computed in fp32. For any
#      representable nonzero gap (down to the fp32 subnormal 2^-149 = 1.4e-45
#      and the bf16 subnormal 2^-133), `gap * 1e64` >= 1.4e19 -- safely above
#      the 1 -- while the single 1e30 multiplier used by the lt_ family leaves
#      a hole for sub-gap ~1e-30 differences. Guarded to scalars exactly
#      representable in A.dtype (`float(B) == wrapped`), making the fp32
#      compare bit-identical to torch's wrapped-scalar semantics; anything
#      else (non-float dtype, non-representable scalar, non-contiguous, tiny
#      shapes) keeps the previous ALWAYS_BOOL generic path, behavior
#      unchanged.
config_inplace_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_inplace_,
)
@triton.jit
def gt_func_scalar_inplace(x, y):
    t = (x.to(tl.float32) - y) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return t


# gt_tensor_ (in-place alias of gt.Tensor, e.g. `x.gt_(y)` on a float
# tensor). torch keeps the input dtype and stores True (1.0) / False (0.0)
# back into x.
#
# Baseline (XPU 3, 2026-08-16): the generic `gt_func(A, B, out0=A)` path
# (promotion ALWAYS_BOOL -- `cmp -> i1 -> bool` per lane, under
# TRITONXPU_COMPARE_FUSION/FP16_FAST) measured 0.3343x kernel-mode and
# 0.1089x operator-mode dtype-equal-weight; big shapes were stuck at
# ~0.088-0.18x ([1024,65536] fp16 gems 2.67ms vs torch 0.24ms).
#
# Fix (same single-kernel saturating-fp recipe as the committed in-place
# lt_/ge_ family and gt_scalar_ above, 0i1 materialized): a tensor-tensor
# fp compare (`x > y`) followed by ANY use of the i1 result lowers to a
# per-lane slow path on this backend (~11x slower than pure fp arithmetic
# on 268M fp16, lt_ evidence); a tensor-tensor subtraction + fp-constant
# math stays on the fast vector channel. Emulate `x > y` with fast
# saturated fp arithmetic only:
#   t = (x - y) * 1e32 * 1e32    -> for x > y saturates to +-inf-scale, else -scale
#   t = max(0, t)                -> 0 for x <= y, keeps the +scale for x > y
#   t = min(1, t)                -> 1 for x > y, 0 for x <= y
# The two-stage 1e32*1e32 (= 1e64 effective) is the same factor family as
# the closed gt_scalar_ in-place fast path: every representable fp32/bf16/
# fp16 gap (down to the fp32/bf16 subnormals) saturates to a value >= 1,
# closing the sub-gap hole the single 1e30 multiplier leaves. max/min on
# this backend prefer the non-NaN operand, so NaN (x - y = NaN) collapses
# to 0, matching torch gt semantics; +-0, +-inf and equality all match
# IEEE/torch. The slow i1/bool path is never materialized.
#
# The config is config_inplace_ (DEFAULT isCloseMemoryAsync = async copy
# closed): reusing gt's config_ (isCloseMemoryAsync=False) with in-place
# aliasing is the documented "noc idle timeout" deadlock, same note as
# lt.py / gt_scalar_ above.
@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_inplace_)
@triton.jit
def gt_func_tensor_inplace(x, y):
    t = (x.to(tl.float32) - y) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return t


def gt_tensor_(A, B):
    logger.debug("GEMS_KUNLUNXIN GT_ TENSOR")
    if A.device != B.device:
        B = B.to(A.device)
    gt_func_tensor_inplace(A, B, out0=A)
    return A


def gt_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN GT_ SCALAR")
    numel = A.numel()
    if (
        A.is_contiguous()
        and A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and float(B) == float(torch.tensor(float(B), dtype=A.dtype).item())
    ):
        if (
            A.dtype in (torch.float16, torch.float32)
            and numel >= _GT_SCALAR_INPLACE_FAST_TILE * _GT_SCALAR_INPLACE_MIN_GRID
            and numel % _GT_SCALAR_INPLACE_FAST_TILE == 0
        ):
            # exact-multiple flat tiles: no mask at all; grid fixed.
            return _gt_scalar_inplace_fast(A, float(B))
        gt_func_scalar_inplace(A, B, out0=A)
        return A
    return gt_func_scalar(A, B, out0=A)


# in-place alias safety: the fast kernel writes into the SAME tensor it reads,
# so it must keep the DEFAULT isCloseMemoryAsync (True = async copy closed);
# passing False with in-place aliasing is the documented "noc idle timeout"
# deadlock, same as lt.py's lt_scalar_inplace_fast_kernel.
_GT_SCALAR_INPLACE_FAST_TILE = 131072
_GT_SCALAR_INPLACE_MIN_GRID = 128


@triton.jit
def gt_scalar_inplace_fast_kernel(x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    t = (x.to(tl.float32) - scalar) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, t)


def _gt_scalar_inplace_fast(A, scalar):
    grid = (A.numel() // _GT_SCALAR_INPLACE_FAST_TILE,)
    gt_scalar_inplace_fast_kernel[grid](
        A,
        scalar,
        TILE=_GT_SCALAR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A
