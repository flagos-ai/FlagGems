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

from flag_gems.ops.eq_ import (
    eq_ as _generic_eq_,
    eq_scalar_ as _generic_eq_scalar_,
)
from flag_gems.runtime import device

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
device = device.name

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
def eq_func(x, y):
    return x.to(tl.float32) == y.to(tl.float32)


def eq(A, B):
    if A.device != B.device:
        if A.device.type == device:
            B = B.to(A.device)
        else:
            A = A.to(B.device)
    logger.debug("GEMS_KUNLUNXIN EQ")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = eq_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def eq_func_scalar(x, y):
    return x.to(tl.float32) == y


def eq_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN EQ_SCALAR")
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
    ):
        s = float(B)
        wrapped = torch.tensor(s, dtype=dtype).item()
        if math.isfinite(wrapped):
            # wrapped == torch's wrapped-scalar semantics (compare in the
            # input dtype). Only take the fast path when the wrapped scalar
            # is finite: when |s| overflows the dtype (e.g. 66000 for fp16,
            # 1e300 for fp32) torch wraps it to +/-inf and x == +/-inf must
            # stay on the exact generic compare path.
            if numel >= _EQ_SCALAR_FAST_TILE * _EQ_SCALAR_MIN_GRID and numel % _EQ_SCALAR_FAST_TILE == 0:
                # exact-multiple flat tiles (grid >= MIN_GRID): no mask, no
                # i1 -- a saturating fp32 store + vendor bool conversion.
                return _eq_scalar_fast(A, float(wrapped), (numel // _EQ_SCALAR_FAST_TILE,))
            if numel >= _EQ_SCALAR_MASKED_MIN and numel % _EQ_SCALAR_FAST_TILE != 0:
                # non-multiple mid sizes (e.g. 2.56M, [10000,256]): flat
                # tiles with a real tail mask. The mask is genuine (tail
                # elements), so the masked-memory path is the only penalty.
                return _eq_scalar_fast_masked(A, float(wrapped), numel)
    return eq_func_scalar(A, B)


# ---------------------------------------------------------------------------
# eq_scalar fast paths (fp16/fp32/bf16, contiguous, finite wrapped scalar).
#
# Why: like the gt/lt/greater scalar family, the generic scalar-compare path
# (pointwise_dynamic 1d-tile codegen) always materializes
# `arith.cmpf -> i1 -> bool store` per lane. On XPU the i1 compare alone is a
# per-lane slow path (~10-20x): measured with a where(x==s) kernel, the same
# flat tile in fp32 saturating arithmetic is 8-15x faster than the i1 variant
# on [10000,65536] (probe 2026-08-13, XPU 1).
#
# Equality cannot use the gt/lt `max(0, min(1, (x-s)*K))` shape because x==s
# has no natural gap direction; instead we saturate the *distance*:
#   t = min(1, |x - s| * 2^149-ish)   -> 0.0 when x == s, 1.0 when x != s
#   out = max(0, 1 - t)                -> 1.0 when equal, 0.0 otherwise
# SCALE = 1e30 * 1e15: every representable fp16/bf16/fp32 gap (min 2^-149
# subnormal spacing) saturates t to exactly 1.0, while a zero difference
# stays exactly 0.0. subnormal-vs-zero gaps are exact (power-of-two scaling),
# +-0 == +-0 -> True, NaN input -> False (the trailing max(0, 1-t) maps the
# NaN from |NaN - s| to 0; on bf16 the naive 1 - min(1, NaN) yields NaN and
# NaN converts to True, hence max(0, .) is required).
#
# The tensored scalar passed to the kernel is float(wrapped) -- the scalar
# rounded to the input dtype -- which is bit-identical to torch's wrapped
# scalar for the comparison (benchmark 0.001 in fp16/bf16/fp32 is admitted).
# The +/-inf corner (x = s = +/-inf -> True) requires a wrapped scalar of
# +/-inf: those scalars are rejected above by math.isfinite, keeping the
# exact generic compare path. NaN scalars also stay generic.
#
# Second stage: fp32 -> bool via `torch.ops.aten._copy_from` (NOT registered
# by gems, so it always reaches the vendor's native conversion kernel;
# measured ~1.97 ms on [10000,65536] fp16, vs the generic path's 15.9 ms).
_EQ_SCALAR_FAST_TILE = 131072
_EQ_SCALAR_MIN_GRID = 128
_EQ_SCALAR_MASKED_MIN = 1 << 20


@triton.jit
def eq_scalar_fast_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    d = tl.abs(x - scalar)
    t = tl.minimum(1.0, d * 1.0e30 * 1.0e15)
    tl.store(out_ptr + tid, tl.maximum(0.0, 1.0 - t))


def _eq_scalar_fast(A, scalar, grid):
    out32 = torch.empty_like(A, dtype=torch.float32)
    eq_scalar_fast_kernel[grid](
        out32,
        A,
        scalar,
        TILE=_EQ_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out, False)
    return out


@triton.jit
def eq_scalar_fast_masked_kernel(out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float32)
    d = tl.abs(x - scalar)
    t = tl.minimum(1.0, d * 1.0e30 * 1.0e15)
    tl.store(out_ptr + tid, tl.maximum(0.0, 1.0 - t), mask=mask)


def _eq_scalar_fast_masked(A, scalar, numel):
    out32 = torch.empty_like(A, dtype=torch.float32)
    grid = (math.ceil(numel / _EQ_SCALAR_FAST_TILE),)
    eq_scalar_fast_masked_kernel[grid](
        out32,
        A,
        scalar,
        numel,
        TILE=_EQ_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out, False)
    return out


# ---------------------------------------------------------------------------
# eq_ (in-place alias of eq.Tensor, e.g. `x.eq_(y)` on a float tensor).
# torch keeps the input dtype and stores 0.0/1.0 (False/True) back into x.
#
# Before this change eq_ was NOT overridden by the kunlunxin backend, so it
# fell to the generic ops/eq_.py wrapper (promotion ALWAYS_BOOL;
# `arith.cmpf -> i1 -> bool` per lane, out0=A). On XPU a fp compare followed
# by ANY use of the i1 result lowers to a per-lane slow path: the sibling
# gt_/lt_ in-place family measured this traversal at 0.27-0.33x
# dtype-equal-weight (big shapes as low as 0.07x, e.g. [10000,65536] fp16
# gems ~22ms vs torch ~1.4ms).
#
# Fix (same single-kernel saturating-fp recipe as the committed in-place
# lt_/ge_/gt_scalar_/gt_tensor_ family, no i1 ever materialized):
#   1. generic in-place kernel with DEFAULT promotion + saturating fp
#      arithmetic, under the in-place-safe CodeGenConfig below (the
#      out-of-place config_ has isCloseMemoryAsync=False = async copy ON,
#      which with in-place aliasing is the documented "noc idle timeout"
#      deadlock, see the config_inplace_ note in lt.py / gt.py).
#   2. unmasked flat-tile in-place fast kernel for fp16/fp32 contiguous
#      tensors whose numel is an exact multiple of TILE (grid >= MIN_GRID):
#      the always-true runtime mask of the codegen path would force the slow
#      masked-memory channel, so a fixed pow2 TILE with no mask at all
#      restores the fast DMA path (per lt.py/gt.py in-place probes).
#   3. Equality has no gap direction, so the gt/lt `max(0, min(1, (x-y)*K))`
#      shape cannot be used; instead saturate the *distance* (the same
#      two-stage 1e32*1e32 = 1e64 factor as the gt_scalar_ in-place fast
#      path):
#        t   = min(1, |x - y| * 1e32 * 1e32)   -> 0 when x == y, 1 when x != y
#        out = max(0, 1 - t)                   -> 1 when equal, 0 otherwise
#      Every representable nonzero gap (down to the fp16 gap 2^-24, the bf16
#      subnormal 2^-133 and the fp32 subnormal 2^-149 = 1.4e-45) saturates
#      to a value >= 1, while a zero difference stays exactly 0. max/min on
#      this backend prefer the non-NaN operand, so |NaN - y| = NaN collapses
#      to False downstream (matching `NaN != anything`), +-0 == +-0 -> True,
#      and equal +-inf pairs are exact. The slow i1/bool path is never
#      materialized.
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
def eq_func_tensor_inplace(x, y):
    t = (x.to(tl.float32) - y.to(tl.float32)) * 1.0e32
    t = t * 1.0e32
    t = tl.abs(t)
    t = tl.minimum(1.0, t)
    t = tl.maximum(0.0, 1.0 - t)
    return t


def eq_(A, B):
    logger.debug("GEMS_KUNLUNXIN EQ_ TENSOR")
    if A.device != B.device:
        B = B.to(A.device)
    numel = A.numel()
    if A.is_contiguous() and A.dtype in (torch.float16, torch.float32, torch.bfloat16):
        if (
            A.dtype in (torch.float16, torch.float32)
            and B.is_contiguous()
            and B.dtype == A.dtype
            and A.shape == B.shape
            and numel >= _EQ_TENSOR_INPLACE_FAST_TILE * _EQ_TENSOR_INPLACE_MIN_GRID
            and numel % _EQ_TENSOR_INPLACE_FAST_TILE == 0
        ):
            # exact-multiple flat tiles: no mask at all; grid fixed.
            return _eq_tensor_inplace_fast(A, B, numel)
        eq_func_tensor_inplace(A, B, out0=A)
        return A
    # Everything else (non-float dtype, non-contiguous, ...) keeps the
    # original generic in-place path, behavior unchanged.
    return _generic_eq_(A, B)


# in-place alias safety: the fast kernel writes into the SAME tensor it
# reads, so it must keep the DEFAULT isCloseMemoryAsync (True = async copy
# closed); passing False with in-place aliasing is the documented "noc idle
# timeout" deadlock, same as gt.py's _gt_scalar_inplace_fast note.
_EQ_TENSOR_INPLACE_FAST_TILE = 131072
_EQ_TENSOR_INPLACE_MIN_GRID = 128


@triton.jit
def eq_tensor_inplace_fast_kernel(x_ptr, y_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    y = tl.load(y_ptr + tid)
    t = (x.to(tl.float32) - y.to(tl.float32)) * 1.0e32
    t = t * 1.0e32
    t = tl.abs(t)
    t = tl.minimum(1.0, t)
    t = tl.maximum(0.0, 1.0 - t)
    tl.store(x_ptr + tid, t)


def _eq_tensor_inplace_fast(A, B, numel):
    grid = (numel // _EQ_TENSOR_INPLACE_FAST_TILE,)
    eq_tensor_inplace_fast_kernel[grid](
        A,
        B,
        TILE=_EQ_TENSOR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A


# ---------------------------------------------------------------------------
# eq_scalar_ (in-place alias of eq.Scalar, e.g. `x.eq_(0)` on a float tensor).
# torch keeps the input dtype and stores 0.0/1.0 (False/True) back into x.
#
# Before this change eq_scalar_ was NOT overridden by the kunlunxin backend,
# so it fell to the generic ops/eq_.py wrapper (promotion ALWAYS_BOOL;
# `arith.cmpf -> i1 -> bool` per lane, out0=A). On XPU the big shapes hit the
# documented slow path: [10000,65536] fp16 gems ~1.59s vs torch ~1.43s and
# the dtype-equal-weight baseline measured 0.0446x (fp16 0.0349 / fp32
# 0.0606 / bf16 0.0383, 2026-08-16, XPU 2) -- the same catastrophic
# ALWAYS_BOOL path the eq_/gt_/lt_ in-place family fixed with the saturating
# fp recipe.
#
# Fix (the exact in-place scalar recipe of gt_scalar_ / le_scalar_ family,
# mirrored from _eq_tensor_inplace_ above):
#   1. generic in-place kernel `eq_func_scalar_inplace` (is_tensor=[True,
#      False], DEFAULT promotion, no i1 materialized) under the in-place-safe
#      config_inplace_ (async copy closed — in-place aliasing + async streams
#      is the documented "noc idle timeout" deadlock, see note above).
#   2. unmasked flat-tile in-place fast kernel for fp16/fp32 contiguous
#      tensors whose numel is an exact multiple of TILE (grid >= MIN_GRID):
#      no runtime mask -> fast DMA path.
#   3. scalar-representability gate (wrapped-scalar semantics, same as
#      gt_scalar_): `float(B) == float(torch.tensor(float(B), dtype=A.dtype))`
#      plus a finite check on the wrapped value. The fp32 body then compares
#      bit-identically to torch's wrapped-scalar comparison (e.g. fp16 x vs
#      0.5). Non-representable scalars (fp16 0.1, bf16 0.001, fp32 pi), the
#      +/-inf/NaN scalar corners (x == +/-inf must stay exact), non-float
#      dtypes, and non-contiguous layouts keep the previous generic
#      ALWAYS_BOOL in-place path (forwarded to _generic_eq_scalar_), behavior
#      unchanged.
#   4. body: saturation of the *distance* (equality has no gap direction):
#        t   = min(1, |x - s| * 1e32 * 1e32)   -> 0 when x == s, 1 when x != s
#        out = max(0, 1 - t)                   -> 1 when equal, 0 otherwise
#      Every representable nonzero gap (down to the fp32/bf16 subnormals
#      ~1.4e-45) saturates to >= 1. The slow i1/bool path is never
#      materialized on the fast/saturate paths.
@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_inplace_,
)
@triton.jit
def eq_func_scalar_inplace(x, y):
    t = (x.to(tl.float32) - y) * 1.0e32
    t = t * 1.0e32
    t = tl.abs(t)
    t = tl.minimum(1.0, t)
    t = tl.maximum(0.0, 1.0 - t)
    return t


def eq_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN EQ_ SCALAR")
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and float(B) == float(torch.tensor(float(B), dtype=dtype).item())
    ):
        wrapped = float(torch.tensor(float(B), dtype=dtype).item())
        if math.isfinite(wrapped):
            if (
                dtype in (torch.float16, torch.float32)
                and numel >= _EQ_SCALAR_INPLACE_FAST_TILE * _EQ_SCALAR_INPLACE_MIN_GRID
                and numel % _EQ_SCALAR_INPLACE_FAST_TILE == 0
            ):
                # exact-multiple flat tiles: no mask at all; grid fixed.
                return _eq_scalar_inplace_fast(A, wrapped)
            eq_func_scalar_inplace(A, B, out0=A)
            return A
    # Everything else (non-representable scalar, non-finite scalar, non-float
    # dtype, non-contiguous, ...) keeps the original generic in-place path,
    # behavior unchanged.
    return _generic_eq_scalar_(A, B)


# in-place alias safety: the fast kernel writes into the SAME tensor it reads,
# so it must keep isCloseMemoryAsync=True (async copy closed); passing False
# with in-place aliasing is the documented "noc idle timeout" deadlock, same
# as _eq_tensor_inplace_fast above.
_EQ_SCALAR_INPLACE_FAST_TILE = 131072
_EQ_SCALAR_INPLACE_MIN_GRID = 128


@triton.jit
def eq_scalar_inplace_fast_kernel(x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    t = (x.to(tl.float32) - scalar) * 1.0e32
    t = t * 1.0e32
    t = tl.abs(t)
    t = tl.minimum(1.0, t)
    t = tl.maximum(0.0, 1.0 - t)
    tl.store(x_ptr + tid, t)


def _eq_scalar_inplace_fast(A, scalar):
    grid = (A.numel() // _EQ_SCALAR_INPLACE_FAST_TILE,)
    eq_scalar_inplace_fast_kernel[grid](
        A,
        scalar,
        TILE=_EQ_SCALAR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A
