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

from flag_gems.ops.ne_ import (
    ne_ as _generic_ne_,
    ne_scalar_ as _generic_ne_scalar_,
)

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


# NOTE: `kunlunAutoGrid=True` + `unroll_num=8` are what make the sibling tuned
# comparison ops (gt / greater / greater_scalar) reach ~0.23-0.41 on large
# shapes. ne/ne_scalar previously shipped a bare config WITHOUT them and were
# stuck at ~0.14 (gems ~7.95ms vs torch ~1.08ms on the 65536-wide shapes, IR
# baseline `harness/perf_ir_3/ir-ne_scalar-dev3.log`). Adding the two params
# lifts throughput ~1.6x (mirrors greater_scalar, zero algorithm change).
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
def ne_func(x, y):
    return x.to(tl.float32) != y.to(tl.float32)


def ne(A, B):
    logger.debug("GEMS_KUNLUNXIN NE")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = ne_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def ne_func_scalar(x, y):
    return x.to(tl.float32) != y


def ne_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN NE_SCALAR")
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
            # 1e300 for fp32) torch wraps it to +/-inf and x != +/-inf must
            # stay on the exact generic compare path (x = +/-inf vs
            # s = +/-inf -> False cannot be expressed by the saturating
            # distance formula, which sees inf-inf = NaN -> True).
            if numel >= _NE_SCALAR_FAST_TILE * _NE_SCALAR_MIN_GRID and numel % _NE_SCALAR_FAST_TILE == 0:
                # exact-multiple flat tiles (grid >= MIN_GRID): no mask, no
                # i1 -- a saturating fp32 store + vendor bool conversion.
                return _ne_scalar_fast(A, float(wrapped), (numel // _NE_SCALAR_FAST_TILE,))
            if numel >= _NE_SCALAR_MASKED_MIN and numel % _NE_SCALAR_FAST_TILE != 0:
                # non-multiple mid sizes (e.g. 2.56M, [10000,256]): flat
                # tiles with a real tail mask. The mask is genuine (tail
                # elements), so the masked-memory path is the only penalty.
                return _ne_scalar_fast_masked(A, float(wrapped), numel)
    # Like gt_scalar / greater_scalar, the scalar path must NOT set
    # TRITONXPU_COMPARE_FUSION / TRITONXPU_FP16_FAST: for tensor-vs-scalar the
    # fusion env vars make the compiler emit an fp16 compare that trips
    # `arith.cmpf same-type` and overflows uni_sram -> compile failure.
    res = ne_func_scalar(A, B)
    return res


# ---------------------------------------------------------------------------
# ne_scalar fast paths (fp16/fp32/bf16, contiguous, finite wrapped scalar).
#
# Why: like the eq_scalar family, the generic scalar-compare path
# (pointwise_dynamic 1d-tile codegen) always materializes
# `arith.cmpf -> i1 -> bool store` per lane. On XPU the i1 compare alone is a
# per-lane slow path (~10-20x): eq_scalar measured a saturating-arithmetic
# flat tile 8-15x faster than the i1 variant on [10000,65536] (probe
# 2026-08-13, XPU 1), and ne_scalar showed the identical 17.6ms-vs-1.09ms
# disaster profile on the same shapes (baseline 2026-08-14, XPU 7).
#
# ne(scalar) is the logical complement of eq(scalar), but with *opposite* NaN
# semantics: ne(NaN, s) == True while eq(NaN, s) == False. The eq formula's
# saturating distance is exactly what ne needs, stored directly:
#   t = min(1, |x - s| * SCALE)   -> 0.0 when x == s, 1.0 otherwise
# SCALE = 1e30 * 1e15: every representable fp16/bf16/fp32 gap (min 2^-149
# subnormal spacing) saturates t to exactly 1.0 while a zero difference stays
# exactly 0.0; +-0 != +-0 -> False. NaN input -> t = min(1, NaN): on
# fp16/fp32 the min prefers the non-NaN operand (1.0) and on bf16 it yields
# NaN; both convert to bool True via the vendor conversion below -- which is
# exactly the required ne(NaN, s) == True (the eq path had to wrap the NaN
# away with max(0, .), ne must NOT).
#
# The tensored scalar passed to the kernel is float(wrapped) -- the scalar
# rounded to the input dtype -- which is bit-identical to torch's wrapped
# scalar for the comparison. The +/-inf corner (x = s = +/-inf -> False)
# requires a wrapped scalar of +/-inf: rejected above by math.isfinite,
# keeping the exact generic compare path. NaN scalars also stay generic.
#
# Second stage: fp32 -> bool via `torch.ops.aten._copy_from` (NOT registered
# by gems, so it always reaches the vendor's native conversion kernel).
_NE_SCALAR_FAST_TILE = 131072
_NE_SCALAR_MIN_GRID = 128
_NE_SCALAR_MASKED_MIN = 1 << 20


@triton.jit
def ne_scalar_fast_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    d = tl.abs(x - scalar)
    t = tl.minimum(1.0, d * 1.0e30 * 1.0e15)
    tl.store(out_ptr + tid, t)


def _ne_scalar_fast(A, scalar, grid):
    out32 = torch.empty_like(A, dtype=torch.float32)
    ne_scalar_fast_kernel[grid](
        out32,
        A,
        scalar,
        TILE=_NE_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out_bool = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out_bool, False)
    return out_bool


@triton.jit
def ne_scalar_fast_masked_kernel(out_ptr, y_ptr, scalar, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    y = tl.load(y_ptr + tid, mask=mask).to(tl.float32)
    d = tl.abs(y - scalar)
    t = tl.minimum(1.0, d * 1.0e30 * 1.0e15)
    tl.store(out_ptr + tid, t, mask=mask)


def _ne_scalar_fast_masked(A, scalar, numel):
    out32 = torch.empty_like(A, dtype=torch.float32)
    grid = (math.ceil(numel / _NE_SCALAR_FAST_TILE),)
    ne_scalar_fast_masked_kernel[grid](
        out32,
        A,
        scalar,
        numel,
        TILE=_NE_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out_bool = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out_bool, False)
    return out_bool


# ---------------------------------------------------------------------------
# ne_ (in-place alias of ne.Tensor, e.g. `x.ne_(y)` on a float tensor).
# torch keeps the input dtype and stores 0.0/1.0 (False/True) back into x.
#
# Before this change ne_ was NOT overridden by the kunlunxin backend, so it
# fell to the generic ops/ne_.py wrapper (promotion ALWAYS_BOOL;
# `arith.cmpf -> i1 -> bool` per lane, out0=A). That is the same documented
# XPU slow path the eq_/gt_/lt_ in-place family doomed at 0.05-0.27x; the
# generic ne_ path measured equally catastrophic (ne_ equal-weight ~0.05x,
# [10000,65536] fp16 gems ~22ms vs torch ~1.4ms).
#
# ne_ is the logical complement of eq_: eq_ stores max(0, 1 - t) where
# t = min(1, |x - y| * 1e32 * 1e32), so ne_ stores exactly t (no negation):
#   t = min(1, |x - y| * 1e32 * 1e32)  -> 0 when x == y, 1 when x != y
# Same saturating-distance math as the committed eq_/lt_/gt_/le_ in-place
# family (two-stage 1e32*1e32 = 1e64 factor), no i1 ever materialized.
#   * NaN   -> min(1, NaN): fp16/fp32 prefer the non-NaN operand so t = 1
#       (ne(NaN, y) == True, matches torch); bf16 yields NaN, same as the
#       eq_ in-place family's documented behavior.
#   * +-0  -> 0.0 (False), equal +-inf pairs -> 0.0 (False), exact.
# Same gates as eq_: fast unmasked flat tiles only for fp16/fp32 contiguous
# same-shape tensors with exact-multiple numel and grid >= MIN_GRID; all
# other dtypes/shapes/aliasing fall into the DEFAULT-promotion pointwise
# in-place kernel under the in-place-safe config; non-float/non-contiguous
# keep the previous generic in-place path, behavior unchanged.
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
def ne_func_tensor_inplace(x, y):
    t = (x.to(tl.float32) - y.to(tl.float32)) * 1.0e32
    t = t * 1.0e32
    t = tl.abs(t)
    t = tl.minimum(1.0, t)
    return t


def ne_(A, B):
    logger.debug("GEMS_KUNLUNXIN NE_ TENSOR")
    if A.device != B.device:
        B = B.to(A.device)
    numel = A.numel()
    if A.is_contiguous() and A.dtype in (torch.float16, torch.float32, torch.bfloat16):
        if (
            A.dtype in (torch.float16, torch.float32)
            and B.is_contiguous()
            and B.dtype == A.dtype
            and A.shape == B.shape
            and numel >= _NE_TENSOR_INPLACE_FAST_TILE * _NE_TENSOR_INPLACE_MIN_GRID
            and numel % _NE_TENSOR_INPLACE_FAST_TILE == 0
        ):
            # exact-multiple flat tiles: no mask at all; grid fixed.
            return _ne_tensor_inplace_fast(A, B, numel)
        ne_func_tensor_inplace(A, B, out0=A)
        return A
    # Everything else (non-float dtype, non-contiguous, ...) keeps the
    # original generic in-place path, behavior unchanged.
    return _generic_ne_(A, B)


# in-place alias safety: the fast kernel writes into the SAME tensor it
# reads, so it must keep the DEFAULT isCloseMemoryAsync (True = async copy
# closed); passing False with in-place aliasing is the documented "noc idle
# timeout" deadlock, same as _eq_tensor_inplace_fast.
_NE_TENSOR_INPLACE_FAST_TILE = 131072
_NE_TENSOR_INPLACE_MIN_GRID = 128


@triton.jit
def ne_tensor_inplace_fast_kernel(x_ptr, y_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    y = tl.load(y_ptr + tid)
    t = (x.to(tl.float32) - y.to(tl.float32)) * 1.0e32
    t = t * 1.0e32
    t = tl.abs(t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, t)


def _ne_tensor_inplace_fast(A, B, numel):
    grid = (numel // _NE_TENSOR_INPLACE_FAST_TILE,)
    ne_tensor_inplace_fast_kernel[grid](
        A,
        B,
        TILE=_NE_TENSOR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A


# ---------------------------------------------------------------------------
# ne_scalar_ (in-place alias of ne.Scalar, e.g. `x.ne_(0)` on a float
# tensor). torch keeps the input dtype and stores 0.0/1.0 back into x.
#
# Same gate/recipe as eq_scalar_ (wrapped-scalar representability + finite
# check -> fast unmasked flat tiles for fp16/fp32 exact-multiple -> DEFAULT
# config in-place pointwise -> generic fallback), stored value is the
# saturating distance t directly (ne(NaN, s) == True).
@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_inplace_,
)
@triton.jit
def ne_func_scalar_inplace(x, y):
    t = (x.to(tl.float32) - y) * 1.0e32
    t = t * 1.0e32
    t = tl.abs(t)
    t = tl.minimum(1.0, t)
    return t


def ne_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN NE_ SCALAR")
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
                and numel >= _NE_SCALAR_INPLACE_FAST_TILE * _NE_SCALAR_INPLACE_MIN_GRID
                and numel % _NE_SCALAR_INPLACE_FAST_TILE == 0
            ):
                # exact-multiple flat tiles: no mask at all; grid fixed.
                return _ne_scalar_inplace_fast(A, wrapped)
            ne_func_scalar_inplace(A, B, out0=A)
            return A
    # Everything else (non-representable scalar, non-finite scalar, non-float
    # dtype, non-contiguous, ...) keeps the original generic in-place path,
    # behavior unchanged.
    return _generic_ne_scalar_(A, B)


# in-place alias safety: same as _ne_tensor_inplace_fast above.
_NE_SCALAR_INPLACE_FAST_TILE = 131072
_NE_SCALAR_INPLACE_MIN_GRID = 128


@triton.jit
def ne_scalar_inplace_fast_kernel(x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    t = (x.to(tl.float32) - scalar) * 1.0e32
    t = t * 1.0e32
    t = tl.abs(t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, t)


def _ne_scalar_inplace_fast(A, scalar):
    grid = (A.numel() // _NE_SCALAR_INPLACE_FAST_TILE,)
    ne_scalar_inplace_fast_kernel[grid](
        A,
        scalar,
        TILE=_NE_SCALAR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A
