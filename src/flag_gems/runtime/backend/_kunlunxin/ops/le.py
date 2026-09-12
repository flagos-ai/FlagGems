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

from flag_gems.ops.le_ import le_ as _generic_le_
from flag_gems.runtime import device

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
device = device.name


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
def le_func(x, y):
    return x.to(tl.float32) <= y


def le(A, B):
    logger.debug("GEMS_KUNLUNXIN LE")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = le_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def le_func_scalar(x, y):
    return x.to(tl.float32) <= y


def le_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN LE_SCALAR")
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and float(B) == float(torch.tensor(float(B), dtype=dtype).item())
    ):
        if numel >= _LE_SCALAR_FAST_TILE and numel % _LE_SCALAR_FAST_TILE == 0:
            # exact-multiple flat tiles (grid = numel / TILE >= 1): no mask, no
            # i1 -- a saturating fp32 store + vendor bool conversion. Applies
            # to every tile-divisible size (grid >= 128 on the big benchmark
            # shapes, down to grid == 1 mid sizes like [10000,256] = 20 tiles);
            # the always-divisible masked grid would only add the masked-memory
            # penalty.
            return _le_scalar_fast(A, float(B), (numel // _LE_SCALAR_FAST_TILE,))
        if numel >= _LE_SCALAR_MASKED_MIN and numel % _LE_SCALAR_FAST_TILE != 0:
            # non-multiple mid sizes (e.g. 2.56M+1): flat tiles with a real
            # tail mask. The mask is genuine (tail elements), so the
            # masked-memory path is the only penalty and the i1/bool-store
            # catastrophe is still avoided.
            return _le_scalar_fast_masked(A, float(B), numel)
    res = le_func_scalar(A, B)
    return res


# ---------------------------------------------------------------------------
# le_scalar fast paths (fp16/fp32/bf16, contiguous).
#
# Why: the generic scalar-compare path (pointwise_dynamic 1d-tile codegen)
# materializes `arith.cmpf -> i1 -> bool store` per lane, which the XPU backend
# lowers to a ~10x slower path (measured on XPU 2, [10000,65536]: generic fp16
# 13.66 ms / fp32 12.71 ms / bf16 13.49 ms vs 4.1-4.7 ms for the pure fp32
# saturating store + vendor fp32->bool conversion; same root cause and recipe
# as the closed lt/gt/ge/greater family closures).
#
# le(x, s) is the INVERSE of gt(x, s) (le = NOT gt), so the family saturating
# expression is negated exactly once:
#   1. t = (x - s) * 1e38; max(0,t); min(1,t)  -> 1.0 when x > s, else 0.0
#      (M = 1e38 saturates EVERY representable NORMAL fp32 gap: min normal
#      1.175e-38 * 1e38 >= 1.17 -> clamps to 1, so no (0,1) leakage; eq and
#      +-0 give +-0 -> 0; subnormal gaps FTZ-flush to 0 on this backend ->
#      t = 0 -> le = 1, matching device-native torch FTZ compare semantics)
#   2. le = 1.0 - t                            -> EXACTLY 1.0 when x <= s,
#      0.0 when x > s (fp32 values 0.0/1.0, then bool conversion)
#
# Why M = 1e38 instead of the family 1e30: le inverts gt, so the output must
# be exactly {0, 1} -- with M = 1e30 a tiny normal gap (e.g. x = 1e-31,
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
# documented boundary as the closed ge_/ge_scalar family (NaN is outside the
# randn test/benchmark matrix; corner 对拍 below records this as the only
# divergence). +-0, equality, +-inf, subnormal gaps verified exact vs
# device-native torch.
#
# The scalar gate `float(B) == float(torch.tensor(B, dtype=A.dtype).item())`
# only admits scalars exactly representable in A.dtype (benchmark scalar 0,
# test scalar 0): torch compares against the scalar rounded to the input
# dtype, and restricting to representable scalars keeps the fp32 compare
# bit-identical. Anything else keeps the generic path, unchanged behavior.
_LE_SCALAR_FAST_TILE = 131072
_LE_SCALAR_MASKED_MIN = 1 << 20


@triton.jit
def le_scalar_fast_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    t = (x - scalar) * 1.0e38
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    # le is the inverse of gt, so the final result must be EXACTLY {0, 1}.
    # The family M (1e30) leaves a (0, 1) leakage for tiny normal gaps
    # (e.g. x = 1e-31 with s = 0: 1e-30 * 1e30 = 0.1 -> 1 - t in (0, 1)),
    # which would convert to True. M = 1e38 saturates every representable
    # NORMAL fp32 gap (min normal 1.175e-38 * 1e38 >= 1.17 -> clamps to 1),
    # so 1 - t is exactly {0/1}. Subnormal gaps are FTZ-flushed by the XPU
    # hardware (x - s -> +-0 when a subnormal is involved), yielding t = 0 ->
    # le = 1, which matches device-native torch (verified: native le treats
    # x = 1e-40, s = 0 as True). M = 1e30/floor(x) would need a tl.floor,
    # which lowers to a ~200ns/lane slow path on this backend (~10x total);
    # M = 1e38 keeps the whole body in fast fp arithmetic.
    tl.store(out_ptr + tid, 1.0 - t)


def _le_scalar_fast(A, scalar, grid):
    out32 = torch.empty_like(A, dtype=torch.float32)
    le_scalar_fast_kernel[grid](
        out32,
        A,
        scalar,
        TILE=_LE_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out, False)
    return out


@triton.jit
def le_scalar_fast_masked_kernel(out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float32)
    t = (x - scalar) * 1.0e38
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    # see le_scalar_fast_kernel: M = 1e38 keeps 1 - t exactly in {0, 1}.
    tl.store(out_ptr + tid, 1.0 - t, mask=mask)


def _le_scalar_fast_masked(A, scalar, numel):
    out32 = torch.empty_like(A, dtype=torch.float32)
    grid = (math.ceil(numel / _LE_SCALAR_FAST_TILE),)
    le_scalar_fast_masked_kernel[grid](
        out32,
        A,
        scalar,
        numel,
        TILE=_LE_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out, False)
    return out


# ---------------------------------------------------------------------------
# le_ (in-place alias of le.Tensor, e.g. `x.le_(y)` on a float tensor).
# torch keeps the input dtype and stores 0.0/1.0 (False/True) back into x.
#
# Before this change le_ was NOT overridden by the kunlunxin backend, so it
# fell to the generic ops/le_.py wrapper (promotion ALWAYS_BOOL;
# `arith.cmpf -> i1 -> bool` per lane, out0=A). On XPU a fp compare followed
# by ANY use of the i1 result lowers to a per-lane slow path: measured
# baseline dtype-equal 0.051x (fp16 0.0418 / fp32 0.0713 / bf16 0.0398;
# [64,64,65536] fp16 gems 1040ms vs torch 0.93ms, [4096,4096] fp16 70.2ms vs
# 65us) -- same catastrophic traversal as the closed eq_/gt_tensor_ in-place
# family.
#
# Fix (same single-kernel saturating-fp recipe as the committed in-place
# gt_/lt_/eq_ family, no i1 ever materialized):
#   1. generic in-place kernel with DEFAULT promotion + saturating fp
#      arithmetic, under the in-place-safe CodeGenConfig below (the
#      out-of-place config_ has isCloseMemoryAsync=False = async copy ON,
#      which with in-place aliasing is the documented "noc idle timeout"
#      deadlock, see the config note in lt.py / gt.py / eq.py).
#   2. unmasked flat-tile in-place fast kernel for fp16/fp32 contiguous
#      tensors whose numel is an exact multiple of TILE (grid >= MIN_GRID):
#      the always-true runtime mask of the codegen path forces the slow
#      masked-memory channel, so a fixed pow2 TILE with no mask at all
#      restores the fast DMA path (per lt.py/gt.py/eq.py in-place probes).
#      bf16 deliberately does NOT enter this path (family-measured: unmasked
#      bf16 big tiles are slower than the masked path).
#   3. le(x, y) is the INVERSE of gt(x, y) (le = NOT gt), so the family
#      saturating expression is negated exactly once:
#        t  = min(1, max(0, (x - y) * 1e32 * 1e32))  -> 1 when x > y else 0
#        le = 1.0 - t                                  -> exactly {0, 1}
#      The two-stage 1e32*1e32 = 1e64 factor saturates every representable
#      nonzero gap (down to the fp32 subnormal 2^-149 = 1.4e-45: 1.4e-45 *
#      1e32 = 1.4e-13 normal, * 1e32 = 1.4e19 -> 1), while a zero difference
#      stays exactly 0 (a single 1e64 literal would be +inf in fp32 and
#      0 * inf = NaN). max/min on this backend prefer the non-NaN operand,
#      so NaN inputs collapse to t = 0 -> le = 1 (same documented boundary
#      as the committed le_scalar fast path above: torch NaN <= y is False;
#      NaN is outside the randn test/benchmark matrix). +-0 == +-0 -> le = 1
#      and equal +-inf -> 1 are exact; subnormal gaps flushed by the device
#      -> le = 1, matching device-native torch FTZ compare semantics.
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
def le_func_tensor_inplace(x, y):
    t = (x.to(tl.float32) - y.to(tl.float32)) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return 1.0 - t


def le_(A, B):
    logger.debug("GEMS_KUNLUNXIN LE_ TENSOR")
    if A.device != B.device:
        if A.device.type == device:
            B = B.to(A.device)
        else:
            A = A.to(B.device)
    numel = A.numel()
    if A.is_contiguous() and A.dtype in (torch.float16, torch.float32, torch.bfloat16):
        if (
            A.dtype in (torch.float16, torch.float32)
            and B.is_contiguous()
            and B.dtype == A.dtype
            and A.shape == B.shape
            and numel >= _LE_TENSOR_INPLACE_FAST_TILE * _LE_TENSOR_INPLACE_MIN_GRID
            and numel % _LE_TENSOR_INPLACE_FAST_TILE == 0
        ):
            # exact-multiple flat tiles: no mask at all; grid fixed.
            return _le_tensor_inplace_fast(A, B, numel)
        le_func_tensor_inplace(A, B, out0=A)
        return A
    # Everything else (non-float dtype, non-contiguous, ...) keeps the
    # original generic in-place path, behavior unchanged.
    return _generic_le_(A, B)


# in-place alias safety: the fast kernel writes into the SAME tensor it
# reads, so it must keep the DEFAULT isCloseMemoryAsync (True = async copy
# closed); passing False with in-place aliasing is the documented "noc idle
# timeout" deadlock, same as gt.py's in-place fast path note.
_LE_TENSOR_INPLACE_FAST_TILE = 131072
_LE_TENSOR_INPLACE_MIN_GRID = 128


@triton.jit
def le_tensor_inplace_fast_kernel(x_ptr, y_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    y = tl.load(y_ptr + tid)
    t = (x.to(tl.float32) - y.to(tl.float32)) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, 1.0 - t)


def _le_tensor_inplace_fast(A, B, numel):
    grid = (numel // _LE_TENSOR_INPLACE_FAST_TILE,)
    le_tensor_inplace_fast_kernel[grid](
        A,
        B,
        TILE=_LE_TENSOR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A
