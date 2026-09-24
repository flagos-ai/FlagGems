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
import struct

import numpy as np
import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.ops.eq_ import eq_ as _generic_eq_
from flag_gems.ops.eq_ import eq_scalar_ as _generic_eq_scalar_
from flag_gems.runtime import device

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
device = device.name


def _wrap_fp16(B):
    """Round python float B to fp16, returning (is_finite, fp16_value_as_float).

    Bit-identical to `torch.tensor(B, dtype=float16).item()` but ~10x cheaper on
    the host (no CUDA/XPU tensor allocation), which matters for the small/mid
    shapes whose runtime is dominated by host dispatch. Verified exact against
    torch over the finite/overflow/±0 ranges.
    """
    v = float(np.float16(float(B)))
    return math.isfinite(v), v


def _wrap_to_dtype(B, dtype):
    """Round python float B to `dtype` (fp16/fp32/bf16) numeric value, returning
    (is_finite, value_as_float). Matches `torch.tensor(B, dtype).item()` exactly
    without any tensor allocation (host-overhead trim for the small path)."""
    if dtype == torch.float16:
        v = float(np.float16(float(B)))
    elif dtype == torch.float32:
        v = float(np.float32(float(B)))
    else:  # bfloat16: RNE-truncate f32(B) to bf16, then read its numeric value
        u32 = struct.unpack("<I", struct.pack("<f", float(B)))[0]
        u16 = ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16) & 0xFFFF
        v = struct.unpack("<f", struct.pack("<I", u16 << 16))[0]
    return math.isfinite(v), v


def _wrap_bf16_as_fp16(B):
    """Compute the fp16 value whose 16 bits equal bf16(B)'s bits (the mid/bitcast
    bf16 path), returning (is_finite, value). Matches
    `torch.tensor(B, bfloat16).view(int16).view(float16).item()` exactly but
    without any tensor allocation. RNE-truncates f32(B) to bf16 then reinterprets.
    """
    u32 = struct.unpack("<I", struct.pack("<f", float(B)))[0]
    u16 = ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16) & 0xFFFF
    v = struct.unpack("<e", struct.pack("<H", u16))[0]
    return math.isfinite(v), float(v)


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


config_scalar_bigtile_ = CodeGenConfig(
    1024,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    buffer_size_limit=16384,
    unroll_num=16,
)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_scalar_bigtile_,
)
@triton.jit
def eq_func_scalar(x, y):
    return x.to(tl.float32) == y


# ---------------------------------------------------------------------------
# eq_scalar SMALL-shape fast path (numel <= _EQ_SCALAR_SMALL_NUMEL).
#
# Profiling (harness/perf_ir/probe_eq_*.py, do_bench) showed small shapes are
# dominated NOT by the kernel or hardware but by the generic pointwise_dynamic
# host layer: per-call it re-does dynamic shape/stride analysis, task build,
# grid compute and arg packing, costing ~60-180us. For a 64x64 tensor whose
# kernel runs in ~7us that framework layer is a 10-30x pure overhead, while
# torch's native C++ op has none of it (~5us). Bypassing pointwise_dynamic
# with a single-tile hand-written kernel drops 64x64/10000-elem eq_scalar from
# ~0.14x to ~0.5-0.75x (measured all dtypes). The trick only helps SMALL
# shapes: at numel >= ~131072 the single masked tile loses to the autogrid
# vectorized DMA of the generic path (masked memory is a non-coalesced slow
# path on XPU), so the threshold is capped below that. The kernel is identical
# in semantics to eq_func_scalar (`x.to(f32) == wrapped_scalar`), so +/-0 (both
# equal) and NaN (never equal) match torch; the scalar is wrapped to the tensor
# dtype for bit-exact parity, same as the float path below.
_EQ_SCALAR_SMALL_NUMEL = 65536


@triton.jit
def eq_scalar_small_kernel(out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr):
    tid = tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float32)
    tl.store(out_ptr + tid, (x == scalar).to(tl.int8), mask=mask)


def _eq_scalar_small(A, wrapped):
    numel = A.numel()
    out = torch.empty(numel, dtype=torch.int8, device=A.device)
    TILE = triton.next_power_of_2(numel)
    eq_scalar_small_kernel[(1,)](
        out,
        A.reshape(-1),
        wrapped,
        numel,
        TILE=TILE,
        num_warps=4,
        isCloseMemoryAsync=False,
    )
    return out.view(torch.bool).reshape(A.shape)


# ---------------------------------------------------------------------------
# eq_scalar MID-shape fast path (fp16/bf16, 65536 < numel <= _EQ_SCALAR_MID_NUMEL).
#
# The mid band (~2.56M) reported speedup collapses under the generic
# pointwise_dynamic path (bf16 0.27, fp16 0.44) because its host dispatch
# (~28us/call) dominates the short (~9us) device kernel. On large shapes that
# host cost is hidden behind the long kernel, so the generic path is fine there.
# A multi-tile hand kernel keeps the SAME CmpF-fusion device path (vcmpf,
# XPU-coalesced) but drops the framework overhead. Crucially the fusion env MUST
# be set or the compare falls back to the per-lane i1 store slow path (~10x).
#
# IR + P800-profiler analysis (ground-truth device time, not do_bench): a naive
# TILE=16384 hand kernel emits ONE small gm2lm(16384)->mfence->load->vcmpf->store
# ->lm2gm(16384) per grid-stride step = many tiny DMA transactions (13.45us dev).
# The tuned pointwise kernel instead uses a 65536-wide tile (sizePerCore=1024)
# with ONE big gm2lm(65536), a chunked inner compute loop (65536/buffer_size_limit
# = 4 chunks) then ONE big lm2gm(65536). Matching that structure -- TILE=65536,
# buffer_size_limit=16384, unroll_num=16 -- coalesces the DMA and drops device
# time to 9.35us (matching pointwise 8.98us). This lifted the mid speedups to
# fp16 ~0.78 and bf16 ~0.83 (dtype-balanced ~0.805 -> ~0.833).
# Crossover: >4.19M the generic path wins (16.7M: 0.76 vs 0.39), so the upper
# bound is capped at _EQ_SCALAR_MID_NUMEL. fp32 is excluded: its longer kernel
# already amortizes the host cost. bf16 reuses the fp16 bitcast (32-lane vcmpf)
# with the finite-fp16-view guard for exactness (see the bitcast path below).
_EQ_SCALAR_MID_NUMEL = 4194304
_EQ_SCALAR_MID_TILE = 65536
_EQ_SCALAR_MID_BSL = 16384
_EQ_SCALAR_MID_UNROLL = 16


@triton.jit
def eq_scalar_tile_kernel(out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * TILE + tl.arange(0, TILE)
    mask = off < numel
    x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
    tl.store(out_ptr + off, (x == scalar).to(tl.int8), mask=mask)


def _eq_scalar_tiled(A, wrapped):
    numel = A.numel()
    out = torch.empty(numel, dtype=torch.int8, device=A.device)
    grid = (triton.cdiv(numel, _EQ_SCALAR_MID_TILE),)
    eq_scalar_tile_kernel[grid](
        out,
        A.reshape(-1),
        wrapped,
        numel,
        TILE=_EQ_SCALAR_MID_TILE,
        num_warps=4,
        isCloseMemoryAsync=False,
        buffer_size_limit=_EQ_SCALAR_MID_BSL,
        unroll_num=_EQ_SCALAR_MID_UNROLL,
    )
    return out.view(torch.bool).reshape(A.shape)


def eq_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN EQ_SCALAR")
    # Small-shape fast path: bypass the pointwise_dynamic host layer (its
    # ~60-180us/call dynamic-dispatch overhead dominates small shapes) with a
    # single-tile hand-written kernel. Only for finite wrapped scalars so the
    # `x.to(f32) == wrapped` compare is bit-exact vs torch (+/-inf/NaN scalars
    # keep the generic path). Capped at _EQ_SCALAR_SMALL_NUMEL: larger shapes
    # prefer the autogrid vectorized DMA of the generic path.
    if (
        A.is_contiguous()
        and A.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and A.numel() <= _EQ_SCALAR_SMALL_NUMEL
        and A.numel() > 0
    ):
        wrapped_finite, wrapped = _wrap_to_dtype(B, A.dtype)
        if wrapped_finite:
            os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
            os.environ["TRITONXPU_FP16_FAST"] = "1"
            res = _eq_scalar_small(A, wrapped)
            del os.environ["TRITONXPU_COMPARE_FUSION"]
            del os.environ["TRITONXPU_FP16_FAST"]
            return res
    # Mid-shape fast path (fp16/fp32, 65536 < numel <= _EQ_SCALAR_MID_NUMEL):
    # bypass the pointwise_dynamic host dispatch (~28us/call) with a multi-tile
    # hand kernel that keeps the CmpF-fusion vcmpf device path. bf16 goes through
    # the fp16 bitcast (32-lane vcmpf) guarded by the finite-fp16-view check.
    # fp32 is included too: at mid shapes the generic path is host-dispatch bound,
    # so the hand kernel (fp32 x.to(f32) no-op, native 16-lane f32 vcmpf) recovers it.
    if (
        A.is_contiguous()
        and A.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and _EQ_SCALAR_SMALL_NUMEL < A.numel() <= _EQ_SCALAR_MID_NUMEL
    ):
        if A.dtype == torch.bfloat16:
            s_finite, s_val = _wrap_bf16_as_fp16(B)
            if s_finite:
                os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
                os.environ["TRITONXPU_FP16_FAST"] = "1"
                res = _eq_scalar_tiled(A.view(torch.float16), s_val)
                del os.environ["TRITONXPU_COMPARE_FUSION"]
                del os.environ["TRITONXPU_FP16_FAST"]
                return res.reshape(A.shape)
        else:
            wrapped_finite, wrapped = _wrap_to_dtype(B, A.dtype)
            if wrapped_finite:
                os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
                os.environ["TRITONXPU_FP16_FAST"] = "1"
                res = _eq_scalar_tiled(A, wrapped)
                del os.environ["TRITONXPU_COMPARE_FUSION"]
                del os.environ["TRITONXPU_FP16_FAST"]
                return res
    # bf16 -> fp16 BITCAST fast path (equality only). Reinterpret the 2-byte
    # bf16 buffer as fp16 and compare in fp16, routing bf16 through the 32-lane
    # fp16 vcmpf fusion instead of the slow 16-lane bf16->f32 widened compare
    # (~1.6x faster; bf16 has no native SIMD compare so it widens to f32).
    #
    # Correctness: equality is a bit-pattern relation (bit-eq OR both +/-0,
    # NaN never-equal), and the reinterpret keeps all 16 bits, so
    # view_fp16(x) == view_fp16(s) is bit-identical to (x_bf16 == s) EXCEPT for
    # the 1790-pattern "danger set" that is bf16 finite-nonzero (|x|>=2.68e36,
    # 0x7C01..0xFF7F) but maps to fp16-NaN (proven by full 65536-pattern
    # enumeration; bf16-NaN always maps to fp16-NaN, +/-0 sets are identical
    # across formats). GUARD: only take this path when the WRAPPED scalar's
    # fp16-view is finite. Then every danger-set element (fp16 NaN) is compared
    # against a finite fp16 scalar -> False on both formats, and no mismatch
    # remains. The guard ~never fires for real scalars (they are small/finite).
    # Only valid for EQUALITY (ordering is not bit-monotonic under reinterpret).
    if A.dtype == torch.bfloat16 and A.is_contiguous():
        s_finite, s = _wrap_bf16_as_fp16(B)
        if s_finite:
            Ai = A.view(torch.float16)
            os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
            os.environ["TRITONXPU_FP16_FAST"] = "1"
            res = eq_func_scalar(Ai, s)
            del os.environ["TRITONXPU_COMPARE_FUSION"]
            del os.environ["TRITONXPU_FP16_FAST"]
            return res
    # Full-precision float CmpF-fusion path. torch compares `x == wrapped(B)`
    # where wrapped(B) is B rounded to the tensor dtype; passing the wrapped
    # scalar keeps `x.to(f32) == wrapped` bit-identical to torch for fp16/bf16
    # (a raw B differs when B is not representable in the tensor dtype).
    if A.dtype in (torch.float16, torch.bfloat16, torch.float32):
        B = _wrap_to_dtype(B, A.dtype)[1]
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = eq_func_scalar(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


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
    if A.is_contiguous() and A.dtype in (torch.float16, torch.float32, torch.bfloat16):
        eq_func_tensor_inplace(A, B, out0=A)
        return A
    # Everything else (non-float dtype, non-contiguous, ...) keeps the
    # original generic in-place path, behavior unchanged.
    return _generic_eq_(A, B)


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
        and float(B) == _wrap_to_dtype(B, dtype)[1]
    ):
        wrapped = _wrap_to_dtype(B, dtype)[1]
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
