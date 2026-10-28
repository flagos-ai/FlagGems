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

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


_TL_TYPES = {
    torch.int8: tl.int8,
    torch.uint8: tl.uint8,
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
}

# Static Euclid steps run before the block-convergence while loop (phase 1).
# P1 is chosen just below the typical max per-block Euclid iteration count for
# the output dtype so most blocks finish with one cheap reduction or none:
#   int8/uint8 (values < 2^8):      P1=8
#   int16 (values < 2^16):          P1=14  (typical block max ~18)
#   int32 (values < 2^31):          P1=24  (typical block max ~30)
#   int64 (values < 2^63):          P1=32  (typical block max ~55)
# Masked steps are exact even past convergence (a % 1 == 0), so P1 is always
# correctness-safe; it only trades wasted steps against fewer block reductions.
# CHUNK: static steps per block max check in the while phase. int32 stays 1;
# int16 uses CHUNK=1 for numel >= _LARGE_NUMEL (measured 44.66 vs 45.35ms at
# 1G, 0.704 vs 0.714ms at 4096^2: the 6-shuffle B64 reduction is cheap enough
# that the CHUNK=2 over-run step costs more than the halved reduction count)
# and CHUNK=2 on tiny launch-bound shapes (proven value).
_LARGE_NUMEL = 1 << 20
_P1_BY_OUT = {
    torch.int8: 8,
    torch.uint8: 8,
    torch.int16: 14,
    torch.int32: 24,
    torch.int64: 32,
}
_CHUNK_BY_OUT = {
    torch.int8: 2,
    torch.uint8: 2,
    torch.int16: 2,
    torch.int32: 1,
    torch.int64: 2,
}
# int32: BLOCK=64 with unsigned Euclid measured fastest at 1G and 4096^2 on
# C550 (79.8 vs 79.9ms at 1G for B128; B32 is 2x worse from grid overhead).
# int16: B64 (46.65 vs 46.66ms at 1G, 0.735 vs 0.735ms at 4096^2) and gives
# finer grid granularity on tiny shapes.
_BLOCK_BY_OUT = {
    torch.int8: 256,
    torch.uint8: 256,
    torch.int16: 64,
    torch.int32: 64,
    torch.int64: 256,
}


@triton.jit
def _lcm_core(
    a0,
    b0,
    CT: tl.constexpr,
    OT: tl.constexpr,
    UT: tl.constexpr,
    P1: tl.constexpr,
    CHUNK: tl.constexpr,
):
    # torch.lcm semantics: g = gcd(|a|,|b|); res = |a/g*b| (0 if a==0 or b==0).
    # gcd via Euclid entirely in UNSIGNED arithmetic: bit-preserving cast of the
    # inputs to UT, then unsigned abs (negate-when-negative, so abs(INT_MIN) =
    # 2^(bits-1) exactly). Two reasons:
    #  - metax's remainder instruction miscomputes negative dividends, so
    #    keeping every Euclid operand non-negative is exact by construction
    #    (no INT_MIN substitution needed);
    #  - uint32 div/mod lowers to a much faster path than int32 on C550
    #    (measured 79.8 vs 101.7ms on 1G int32, 4096^2 1.24 vs 1.65ms).
    # The final quotient |a0|//g is also computed in unsigned (see epilogue).
    # Phase 1 runs P1 static masked Euclid steps with no block reduction;
    # phase 2 is a block-convergence while loop with CHUNK static steps per
    # reduction. All steps are masked so any over-run is exact.
    # int8/int16 compute in int32 then truncate (matches torch C++ wrap).
    if a0.dtype != CT:
        a0 = tl.cast(a0, CT)
    if b0.dtype != CT:
        b0 = tl.cast(b0, CT)
    au = tl.cast(a0, UT)
    bu = tl.cast(b0, UT)
    a = tl.where(a0 < 0, 0 - au, au)
    b = tl.where(b0 < 0, 0 - bu, bu)
    for _ in tl.static_range(P1):
        bs = tl.where(b != 0, b, 1)
        nb = a % bs
        a = tl.where(b != 0, b, a)
        b = nb
    while tl.max(b) != 0:
        for _ in tl.static_range(CHUNK):
            bs = tl.where(b != 0, b, 1)
            nb = a % bs
            a = tl.where(b != 0, b, a)
            b = nb
    # Epilogue: res = |(a0/g)*b0| with int32 wrap, matching torch. The quotient
    # is computed in unsigned arithmetic because unsigned div is faster than
    # signed div on C550 (1G int16: 45.35 vs 46.16ms, int32: 79.5 vs 80.2ms).
    # The sign of the quotient and the a0==0/b0==0 zero cases are both absorbed
    # by the outer abs (|sgn(a0)*q_abs*b0| = |q_abs*b0|, and q_abs = 0 or
    # q*b0 = 0 when an input is 0), so no sign-correct select or zero mask is
    # needed. Verified 0 mismatches vs torch.lcm on all dtypes incl. INT_MIN.
    au0 = tl.where(a0 < 0, 0 - au, au)
    as_ = tl.where(a != 0, a, 1)
    q_abs = au0 // as_
    res = tl.abs(tl.cast(q_abs, CT) * b0)
    if CT != OT:
        res = tl.cast(res, OT)
    return res


@triton.jit
def _lcm_1d(
    x_ptr,
    y_ptr,
    out_ptr,
    numel,
    CT: tl.constexpr,
    OT: tl.constexpr,
    UT: tl.constexpr,
    P1: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < numel
    a0 = tl.load(x_ptr + offs, mask=mask, other=0)
    b0 = tl.load(y_ptr + offs, mask=mask, other=0)
    res = _lcm_core(a0, b0, CT, OT, UT, P1, CHUNK)
    tl.store(out_ptr + offs, res, mask=mask)


@triton.jit
def _lcm_bcast(
    x_ptr,
    y_ptr,
    out_ptr,
    numel,
    shape_ptr,
    xst_ptr,
    yst_ptr,
    CT: tl.constexpr,
    OT: tl.constexpr,
    UT: tl.constexpr,
    P1: tl.constexpr,
    CHUNK: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < numel
    rem = offs
    xo = tl.zeros([BLOCK], dtype=tl.int64)
    yo = tl.zeros([BLOCK], dtype=tl.int64)
    for d in tl.static_range(NDIM):
        s = tl.load(shape_ptr + d)
        c = rem % s
        rem = rem // s
        xo += c * tl.load(xst_ptr + d)
        yo += c * tl.load(yst_ptr + d)
    a0 = tl.load(x_ptr + xo, mask=mask, other=0)
    b0 = tl.load(y_ptr + yo, mask=mask, other=0)
    res = _lcm_core(a0, b0, CT, OT, UT, P1, CHUNK)
    tl.store(out_ptr + offs, res, mask=mask)


def _broadcast_eff_strides(shape, strides, out_ndim):
    # Align input dims to the right of the output. Effective stride is 0 for
    # broadcast (size-1) or absent dims, else the actual stride.
    pad = out_ndim - len(shape)
    eff = [0] * pad
    for j in range(len(shape)):
        if shape[j] == 1:
            eff.append(0)
        else:
            eff.append(strides[j])
    # innermost-first order for the kernel
    return list(reversed(eff))


def lcm(self, other):
    out_dtype = torch.promote_types(self.dtype, other.dtype)
    out_shape = torch.broadcast_shapes(self.shape, other.shape)
    numel = 1
    for s in out_shape:
        numel *= s
    out = torch.empty(out_shape, dtype=out_dtype, device=self.device)
    if numel == 0:
        return out

    if out_dtype == torch.int64:
        ct = tl.int64
        ut = tl.uint64
    else:
        ct = tl.int32
        ut = tl.uint32
    ot = _TL_TYPES[out_dtype]
    BLOCK = _BLOCK_BY_OUT[out_dtype]
    WARPS = 1
    CHUNK = _CHUNK_BY_OUT[out_dtype]
    P1 = _P1_BY_OUT[out_dtype]
    # int16: CHUNK=1 is faster on large workloads (cheap B64 reduction), while
    # CHUNK=2 is the proven value on tiny launch-bound shapes; dispatch on numel.
    if out_dtype == torch.int16 and numel >= _LARGE_NUMEL:
        CHUNK = 1

    if self.shape == other.shape and self.is_contiguous() and other.is_contiguous():
        grid = (triton.cdiv(numel, BLOCK),)
        _lcm_1d[grid](
            self,
            other,
            out,
            numel,
            CT=ct,
            OT=ot,
            UT=ut,
            P1=P1,
            CHUNK=CHUNK,
            BLOCK=BLOCK,
            num_warps=WARPS,
        )
    else:
        ndim = len(out_shape)
        shape_rev = torch.tensor(
            list(reversed(out_shape)), dtype=torch.int64, device=self.device
        )
        xst_rev = torch.tensor(
            _broadcast_eff_strides(self.shape, self.stride(), ndim),
            dtype=torch.int64,
            device=self.device,
        )
        yst_rev = torch.tensor(
            _broadcast_eff_strides(other.shape, other.stride(), ndim),
            dtype=torch.int64,
            device=self.device,
        )
        grid = (triton.cdiv(numel, BLOCK),)
        _lcm_bcast[grid](
            self,
            other,
            out,
            numel,
            shape_rev,
            xst_rev,
            yst_rev,
            CT=ct,
            OT=ot,
            UT=ut,
            P1=P1,
            CHUNK=CHUNK,
            NDIM=ndim,
            BLOCK=BLOCK,
            num_warps=WARPS,
        )
    return out
