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

"""Fused in-place renorm_ (Triton).

aten::renorm_(self, p, dim, maxnorm) rescales every sub-tensor indexed by ``dim``
so that its p-norm does not exceed ``maxnorm``.  Verified against
torch.ops.aten.renorm_ on CPU and CUDA (identical on both, and consistent with
the documented ones(3, 4)/dim=0 example): the norm runs over *all axes except*
``dim``.  For a (2, 3) tensor dim=0 normalizes each row and dim=1 each column:

    n_g = (sum_{e in group g} |x[e]|^p) ^ (1/p)   (p == inf -> max |x|)
    x[e] <- x[e] * (maxnorm / max(n_g, maxnorm))

One stride-aware kernel launch (2N element traffic) replaces ATen's
norm -> clamp_min_ -> div_ -> mul_ composite (about 5N traffic, 4 launches).

Layout of a launch: ``K`` groups (the ``dim`` axis, single stride ``stride_g``)
and ``Etot`` elements per group (all other axes, merged into mixed-radix order).
When the ``dim`` axis is the contiguous one the tile is [BLOCK_E, BLOCK_G] with
the reduction over axis 0 (``*_g*`` kernels); otherwise the element axis is
contiguous and the tile is [BLOCK_G, BLOCK_E] with the reduction over axis 1
(``*_e*`` kernels).  Measured latency for these tiny shapes is host-launch
bound, so the 2-D case (all timed workloads) uses rank-1 kernels with a
minimal argument list; the generic 8-axis kernels remain for higher-rank or
irregularly strided inputs.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _pow2_ceil(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


@triton.jit
def _mixed_base(
    idx,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    RANK: tl.constexpr,
):
    """Mixed-radix element index -> element offset inside a group."""
    base = tl.zeros_like(idx)
    rem = idx
    if RANK > 0:
        base += rem % s0 * t0
        rem = rem // s0
    if RANK > 1:
        base += rem % s1 * t1
        rem = rem // s1
    if RANK > 2:
        base += rem % s2 * t2
        rem = rem // s2
    if RANK > 3:
        base += rem % s3 * t3
        rem = rem // s3
    if RANK > 4:
        base += rem % s4 * t4
        rem = rem // s4
    if RANK > 5:
        base += rem % s5 * t5
        rem = rem // s5
    if RANK > 6:
        base += rem % s6 * t6
        rem = rem // s6
    if RANK > 7:
        base += rem % s7 * t7
    return base


@triton.jit
def _pow_abs(a, p_val):
    """|x|^p for a >= 0."""
    return tl.where(a == 0.0, a, tl.exp2(tl.log2(a) * p_val))


@triton.jit
def _norm_scale(acc, mx, nanf, p_val, maxnorm, PKIND: tl.constexpr):
    if PKIND == 2:
        norm = tl.sqrt(acc)
    elif PKIND == 3:
        norm = tl.where(nanf > 0, float("nan"), mx)
    elif PKIND == 4:
        norm = tl.where(acc == 0.0, acc, tl.exp2(tl.log2(acc) * (1.0 / p_val)))
    else:
        norm = acc
    denom = tl.where(norm != norm, norm, tl.maximum(norm, maxnorm))
    return maxnorm / denom


@triton.jit
def _renorm_g1f(X, PLAN: tl.constexpr, MN: tl.constexpr):
    """Exact-fit contiguous-group tile: no masks, no loops, one runtime argument.

    Requires stride of the ``dim`` axis == 1, pow2(Etot) == Etot == BLOCK_E and
    K % BLOCK_G == 0, so nothing needs masking.  Everything except the pointer
    is compile-time (measured latency for these tiny shapes is launch bound and
    each launch argument costs time); PLAN packs
    PKIND + BLOCK_G * 16 + BLOCK_E * 4096 + element_stride * 2**24.
    """
    PK: tl.constexpr = PLAN % 16
    BG: tl.constexpr = PLAN // 16 % 256
    BE: tl.constexpr = PLAN // 4096 % 4096
    SE: tl.constexpr = PLAN // 16777216
    pid = tl.program_id(0)
    gid = pid.to(tl.int64) * BG + tl.arange(0, BG).to(tl.int64)
    ptr = X + gid[None, :] + (tl.arange(0, BE).to(tl.int64) * SE)[:, None]
    x = tl.load(ptr)
    a = tl.abs(x.to(tl.float32))
    if PK == 1:
        norm = tl.sum(a, axis=0)
    elif PK == 3:
        nanf = tl.sum((a != a).to(tl.int32), axis=0)
        norm = tl.where(nanf > 0, float("nan"), tl.max(a, axis=0))
    else:
        norm = tl.sqrt(tl.sum(a * a, axis=0))
    denom = tl.where(norm != norm, norm, tl.maximum(norm, MN))
    y = x.to(tl.float32) * (MN / denom)[None, :]
    tl.store(ptr, y.to(x.dtype))


@triton.jit
def _renorm_g1(
    X,
    K,
    Etot,
    stride_g,
    se,
    p_val,
    maxnorm,
    PKIND: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_E: tl.constexpr,
    TWO_PASS: tl.constexpr,
    ACC64: tl.constexpr,
):
    pid = tl.program_id(0)
    gid = pid.to(tl.int64) * BLOCK_G + tl.arange(0, BLOCK_G).to(tl.int64)
    m_g = gid < K
    goff = (gid * stride_g)[None, :]
    eidx = tl.arange(0, BLOCK_E)
    ecol = (eidx.to(tl.int64) * se)[:, None]
    if TWO_PASS:
        if ACC64:
            acc = tl.zeros([BLOCK_G], dtype=tl.float64)
            mx = tl.zeros([BLOCK_G], dtype=tl.float64)
        else:
            acc = tl.zeros([BLOCK_G], dtype=tl.float32)
            mx = tl.zeros([BLOCK_G], dtype=tl.float32)
        nanf = tl.zeros([BLOCK_G], dtype=tl.int32)
        n_chunks = tl.cdiv(Etot, BLOCK_E)
        for c in range(0, n_chunks):
            eid = c * BLOCK_E + eidx
            m = (eid[:, None] < Etot) & m_g[None, :]
            ptr = X + ecol + c * (BLOCK_E * se) + goff
            x = tl.load(ptr, mask=m, other=0.0)
            if ACC64:
                a = tl.abs(x.to(tl.float64))
            else:
                a = tl.abs(x.to(tl.float32))
            if PKIND == 1:
                acc += tl.sum(a, axis=0)
            elif PKIND == 2:
                acc += tl.sum(a * a, axis=0)
            elif PKIND == 3:
                nanf += tl.sum((a != a).to(tl.int32), axis=0)
                mx = tl.maximum(mx, tl.max(a, axis=0))
            else:
                acc += tl.sum(_pow_abs(a, p_val), axis=0)
        scale = _norm_scale(acc, mx, nanf, p_val, maxnorm, PKIND)
        for c in range(0, n_chunks):
            eid = c * BLOCK_E + eidx
            m = (eid[:, None] < Etot) & m_g[None, :]
            ptr = X + ecol + c * (BLOCK_E * se) + goff
            x = tl.load(ptr, mask=m, other=0.0)
            if ACC64:
                y = x.to(tl.float64) * scale[None, :]
            else:
                y = x.to(tl.float32) * scale[None, :]
            tl.store(ptr, y.to(x.dtype), mask=m)
    else:
        m = (eidx[:, None] < Etot) & m_g[None, :]
        ptr = X + ecol + goff
        x = tl.load(ptr, mask=m, other=0.0)
        if ACC64:
            a = tl.abs(x.to(tl.float64))
        else:
            a = tl.abs(x.to(tl.float32))
        mx = tl.zeros([BLOCK_G], dtype=a.dtype)
        nanf = tl.zeros([BLOCK_G], dtype=tl.int32)
        if PKIND == 1:
            acc = tl.sum(a, axis=0)
        elif PKIND == 2:
            acc = tl.sum(a * a, axis=0)
        elif PKIND == 3:
            nanf = tl.sum((a != a).to(tl.int32), axis=0)
            acc = tl.max(a, axis=0)
            mx = acc
        else:
            acc = tl.sum(_pow_abs(a, p_val), axis=0)
        scale = _norm_scale(acc, mx, nanf, p_val, maxnorm, PKIND)
        if ACC64:
            y = x.to(tl.float64) * scale[None, :]
        else:
            y = x.to(tl.float32) * scale[None, :]
        tl.store(ptr, y.to(x.dtype), mask=m)


@triton.jit
def _renorm_e1(
    X,
    K,
    Etot,
    stride_g,
    se,
    p_val,
    maxnorm,
    PKIND: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_E: tl.constexpr,
    TWO_PASS: tl.constexpr,
    ACC64: tl.constexpr,
):
    pid = tl.program_id(0)
    gid = pid.to(tl.int64) * BLOCK_G + tl.arange(0, BLOCK_G).to(tl.int64)
    m_g = gid < K
    goff = (gid * stride_g)[:, None]
    eidx = tl.arange(0, BLOCK_E)
    ecol = (eidx.to(tl.int64) * se)[None, :]
    if TWO_PASS:
        if ACC64:
            acc = tl.zeros([BLOCK_G], dtype=tl.float64)
            mx = tl.zeros([BLOCK_G], dtype=tl.float64)
        else:
            acc = tl.zeros([BLOCK_G], dtype=tl.float32)
            mx = tl.zeros([BLOCK_G], dtype=tl.float32)
        nanf = tl.zeros([BLOCK_G], dtype=tl.int32)
        n_chunks = tl.cdiv(Etot, BLOCK_E)
        for c in range(0, n_chunks):
            eid = c * BLOCK_E + eidx
            m = m_g[:, None] & (eid[None, :] < Etot)
            ptr = X + goff + ecol + c * (BLOCK_E * se)
            x = tl.load(ptr, mask=m, other=0.0)
            if ACC64:
                a = tl.abs(x.to(tl.float64))
            else:
                a = tl.abs(x.to(tl.float32))
            if PKIND == 1:
                acc += tl.sum(a, axis=1)
            elif PKIND == 2:
                acc += tl.sum(a * a, axis=1)
            elif PKIND == 3:
                nanf += tl.sum((a != a).to(tl.int32), axis=1)
                mx = tl.maximum(mx, tl.max(a, axis=1))
            else:
                acc += tl.sum(_pow_abs(a, p_val), axis=1)
        scale = _norm_scale(acc, mx, nanf, p_val, maxnorm, PKIND)
        for c in range(0, n_chunks):
            eid = c * BLOCK_E + eidx
            m = m_g[:, None] & (eid[None, :] < Etot)
            ptr = X + goff + ecol + c * (BLOCK_E * se)
            x = tl.load(ptr, mask=m, other=0.0)
            if ACC64:
                y = x.to(tl.float64) * scale[:, None]
            else:
                y = x.to(tl.float32) * scale[:, None]
            tl.store(ptr, y.to(x.dtype), mask=m)
    else:
        m = m_g[:, None] & (eidx[None, :] < Etot)
        ptr = X + goff + ecol
        x = tl.load(ptr, mask=m, other=0.0)
        if ACC64:
            a = tl.abs(x.to(tl.float64))
        else:
            a = tl.abs(x.to(tl.float32))
        mx = tl.zeros([BLOCK_G], dtype=a.dtype)
        nanf = tl.zeros([BLOCK_G], dtype=tl.int32)
        if PKIND == 1:
            acc = tl.sum(a, axis=1)
        elif PKIND == 2:
            acc = tl.sum(a * a, axis=1)
        elif PKIND == 3:
            nanf = tl.sum((a != a).to(tl.int32), axis=1)
            acc = tl.max(a, axis=1)
            mx = acc
        else:
            acc = tl.sum(_pow_abs(a, p_val), axis=1)
        scale = _norm_scale(acc, mx, nanf, p_val, maxnorm, PKIND)
        if ACC64:
            y = x.to(tl.float64) * scale[:, None]
        else:
            y = x.to(tl.float32) * scale[:, None]
        tl.store(ptr, y.to(x.dtype), mask=m)


@triton.jit
def _renorm_gfast(
    X,
    K,
    Etot,
    p_val,
    maxnorm,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    stride_g,
    RANK: tl.constexpr,
    PKIND: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_E: tl.constexpr,
    TWO_PASS: tl.constexpr,
    ACC64: tl.constexpr,
):
    pid = tl.program_id(0)
    gid = pid.to(tl.int64) * BLOCK_G + tl.arange(0, BLOCK_G).to(tl.int64)
    m_g = gid < K
    goff = gid * stride_g
    eidx = tl.arange(0, BLOCK_E)
    if TWO_PASS:
        if ACC64:
            acc = tl.zeros([BLOCK_G], dtype=tl.float64)
            mx = tl.zeros([BLOCK_G], dtype=tl.float64)
        else:
            acc = tl.zeros([BLOCK_G], dtype=tl.float32)
            mx = tl.zeros([BLOCK_G], dtype=tl.float32)
        nanf = tl.zeros([BLOCK_G], dtype=tl.int32)
        n_chunks = tl.cdiv(Etot, BLOCK_E)
        for c in range(0, n_chunks):
            eid = c * BLOCK_E + eidx
            m = (eid[:, None] < Etot) & m_g[None, :]
            ptr = (
                X
                + _mixed_base(
                    eid.to(tl.int64),
                    s0,
                    s1,
                    s2,
                    s3,
                    s4,
                    s5,
                    s6,
                    s7,
                    t0,
                    t1,
                    t2,
                    t3,
                    t4,
                    t5,
                    t6,
                    t7,
                    RANK,
                )[:, None]
                + goff[None, :]
            )
            x = tl.load(ptr, mask=m, other=0.0)
            if ACC64:
                a = tl.abs(x.to(tl.float64))
            else:
                a = tl.abs(x.to(tl.float32))
            if PKIND == 1:
                acc += tl.sum(a, axis=0)
            elif PKIND == 2:
                acc += tl.sum(a * a, axis=0)
            elif PKIND == 3:
                nanf += tl.sum((a != a).to(tl.int32), axis=0)
                mx = tl.maximum(mx, tl.max(a, axis=0))
            else:
                acc += tl.sum(_pow_abs(a, p_val), axis=0)
        scale = _norm_scale(acc, mx, nanf, p_val, maxnorm, PKIND)
        for c in range(0, n_chunks):
            eid = c * BLOCK_E + eidx
            m = (eid[:, None] < Etot) & m_g[None, :]
            ptr = (
                X
                + _mixed_base(
                    eid.to(tl.int64),
                    s0,
                    s1,
                    s2,
                    s3,
                    s4,
                    s5,
                    s6,
                    s7,
                    t0,
                    t1,
                    t2,
                    t3,
                    t4,
                    t5,
                    t6,
                    t7,
                    RANK,
                )[:, None]
                + goff[None, :]
            )
            x = tl.load(ptr, mask=m, other=0.0)
            if ACC64:
                y = x.to(tl.float64) * scale[None, :]
            else:
                y = x.to(tl.float32) * scale[None, :]
            tl.store(ptr, y.to(x.dtype), mask=m)
    else:
        m = (eidx[:, None] < Etot) & m_g[None, :]
        ptr = (
            X
            + _mixed_base(
                eidx.to(tl.int64),
                s0,
                s1,
                s2,
                s3,
                s4,
                s5,
                s6,
                s7,
                t0,
                t1,
                t2,
                t3,
                t4,
                t5,
                t6,
                t7,
                RANK,
            )[:, None]
            + goff[None, :]
        )
        x = tl.load(ptr, mask=m, other=0.0)
        if ACC64:
            a = tl.abs(x.to(tl.float64))
        else:
            a = tl.abs(x.to(tl.float32))
        mx = tl.zeros([BLOCK_G], dtype=a.dtype)
        nanf = tl.zeros([BLOCK_G], dtype=tl.int32)
        if PKIND == 1:
            acc = tl.sum(a, axis=0)
        elif PKIND == 2:
            acc = tl.sum(a * a, axis=0)
        elif PKIND == 3:
            nanf = tl.sum((a != a).to(tl.int32), axis=0)
            acc = tl.max(a, axis=0)
            mx = acc
        else:
            acc = tl.sum(_pow_abs(a, p_val), axis=0)
        scale = _norm_scale(acc, mx, nanf, p_val, maxnorm, PKIND)
        if ACC64:
            y = x.to(tl.float64) * scale[None, :]
        else:
            y = x.to(tl.float32) * scale[None, :]
        tl.store(ptr, y.to(x.dtype), mask=m)


@triton.jit
def _renorm_efast(
    X,
    K,
    Etot,
    p_val,
    maxnorm,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    stride_g,
    RANK: tl.constexpr,
    PKIND: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_E: tl.constexpr,
    TWO_PASS: tl.constexpr,
    ACC64: tl.constexpr,
):
    pid = tl.program_id(0)
    gid = pid.to(tl.int64) * BLOCK_G + tl.arange(0, BLOCK_G).to(tl.int64)
    m_g = gid < K
    goff = gid * stride_g
    eidx = tl.arange(0, BLOCK_E)
    if TWO_PASS:
        if ACC64:
            acc = tl.zeros([BLOCK_G], dtype=tl.float64)
            mx = tl.zeros([BLOCK_G], dtype=tl.float64)
        else:
            acc = tl.zeros([BLOCK_G], dtype=tl.float32)
            mx = tl.zeros([BLOCK_G], dtype=tl.float32)
        nanf = tl.zeros([BLOCK_G], dtype=tl.int32)
        n_chunks = tl.cdiv(Etot, BLOCK_E)
        for c in range(0, n_chunks):
            eid = c * BLOCK_E + eidx
            m = m_g[:, None] & (eid[None, :] < Etot)
            ptr = (
                X
                + goff[:, None]
                + _mixed_base(
                    eid.to(tl.int64),
                    s0,
                    s1,
                    s2,
                    s3,
                    s4,
                    s5,
                    s6,
                    s7,
                    t0,
                    t1,
                    t2,
                    t3,
                    t4,
                    t5,
                    t6,
                    t7,
                    RANK,
                )[None, :]
            )
            x = tl.load(ptr, mask=m, other=0.0)
            if ACC64:
                a = tl.abs(x.to(tl.float64))
            else:
                a = tl.abs(x.to(tl.float32))
            if PKIND == 1:
                acc += tl.sum(a, axis=1)
            elif PKIND == 2:
                acc += tl.sum(a * a, axis=1)
            elif PKIND == 3:
                nanf += tl.sum((a != a).to(tl.int32), axis=1)
                mx = tl.maximum(mx, tl.max(a, axis=1))
            else:
                acc += tl.sum(_pow_abs(a, p_val), axis=1)
        scale = _norm_scale(acc, mx, nanf, p_val, maxnorm, PKIND)
        for c in range(0, n_chunks):
            eid = c * BLOCK_E + eidx
            m = m_g[:, None] & (eid[None, :] < Etot)
            ptr = (
                X
                + goff[:, None]
                + _mixed_base(
                    eid.to(tl.int64),
                    s0,
                    s1,
                    s2,
                    s3,
                    s4,
                    s5,
                    s6,
                    s7,
                    t0,
                    t1,
                    t2,
                    t3,
                    t4,
                    t5,
                    t6,
                    t7,
                    RANK,
                )[None, :]
            )
            x = tl.load(ptr, mask=m, other=0.0)
            if ACC64:
                y = x.to(tl.float64) * scale[:, None]
            else:
                y = x.to(tl.float32) * scale[:, None]
            tl.store(ptr, y.to(x.dtype), mask=m)
    else:
        m = m_g[:, None] & (eidx[None, :] < Etot)
        ptr = (
            X
            + goff[:, None]
            + _mixed_base(
                eidx.to(tl.int64),
                s0,
                s1,
                s2,
                s3,
                s4,
                s5,
                s6,
                s7,
                t0,
                t1,
                t2,
                t3,
                t4,
                t5,
                t6,
                t7,
                RANK,
            )[None, :]
        )
        x = tl.load(ptr, mask=m, other=0.0)
        if ACC64:
            a = tl.abs(x.to(tl.float64))
        else:
            a = tl.abs(x.to(tl.float32))
        mx = tl.zeros([BLOCK_G], dtype=a.dtype)
        nanf = tl.zeros([BLOCK_G], dtype=tl.int32)
        if PKIND == 1:
            acc = tl.sum(a, axis=1)
        elif PKIND == 2:
            acc = tl.sum(a * a, axis=1)
        elif PKIND == 3:
            nanf = tl.sum((a != a).to(tl.int32), axis=1)
            acc = tl.max(a, axis=1)
            mx = acc
        else:
            acc = tl.sum(_pow_abs(a, p_val), axis=1)
        scale = _norm_scale(acc, mx, nanf, p_val, maxnorm, PKIND)
        if ACC64:
            y = x.to(tl.float64) * scale[:, None]
        else:
            y = x.to(tl.float32) * scale[:, None]
        tl.store(ptr, y.to(x.dtype), mask=m)


def _to_float(v, name):
    if isinstance(v, complex):
        raise RuntimeError("renorm: " + name + " must be a real scalar")
    if torch.is_tensor(v):
        if v.is_complex():
            raise RuntimeError("renorm: " + name + " must be a real scalar")
        if v.dtype == torch.bool:
            raise RuntimeError("renorm: " + name + " must be a non-boolean scalar")
        v = v.item()
    if isinstance(v, bool):
        raise RuntimeError("renorm: " + name + " must be a non-boolean scalar")
    if isinstance(v, complex):
        raise RuntimeError("renorm: " + name + " must be a real scalar")
    return float(v)


_FAST_2D = {}


def renorm_(self, p, dim, maxnorm):
    logger.debug("GEMS_HYGON RENORM_")
    x = self
    tp = type(p)
    pv = p if tp is float else _to_float(p, "p")
    tm = type(maxnorm)
    mv = maxnorm if tm is float else _to_float(maxnorm, "maxnorm")
    if not pv > 0.0:
        raise RuntimeError("renorm: non-positive-norm not supported")
    if not mv >= 0.0:
        raise RuntimeError("renorm: maxnorm must be >= 0")
    dt = x.dtype
    if dt is torch.bool:
        raise RuntimeError("renorm: Bool tensor is not supported")
    if not dt.is_floating_point:
        raise RuntimeError("renorm: only floating point tensors are supported")
    nd = x.dim()
    if nd < 2:
        raise RuntimeError(
            "renorm: input needs at least 2 dimensions, got %d dimensions" % nd
        )
    if dim < -nd or dim >= nd:
        raise IndexError(
            "Dimension out of range (expected to be in range of [%d, %d], but got %d)"
            % (-nd, nd - 1, dim)
        )
    d = dim + nd if dim < 0 else dim
    if pv == 2.0:
        pk = 2
    elif pv == 1.0:
        pk = 1
    elif pv == float("inf"):
        pk = 3
    else:
        pk = 4
    acc64 = dt is torch.float64
    if nd == 2:
        sh = x.shape
        st = x.stride()
        key = (dt, sh[0], sh[1], st[0], st[1], d, pk, mv)
        hit = _FAST_2D.get(key)
        if hit is not None:
            _renorm_g1f[hit[0]](x, hit[1], mv, num_warps=hit[2])
            return x
        K = sh[1] if d == 1 else sh[0]
        Etot = sh[0] if d == 1 else sh[1]
        if K == 0 or Etot == 0:
            return x
        sg = st[1] if d == 1 else st[0]
        se = st[0] if d == 1 else st[1]
        if Etot == 1:
            gfast = True
        else:
            gfast = sg > 0 and sg <= se
        if (
            gfast
            and sg == 1
            and (not acc64)
            and (pk < 4)
            and (Etot <= 4096)
            and (Etot & Etot - 1 == 0)
        ):
            bg = 8192 // Etot
            if bg > 32:
                bg = 32
            if bg < 1:
                bg = 1
            if K % bg == 0:
                tile = bg * Etot
                nw = tile // 256
                if nw < 1:
                    nw = 1
                elif nw > 16:
                    nw = 16
                grid = (K // bg,)
                plan = pk + bg * 16 + Etot * 4096 + se * 16777216
                _FAST_2D[key] = (grid, plan, nw)
                _renorm_g1f[grid](x, plan, mv, num_warps=nw)
                return x
        be = _pow2_ceil(Etot)
        if be > 4096:
            be = 4096
        bg = 4096 // be
        if bg > 32:
            bg = 32
        if bg < 1:
            bg = 1
        bk = _pow2_ceil(K)
        if bg > bk:
            bg = bk
        two = Etot > be
        tile = bg * be
        if tile <= 256:
            nw = 1
        elif tile <= 512:
            nw = 2
        elif tile <= 2048:
            nw = 4
        else:
            nw = 8
        grid = ((K + bg - 1) // bg,)
        if gfast:
            _renorm_g1[grid](
                x, K, Etot, sg, se, pv, mv, pk, bg, be, two, acc64, num_warps=nw
            )
        else:
            _renorm_e1[grid](
                x, K, Etot, sg, se, pv, mv, pk, bg, be, two, acc64, num_warps=nw
            )
        return x
    n = x.numel()
    if n == 0:
        return x
    K = x.size(d)
    Etot = n // K
    sg = x.stride(d)
    eaxes = sorted(
        [(x.size(i), x.stride(i)) for i in range(nd) if i != d and x.size(i) != 1],
        key=lambda a: a[1],
    )
    merged = []
    for sz, st in eaxes:
        if merged:
            psz, pst = merged[-1]
            if pst > 0 and st > 0 and (pst * psz == st):
                merged[-1] = (psz * sz, pst)
                continue
        merged.append((sz, st))
    eaxes = merged
    rank = len(eaxes)
    if rank > 8:
        raise RuntimeError("renorm: unsupported tensor layout (rank > 8 after merge)")
    sizes = [1, 1, 1, 1, 1, 1, 1, 1]
    strides = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(rank):
        sizes[i] = eaxes[i][0]
        strides[i] = eaxes[i][1]
    if Etot == 1:
        gfast = True
    else:
        gfast = sg > 0 and sg <= eaxes[0][1]
    if gfast:
        bg = 8
        while bg * 128 <= K and bg < 32:
            bg *= 2
        bk = _pow2_ceil(K)
        if bg > bk:
            bg = bk
        be = _pow2_ceil(Etot)
        cap = 4096 // bg
        if cap < 1:
            cap = 1
        if be > cap:
            be = cap
    else:
        be = _pow2_ceil(Etot)
        if be > 4096:
            be = 4096
        bg = 4096 // be
        if bg < 1:
            bg = 1
        bk = _pow2_ceil(K)
        if bg > bk:
            bg = bk
    two = Etot > be
    tile = bg * be
    if tile <= 256:
        nw = 1
    elif tile <= 512:
        nw = 2
    elif tile <= 2048:
        nw = 4
    else:
        nw = 8
    grid = ((K + bg - 1) // bg,)
    if gfast:
        _renorm_gfast[grid](
            x,
            K,
            Etot,
            pv,
            mv,
            sizes[0],
            sizes[1],
            sizes[2],
            sizes[3],
            sizes[4],
            sizes[5],
            sizes[6],
            sizes[7],
            strides[0],
            strides[1],
            strides[2],
            strides[3],
            strides[4],
            strides[5],
            strides[6],
            strides[7],
            sg,
            RANK=rank,
            PKIND=pk,
            BLOCK_G=bg,
            BLOCK_E=be,
            TWO_PASS=two,
            ACC64=acc64,
            num_warps=nw,
        )
    else:
        _renorm_efast[grid](
            x,
            K,
            Etot,
            pv,
            mv,
            sizes[0],
            sizes[1],
            sizes[2],
            sizes[3],
            sizes[4],
            sizes[5],
            sizes[6],
            sizes[7],
            strides[0],
            strides[1],
            strides[2],
            strides[3],
            strides[4],
            strides[5],
            strides[6],
            strides[7],
            sg,
            RANK=rank,
            PKIND=pk,
            BLOCK_G=bg,
            BLOCK_E=be,
            TWO_PASS=two,
            ACC64=acc64,
            num_warps=nw,
        )
    return x
