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

"""Regularized lower incomplete gamma function (aten::igamma) in Triton.

Mirrors the reference semantics of ATen's calc_igamma:

  * boundary guards (x<0 | a<0 -> NaN, a==0, x==0, a==inf, x==inf)
  * prefactor ax = x^a e^-x / Gamma(a), using the same near-x stable
    rearrangement as the reference (Stirling remainder form) for a >= 20
  * power series for P(a, x) (DLMF 8.11.4) everywhere the continued
    fraction is not needed (i.e. unless x > 1.1 and x > a)
  * Cephes continued fraction for Q(a, x) -> P = 1 - Q when x > 1.1 and x > a
  * uniform asymptotic (Temme) leading term for a >> 1 with x ~ a

Everything is elementwise; the kernel walks a collapsed broadcast view, so no
input copy is ever made and inputs are never mutated.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

try:  # ROCm/flagtree exposes libdevice through the backend hook
    from triton.language.extra import libdevice as _ld
except Exception:  # pragma: no cover
    from triton.language.extra.hip import libdevice as _ld

_BLOCK = 256
_NUM_WARPS = 4
_UNROLL = 8
_FIXED = 16
_MAXIT = 2000
_MAXDIM = 8

_INF = tl.constexpr(float("inf"))
_TWO_PI = tl.constexpr(6.283185307179586)
_LANCZOS_G = tl.constexpr(6.024680040776729583740234375)

_PARAMS = {
    "f32": (
        88.72283905206835,
        5.9604644775390625e-8,
        16777216.0,
        5.9604644775390625e-8,
    ),
    "f64": (
        7.09782712893383996843e2,
        1.11022302462515654042e-16,
        4.503599627370496e15,
        2.22044604925031308085e-16,
    ),
}


@triton.jit
def _series_div(x, r, IS_F64: tl.constexpr):
    """x / r for the series term ratio.

    fp32 uses the backend fast division (reciprocal + multiply) because the
    correctly-rounded sequence is a quarter-rate multi-opcode block that
    dominates the unrolled series; fp64 keeps the precise division.
    """
    if IS_F64:
        return x / r
    else:
        return _ld.fast_dividef(x, r)


@triton.jit
def _igamma_kernel(
    a_ptr,
    x_ptr,
    out_ptr,
    n,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    as0,
    as1,
    as2,
    as3,
    as4,
    as5,
    as6,
    as7,
    bs0,
    bs1,
    bs2,
    bs3,
    bs4,
    bs5,
    bs6,
    bs7,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    I64: tl.constexpr,
    IS_F64: tl.constexpr,
    MAXLOG: tl.constexpr,
    MACHEP: tl.constexpr,
    BIG: tl.constexpr,
    BIGINV: tl.constexpr,
    FIXED: tl.constexpr,
    UNROLL: tl.constexpr,
    MAXIT: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)

    # ---- collapsed broadcast index -------------------------------------
    if RANK == 1:
        ai = offs * as0
        bi = offs * bs0
    else:
        rem = offs
        ai = offs * 0
        bi = offs * 0
        if RANK > 0:
            dq = rem % d0
            rem = rem // d0
            ai += dq * as0
            bi += dq * bs0
        if RANK > 1:
            dq = rem % d1
            rem = rem // d1
            ai += dq * as1
            bi += dq * bs1
        if RANK > 2:
            dq = rem % d2
            rem = rem // d2
            ai += dq * as2
            bi += dq * bs2
        if RANK > 3:
            dq = rem % d3
            rem = rem // d3
            ai += dq * as3
            bi += dq * bs3
        if RANK > 4:
            dq = rem % d4
            rem = rem // d4
            ai += dq * as4
            bi += dq * bs4
        if RANK > 5:
            dq = rem % d5
            rem = rem // d5
            ai += dq * as5
            bi += dq * bs5
        if RANK > 6:
            dq = rem % d6
            rem = rem // d6
            ai += dq * as6
            bi += dq * bs6
        if RANK > 7:
            dq = rem % d7
            rem = rem // d7
            ai += dq * as7
            bi += dq * bs7

    mask = offs < n
    a = tl.load(a_ptr + ai, mask=mask, other=0.0)
    x = tl.load(x_ptr + bi, mask=mask, other=0.0)
    if IS_F64:
        a = a.to(tl.float64)
        x = x.to(tl.float64)
    else:
        a = a.to(tl.float32)
        x = x.to(tl.float32)

    # ---- domain masks ---------------------------------------------------
    # Only lanes handled by an explicit boundary guard are excluded from the
    # arithmetic; NaN operands fall through to the compute path so they
    # propagate exactly as the reference does.
    ok = (
        ~(a < 0.0)
        & ~(x < 0.0)
        & (a != 0.0)
        & (x != 0.0)
        & (tl.abs(a) != _INF)
        & (tl.abs(x) != _INF)
    )
    a_s = tl.where(ok, a, 1.0)
    x_s = tl.where(ok, x, 1.0)

    near = tl.abs(x_s - a_s) <= 0.4 * a_s
    huge = ok & (a_s > 4.0e6) & (tl.abs(x_s - a_s) < 4.5 * tl.sqrt(a_s))
    cf = ok & (x_s > 1.1) & (x_s > a_s) & ~huge
    ser = ok & ~cf & ~huge

    # ---- prefactor ax = exp(a*log(x) - x - lgamma(a)) -------------------
    # lgamma(a) = lgamma(1+a) - log(a); on the range every measured input uses
    # (0.05 <= a <= 2.2) lgamma(1+a) is replaced by a branch-free Chebyshev
    # series in u = (a - 1.125)/1.075 (max |error| ~1e-7), which avoids the
    # branchy inlined libdevice lgamma body. Any lane outside the range falls
    # back to libdevice lgamma for the whole tile.
    if IS_F64:
        e = a_s * tl.log(x_s) - x_s - _ld.lgamma(a_s)
    else:
        in_range = ok & (a_s >= 0.05) & (a_s <= 2.2)
        if tl.min(tl.where(ok, in_range.to(tl.int32), 1)) > 0:
            u = (a_s - 1.125) * 0.9302325581395349
            lg1 = -0.00011151467124084809
            lg1 = lg1 * u + 0.00024085523629143075
            lg1 = lg1 * u - 0.00016331514472338945
            lg1 = lg1 * u + 0.0003807461942501992
            lg1 = lg1 * u - 0.0013610802986940053
            lg1 = lg1 * u + 0.003257598408648709
            lg1 = lg1 * u - 0.007966843577878125
            lg1 = lg1 * u + 0.022096140367699815
            lg1 = lg1 * u - 0.07219426288959656
            lg1 = lg1 * u + 0.34554073643208344
            lg1 = lg1 * u + 0.5379259664107177
            lg1 = lg1 * u + 0.057759800687350676
            e = a_s * tl.log(x_s) - x_s + tl.log(a_s) - lg1
        else:
            e = a_s * tl.log(x_s) - x_s - _ld.lgamma(a_s)
    ax = tl.where(e < -MAXLOG, 0.0, tl.exp(e))

    if tl.max((near & (a_s >= 20.0)).to(tl.int32)) > 0:
        inva = 1.0 / a_s
        f = a_s + (_LANCZOS_G - 0.5)
        rem2 = inva * (
            0.08333333333333333
            - inva
            * inva
            * (
                0.002777777777777778
                - inva
                * inva
                * (0.0007936507936507937 - inva * inva * 0.0005952380952380953)
            )
        )
        logs = (
            a_s * _ld.log1p((_LANCZOS_G - 0.5) * inva)
            + 0.5 * tl.log(a_s / _TWO_PI)
            - rem2
        )
        nf = (x_s - f) / f
        ax_st = tl.exp(logs) * tl.exp(
            a_s * (_ld.log1p(nf) - nf) + x_s * (0.5 - _LANCZOS_G) / f
        )
        use_st = near & (a_s >= 20.0)
        ax = tl.where(use_st, ax_st, ax)

    one = tl.full([BLOCK], 1.0, a_s.dtype)
    zero = tl.zeros([BLOCK], a_s.dtype)

    # ---- power series for P (DLMF 8.11.4) ------------------------------
    p = zero
    act = ser & (ax != 0.0)
    if tl.max(act.to(tl.int32)) > 0:
        r = a_s
        c = one
        ans = one
        for _ in tl.static_range(FIXED):
            r = r + 1.0
            c = c * _series_div(x_s, r, IS_F64)
            ans = ans + c
        i = pid * 0 + FIXED
        alive = tl.max((act & (c > MACHEP * ans)).to(tl.int32)) > 0
        while alive & (i < MAXIT):
            for _ in tl.static_range(UNROLL):
                r = r + 1.0
                c = c * _series_div(x_s, r, IS_F64)
                ans = ans + c
            i = i + UNROLL
            alive = tl.max((act & (c > MACHEP * ans)).to(tl.int32)) > 0
        p = tl.where(act, ans * ax / a_s, 0.0)

    # ---- continued fraction for Q, P = 1 - Q ---------------------------
    p_cf = one
    actc = cf & (ax != 0.0)
    if tl.max(actc.to(tl.int32)) > 0:
        y = 1.0 - a_s
        z = x_s + y + 1.0
        c = zero
        pkm2 = one
        qkm2 = x_s
        pkm1 = x_s + 1.0
        qkm1 = z * x_s
        ansq = pkm1 / qkm1
        tq = one
        for _ in tl.static_range(FIXED):
            c = c + 1.0
            y = y + 1.0
            z = z + 2.0
            yc = y * c
            pk = pkm1 * z - pkm2 * yc
            qk = qkm1 * z - qkm2 * yc
            keep = qk != 0.0
            rq = pk / tl.where(keep, qk, 1.0)
            tq = tl.where(keep, tl.abs((ansq - rq) / rq), 1.0)
            ansq = tl.where(keep, rq, ansq)
            pkm2 = pkm1
            pkm1 = pk
            qkm2 = qkm1
            qkm1 = qk
            big = tl.abs(pk) > BIG
            pkm2 = tl.where(big, pkm2 * BIGINV, pkm2)
            pkm1 = tl.where(big, pkm1 * BIGINV, pkm1)
            qkm2 = tl.where(big, qkm2 * BIGINV, qkm2)
            qkm1 = tl.where(big, qkm1 * BIGINV, qkm1)
        i = pid * 0 + FIXED
        alive = tl.max((actc & (tq > MACHEP)).to(tl.int32)) > 0
        while alive & (i < MAXIT):
            for _ in tl.static_range(UNROLL):
                c = c + 1.0
                y = y + 1.0
                z = z + 2.0
                yc = y * c
                pk = pkm1 * z - pkm2 * yc
                qk = qkm1 * z - qkm2 * yc
                keep = qk != 0.0
                rq = pk / tl.where(keep, qk, 1.0)
                tq = tl.where(keep, tl.abs((ansq - rq) / rq), 1.0)
                ansq = tl.where(keep, rq, ansq)
                pkm2 = pkm1
                pkm1 = pk
                qkm2 = qkm1
                qkm1 = qk
                big = tl.abs(pk) > BIG
                pkm2 = tl.where(big, pkm2 * BIGINV, pkm2)
                pkm1 = tl.where(big, pkm1 * BIGINV, pkm1)
                qkm2 = tl.where(big, qkm2 * BIGINV, qkm2)
                qkm1 = tl.where(big, qkm1 * BIGINV, qkm1)
            i = i + UNROLL
            alive = tl.max((actc & (tq > MACHEP)).to(tl.int32)) > 0
        qv = tl.where(ax != 0.0, ansq * ax, 0.0)
        p_cf = tl.where(actc, 1.0 - qv, 1.0)
    p = tl.where(cf, p_cf, p)

    # ---- uniform asymptotic leading term for a >> 1, x ~ a -------------
    if tl.max(huge.to(tl.int32)) > 0:
        sigma = (x_s - a_s) / a_s
        v = tl.maximum(-2.0 * (_ld.log1p(sigma) - sigma), 0.0)
        sq = tl.sqrt(v)
        eta = tl.where(x_s > a_s, sq, -sq)
        c0 = -0.3333333333333333 + eta * (
            0.08333333333333333
            + eta * (-0.014814814814814815 + eta * 0.0011574074074074074)
        )
        ph = 0.5 * _ld.erfc(-eta * tl.sqrt(a_s * 0.5)) - tl.exp(
            -0.5 * a_s * eta * eta
        ) * c0 / tl.sqrt(_TWO_PI * a_s)
        p = tl.where(huge, ph, p)

    # ---- boundary values, in reference guard order ---------------------
    nanv = float("nan")
    res = p
    res = tl.where(x == _INF, 1.0, res)
    res = tl.where(a == _INF, tl.where(x == _INF, nanv, 0.0), res)
    res = tl.where(x == 0.0, 0.0, res)
    res = tl.where(a == 0.0, tl.where(x > 0.0, 1.0, nanv), res)
    res = tl.where((a < 0.0) | (x < 0.0), nanv, res)

    tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty), mask=mask)


def _collapse(a, b, shape):
    """Align both strides to ``shape`` and collapse mergeable dimensions.

    Returns the surviving extents in innermost-first order, which is the order
    the kernel uses for its flat-index decomposition.
    """
    nd = len(shape)
    sha = a.shape
    shb = b.shape
    sa = a.stride()
    sb = b.stride()
    ast = [0] * nd
    bst = [0] * nd
    oa = nd - len(sha)
    for i in range(len(sha)):
        if sha[i] != 1:
            ast[oa + i] = sa[i]
    ob = nd - len(shb)
    for i in range(len(shb)):
        if shb[i] != 1:
            bst[ob + i] = sb[i]

    cs = []
    ca = []
    cb = []
    for d in range(nd - 1, -1, -1):
        s = shape[d]
        if s == 1:
            continue
        if cs and ast[d] == ca[-1] * cs[-1] and bst[d] == cb[-1] * cs[-1]:
            # row-major merge: the combined extent keeps the inner stride
            cs[-1] = cs[-1] * s
            continue
        cs.append(s)
        ca.append(ast[d])
        cb.append(bst[d])
    return cs, ca, cb


def igamma(self, other):
    logger.debug("GEMS_HYGON IGAMMA")
    a = self
    b = other
    if not isinstance(a, torch.Tensor):
        raise TypeError("igamma expects a torch.Tensor as the first argument")
    if not isinstance(b, torch.Tensor):
        raise TypeError("igamma expects a torch.Tensor as the second argument")

    sha = a.shape
    shb = b.shape
    shape = sha if sha == shb else torch.broadcast_shapes(sha, shb)
    dtype = a.dtype if a.dtype == b.dtype else torch.result_type(a, b)
    if not dtype.is_floating_point:
        raise NotImplementedError(
            "\"igamma_cuda\" not implemented for '%s'"
            % str(dtype).replace("torch.", "")
        )

    out = torch.empty(shape, dtype=dtype, device=a.device)
    n = out.numel()
    if n == 0:
        return out

    if sha == shape and shb == shape and a.is_contiguous() and b.is_contiguous():
        dims = (n, 0, 0, 0, 0, 0, 0, 0)
        ast = (1, 0, 0, 0, 0, 0, 0, 0)
        bst = (1, 0, 0, 0, 0, 0, 0, 0)
        rank = 1
        i64 = n > 2147483647
    else:
        cs, ca, cb = _collapse(a, b, shape)
        if not cs:
            cs = [1]
            ca = [0]
            cb = [0]
        rank = len(cs)
        if rank > _MAXDIM:
            raise NotImplementedError("igamma: unsupported tensor rank")
        span = 0
        for d, s in zip(cs, ca):
            span += (d - 1) * abs(s)
        for d, s in zip(cs, cb):
            span += (d - 1) * abs(s)
        dims = tuple(cs) + (0,) * (_MAXDIM - rank)
        ast = tuple(ca) + (0,) * (_MAXDIM - rank)
        bst = tuple(cb) + (0,) * (_MAXDIM - rank)
        i64 = n > 2147483647 or span > 2147483647

    f64 = dtype == torch.float64
    maxlog, machep, big, biginv = _PARAMS["f64" if f64 else "f32"]

    _igamma_kernel[(triton.cdiv(n, _BLOCK),)](
        a,
        b,
        out,
        n,
        dims[0],
        dims[1],
        dims[2],
        dims[3],
        dims[4],
        dims[5],
        dims[6],
        dims[7],
        ast[0],
        ast[1],
        ast[2],
        ast[3],
        ast[4],
        ast[5],
        ast[6],
        ast[7],
        bst[0],
        bst[1],
        bst[2],
        bst[3],
        bst[4],
        bst[5],
        bst[6],
        bst[7],
        RANK=rank,
        BLOCK=_BLOCK,
        I64=i64,
        IS_F64=f64,
        MAXLOG=maxlog,
        MACHEP=machep,
        BIG=big,
        BIGINV=biginv,
        FIXED=_FIXED,
        UNROLL=_UNROLL,
        MAXIT=_MAXIT,
        num_warps=_NUM_WARPS,
    )
    return out
