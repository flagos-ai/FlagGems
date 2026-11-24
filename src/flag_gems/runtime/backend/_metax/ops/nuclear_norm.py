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

"""Triton nuclear_norm: sum of singular values over the last two dims of a tensor.

Both fast paths are one-sided (Hestenes) Jacobi SVD, which keeps full relative
accuracy for every singular value (no squaring of the spectrum).  The rotation
engine is vectorised: instead of extracting one column pair at a time (which
cost five tile-wide reductions per rotation and dominated the whole runtime),
each round pairs *every* row with a partner by permuting the rows with one
`tl.dot` against a 0/1 involution matrix -- round r pairs i with (r-i) mod D, so
D rounds cover all D*(D-1)/2 pairs exactly once.

* small slices: one program per slice, the (<=16 x R) slab in registers.
* larger slices: blocked cyclic Jacobi over an fp32 scratch copy W whose rows are
  the min-dim columns; the host issues one launch per disjoint block-pair round,
  each program rotating its own pair of 8-row blocks (16 rows) to orthogonality.
"""

import torch
import triton
import triton.language as tl

_SWEEPS_SMALL = 8
_SWEEPS_SMALL_WIDE = 6
_BLK = 8
_B2 = 2 * _BLK
_SWEEPS_OUTER = 6
_SWEEPS_INNER = 1


def _next_pow2(v):
    p = 1
    while p < v:
        p <<= 1
    return p


@triton.jit
def _slice_offset(b, sh1, sh2, sh3, st0, st1, st2, st3, NDIM: tl.constexpr):
    """Flat element offset of slice b, decomposing the batch dims (row-major)."""
    rem = b.to(tl.int64)
    off = rem - rem
    if NDIM == 1:
        off += rem * st0
    elif NDIM == 2:
        off += (rem % sh1) * st1
        rem = rem // sh1
        off += rem * st0
    elif NDIM == 3:
        off += (rem % sh2) * st2
        rem = rem // sh2
        off += (rem % sh1) * st1
        rem = rem // sh1
        off += rem * st0
    elif NDIM == 4:
        off += (rem % sh3) * st3
        rem = rem // sh3
        off += (rem % sh2) * st2
        rem = rem // sh2
        off += (rem % sh1) * st1
        rem = rem // sh1
        off += rem * st0
    return off


@triton.jit
def _jacobi_eig(G, Dv, nrounds, NP: tl.constexpr, SWEEPS: tl.constexpr):
    """Orthogonalise the columns of the NP x NP Gram G, returning (G, V) with
    G_final = G_0 V.  Round r pairs column i with (r - i) mod Dv; partners are
    fetched with a single exact permutation matmul per round."""
    a = tl.arange(0, NP)
    V = tl.where(a[:, None] == a[None, :], 1.0, 0.0)
    real2 = (a[:, None] < Dv) & (a[None, :] < Dv)
    padf = tl.where((a[:, None] == a[None, :]) & (a[:, None] >= Dv), 1.0, 0.0)
    two = a[:, None] + a[None, :]
    self2 = (2 * a) % Dv
    for _ in range(SWEEPS):
        for rr in range(0, nrounds):
            Pm = tl.where((two % Dv == rr) & real2, 1.0, 0.0) + padf
            Yg = tl.dot(G, Pm, input_precision="ieee")
            Yv = tl.dot(V, Pm, input_precision="ieee")
            app = tl.sum(G * G, axis=0)
            aqq = tl.sum(Yg * Yg, axis=0)
            apq = tl.sum(G * Yg, axis=0)
            act = (apq != 0.0) & (self2 != rr)
            zeta = tl.where(act, (aqq - app) / (2.0 * apq), 0.0)
            t = tl.where(zeta >= 0.0, 1.0, -1.0) / (
                tl.abs(zeta) + tl.sqrt(1.0 + zeta * zeta)
            )
            t = tl.where(act, t, 0.0)
            cs = 1.0 / tl.sqrt(1.0 + t * t)
            sn = cs * t
            G = cs[None, :] * G - sn[None, :] * Yg
            V = cs[None, :] * V - sn[None, :] * Yv
    return G, V


@triton.jit
def _nuc_slice_kernel(
    A,
    Out,
    s_row,
    s_col,
    sh1,
    sh2,
    sh3,
    st0,
    st1,
    st2,
    st3,
    M: tl.constexpr,
    N: tl.constexpr,
    MP: tl.constexpr,
    NP: tl.constexpr,
    NDIM: tl.constexpr,
    SWEEPS: tl.constexpr,
):
    b = tl.program_id(0)
    off = _slice_offset(b, sh1, sh2, sh3, st0, st1, st2, st3, NDIM)
    r16 = tl.arange(0, NP)
    cols = tl.arange(0, MP)
    X = tl.load(
        A + off + r16[:, None] * s_col + cols[None, :] * s_row,
        mask=(r16[:, None] < N) & (cols[None, :] < M),
        other=0.0,
    )
    G = tl.dot(X, tl.trans(X), input_precision="ieee")
    G, V = _jacobi_eig(G, N, N, NP, SWEEPS)
    X = tl.dot(tl.trans(V), X, input_precision="ieee")
    nrm = tl.sum(X * X, axis=1)
    total = tl.sum(tl.where(r16 < N, tl.sqrt(tl.maximum(nrm, 0.0)), 0.0))
    tl.store(Out + b, total.to(Out.dtype.element_ty))


@triton.jit
def _nuc_slice_slow_kernel(
    A,
    Out,
    s_row,
    s_col,
    sh1,
    sh2,
    sh3,
    st0,
    st1,
    st2,
    st3,
    M: tl.constexpr,
    N: tl.constexpr,
    MP: tl.constexpr,
    NP: tl.constexpr,
    NDIM: tl.constexpr,
    SWEEPS: tl.constexpr,
    IS_F64: tl.constexpr,
):
    """Reduction-only fallback (float64, or slabs too large for the dot path)."""
    b = tl.program_id(0)
    off = _slice_offset(b, sh1, sh2, sh3, st0, st1, st2, st3, NDIM)
    if IS_F64:
        acc = tl.float64
    else:
        acc = tl.float32
    ridx = tl.arange(0, NP)
    cidx = tl.arange(0, MP)
    tile = tl.load(
        A + off + ridx[:, None] * s_col + cidx[None, :] * s_row,
        mask=(ridx[:, None] < N) & (cidx[None, :] < M),
        other=0.0,
    ).to(acc)
    for _ in range(SWEEPS):
        nrm = tl.sum(tile * tile, axis=1)
        for p in range(0, N):
            for q in range(p + 1, N):
                mp = ridx == p
                mq = ridx == q
                mpf = tl.where(mp, 1.0, 0.0).to(acc)
                mqf = tl.where(mq, 1.0, 0.0).to(acc)
                cp = tl.sum(tl.where(mp[:, None], tile, 0.0), axis=0)
                cq = tl.sum(tl.where(mq[:, None], tile, 0.0), axis=0)
                app = tl.sum(tl.where(mp, nrm, 0.0))
                aqq = tl.sum(tl.where(mq, nrm, 0.0))
                apq = tl.sum(cp * cq)
                zeta = tl.where(apq != 0.0, (aqq - app) / (2.0 * apq), 0.0)
                t = tl.where(zeta >= 0.0, 1.0, -1.0) / (
                    tl.abs(zeta) + tl.sqrt(1.0 + zeta * zeta)
                )
                t = tl.where(apq != 0.0, t, 0.0)
                cs = 1.0 / tl.sqrt(1.0 + t * t)
                sn = cs * t
                newp = cs * cp - sn * cq
                newq = sn * cp + cs * cq
                tile = (
                    tile
                    + (newp - cp)[None, :] * mpf[:, None]
                    + (newq - cq)[None, :] * mqf[:, None]
                )
    nrm = tl.sum(tile * tile, axis=1)
    total = tl.sum(tl.where(ridx < N, tl.sqrt(tl.maximum(nrm, 0.0)), 0.0))
    tl.store(Out + b, total.to(Out.dtype.element_ty))


@triton.jit
def _nuc_zero_kernel(Out):
    b = tl.program_id(0)
    tl.store(Out + b, 0.0)


@triton.jit
def _nuc_w_init(
    A,
    W,
    R,
    D,
    s_d,
    s_r,
    sh1,
    sh2,
    sh3,
    st0,
    st1,
    st2,
    st3,
    nrows,
    P: tl.constexpr,
    NDIM: tl.constexpr,
):
    s = tl.program_id(0)
    i = tl.program_id(1)
    off = _slice_offset(s, sh1, sh2, sh3, st0, st1, st2, st3, NDIM)
    cols = tl.arange(0, P)
    if i < D:
        v = tl.load(A + off + i * s_d + cols * s_r, mask=cols < R, other=0.0)
    else:
        v = tl.zeros((P,), tl.float32)
    tl.store(W + (s * nrows + i) * R + cols, v, mask=cols < R)


@triton.jit
def _nuc_pair_kernel(
    W,
    r,
    NB,
    R,
    nrounds,
    CH: tl.constexpr,
    BLK: tl.constexpr,
    SWIN: tl.constexpr,
):
    s = tl.program_id(0)
    pid = tl.program_id(1)
    nb1 = NB - 1
    ib = tl.where(pid == 0, NB - 1, (r + pid) % nb1)
    jb = tl.where(pid == 0, r, (r - pid + nb1) % nb1)

    rows = tl.arange(0, 16)
    cols = tl.arange(0, CH)
    grow = tl.where(rows < BLK, ib, jb) * BLK + (rows % BLK)
    base = s.to(tl.int64) * (NB * BLK * R)
    off = base + grow[:, None] * R

    # Gram of the 16-row slab, accumulated over column chunks so the ieee dot
    # operands stay inside the shared-memory budget.
    G = tl.zeros((16, 16), tl.float32)
    for c0 in range(0, R, CH):
        ci = c0 + cols
        X = tl.load(W + off + ci[None, :], mask=ci[None, :] < R, other=0.0)
        G += tl.dot(X, tl.trans(X), input_precision="ieee")

    G, V = _jacobi_eig(G, 16, nrounds, 16, SWIN)
    Vt = tl.trans(V)
    for c0 in range(0, R, CH):
        ci = c0 + cols
        m = ci[None, :] < R
        X = tl.load(W + off + ci[None, :], mask=m, other=0.0)
        tl.store(W + off + ci[None, :], tl.dot(Vt, X, input_precision="ieee"), mask=m)


@triton.jit
def _nuc_row_norms(W, NR, R, nrows, P: tl.constexpr):
    s = tl.program_id(0)
    i = tl.program_id(1)
    cols = tl.arange(0, P)
    v = tl.load(W + (s * nrows + i) * R + cols, mask=cols < R, other=0.0)
    tl.store(NR + s * nrows + i, tl.sum(v * v))


@triton.jit
def _nuc_finish(NR, Out, nrows, BLK: tl.constexpr):
    s = tl.program_id(0)
    acc = 0.0
    for c0 in range(0, nrows, BLK):
        idx = c0 + tl.arange(0, BLK)
        v = tl.load(NR + s * nrows + idx, mask=idx < nrows, other=0.0)
        acc += tl.sum(tl.sqrt(tl.maximum(v, 0.0)))
    tl.store(Out + s, acc.to(Out.dtype.element_ty))


# --------------------------------------------------------------------------- #
# host wrapper
# --------------------------------------------------------------------------- #
def nuclear_norm(self, keepdim=False):
    A = self
    if A.dim() < 2:
        raise RuntimeError(
            "nuclear_norm: Expected a tensor with 2 or more dimensions, got %d"
            % A.dim()
        )
    if A.dtype not in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        raise RuntimeError("nuclear_norm: unsupported dtype %s" % (A.dtype,))

    if keepdim:
        out_shape = tuple(A.shape[:-2]) + (1, 1)
    else:
        out_shape = tuple(A.shape[:-2])
    out = torch.empty(out_shape, dtype=A.dtype, device=A.device)

    S = 1
    for sz in A.shape[:-2]:
        S *= sz
    if S == 0:
        return out

    m = A.shape[-2]
    n = A.shape[-1]
    if m == 0 or n == 0:
        _nuc_zero_kernel[(S,)](out)
        return out

    if m >= n:
        D, R = n, m
        s_d = A.stride(-1)
        s_r = A.stride(-2)
    else:
        D, R = m, n
        s_d = A.stride(-2)
        s_r = A.stride(-1)

    bs = A.shape[:-2]
    bst = A.stride()[:-2]
    ndim = len(bs)
    if ndim > 4:
        raise RuntimeError("nuclear_norm: more than 4 batch dimensions unsupported")

    sh1 = bs[1] if ndim > 1 else 0
    sh2 = bs[2] if ndim > 2 else 0
    sh3 = bs[3] if ndim > 3 else 0
    st = list(bst) + [0, 0, 0, 0]
    is_f64 = A.dtype == torch.float64
    idx = (sh1, sh2, sh3, st[0], st[1], st[2], st[3])

    MP = _next_pow2(R)
    if not is_f64 and D <= 16 and MP <= 64:
        npp = 16 if D <= 16 else 32
        _nuc_slice_kernel[(S,)](
            A,
            out,
            s_r,
            s_d,
            *idx,
            M=R,
            N=D,
            MP=MP if MP >= 16 else 16,
            NP=npp,
            NDIM=ndim,
            SWEEPS=_SWEEPS_SMALL if npp == 16 else _SWEEPS_SMALL_WIDE,
            num_warps=1 if npp == 16 else 2,
            num_stages=1,
        )
        return out

    if not is_f64:
        nb = (D + _BLK - 1) // _BLK
        if nb < 2:
            nb = 2
        if nb % 2:
            nb += 1
        nrows = nb * _BLK
        W = torch.empty((S, nrows, R), dtype=torch.float32, device=A.device)
        NR = torch.empty((S, nrows), dtype=torch.float32, device=A.device)
        P = _next_pow2(R)
        nw = 2 if P <= 64 else 4
        _nuc_w_init[(S, nrows)](
            A,
            W,
            R,
            D,
            s_d,
            s_r,
            *idx,
            nrows,
            P=P,
            NDIM=ndim,
            num_warps=nw,
            num_stages=1,
        )
        nrounds = _B2
        nrounds_total = nb - 1
        npair = nb // 2
        for it in range(_SWEEPS_OUTER * nrounds_total):
            _nuc_pair_kernel[(S, npair)](
                W,
                it % nrounds_total,
                nb,
                R,
                nrounds,
                CH=128,
                BLK=_BLK,
                SWIN=_SWEEPS_INNER,
                num_warps=nw,
                num_stages=1,
            )
        _nuc_row_norms[(S, nrows)](W, NR, R, nrows, P=P, num_warps=nw, num_stages=1)
        _nuc_finish[(S,)](NR, out, nrows, BLK=128, num_warps=1, num_stages=1)
        return out

    NP = _next_pow2(D)
    if NP > 128 or MP > 128:
        raise RuntimeError(
            "nuclear_norm: slice %dx%d exceeds supported tile size" % (m, n)
        )
    _nuc_slice_slow_kernel[(S,)](
        A,
        out,
        s_r,
        s_d,
        *idx,
        M=R,
        N=D,
        MP=MP,
        NP=NP,
        NDIM=ndim,
        SWEEPS=_SWEEPS_SMALL,
        IS_F64=True,
        num_warps=1,
        num_stages=1,
    )
    return out
