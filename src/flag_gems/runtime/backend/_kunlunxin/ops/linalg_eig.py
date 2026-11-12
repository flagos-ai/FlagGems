# Copyright 2026, The FlagOS Contributors.
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
#
# Kunlunxin (XPU) specialized linalg.eig: authoritative LAPACK-style
# (dlahqr) double-shift QR pipeline, host-driven with row/column-parallel
# 1D Triton kernels (the only pattern the XPU triton fork reliably
# compiles).  Works in float64 internally; input float32 -> complex64 out.
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_TINY = tl.constexpr(1e-30)
_DFLAT = tl.constexpr(1e-8)
# cycle budget: it=30*max(10,n)+30 corresponds to LAPACK dhseqr ITMAX
_MAXIT_COEF = 30


@triton.jit
def _h_xyz_k(
    A,
    SC,
    P,
    LL,
    KK,
    MODE: tl.constexpr,
    NR: tl.constexpr,
    STRIDE,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """grid=batch. Compute bulge vector (x,y,z) at rows KK..KK+2.

    MODE=0: initial shift vector from the bottom 2x2 (Francis double shift,
    branchless real arithmetic, LAPACK dlahqr lines ~425-475).
    MODE=1: settle step, V = column KK-1 (rows KK..KK+2).
    NR=2: last (Givens) step, support is only KK,KK+1.
    """
    pid = tl.program_id(0)
    base = pid * BLOCK * BLOCK
    idx = tl.arange(0, BLOCK)
    mask = idx < N
    if MODE == 0:
        rll = tl.load(A + base + LL * STRIDE + idx, mask=mask, other=0.0)
        rl1 = tl.load(A + base + (LL + 1) * STRIDE + idx, mask=mask, other=0.0)
        rl2 = tl.load(A + base + (LL + 2) * STRIDE + idx, mask=mask, other=0.0)
        rmm = tl.load(A + base + (P - 1) * STRIDE + idx, mask=mask, other=0.0)
        rm1 = tl.load(A + base + (P - 2) * STRIDE + idx, mask=mask, other=0.0)
        h11 = tl.sum(tl.where(idx == LL, rll, 0.0))
        h12 = tl.sum(tl.where(idx == LL + 1, rll, 0.0))
        h21 = tl.sum(tl.where(idx == LL, rl1, 0.0))
        h22 = tl.sum(tl.where(idx == LL + 1, rl1, 0.0))
        h32 = tl.sum(tl.where(idx == LL + 1, rl2, 0.0))
        a = tl.sum(tl.where(idx == P - 2, rm1, 0.0))
        b = tl.sum(tl.where(idx == P - 1, rm1, 0.0))
        c = tl.sum(tl.where(idx == P - 2, rmm, 0.0))
        d = tl.sum(tl.where(idx == P - 1, rmm, 0.0))
        s = a + d
        t = a * d - b * c
        disc = s * s - 4.0 * t
        sq = tl.sqrt(tl.abs(disc))
        half = tl.where(disc >= 0.0, sq, 0.0)
        rt1r = (s + half) * 0.5
        rt2r = (s - half) * 0.5
        im2 = tl.where(disc >= 0.0, 0.0, -disc * 0.25)
        x = h21 * h12 + (h11 - rt1r) * (h11 - rt2r) + im2
        y = h21 * (h11 + h22 - s)
        z = h21 * h32
    else:
        rk = tl.load(A + base + KK * STRIDE + idx, mask=mask, other=0.0)
        rk1 = tl.load(A + base + (KK + 1) * STRIDE + idx, mask=mask, other=0.0)
        x = tl.sum(tl.where(idx == KK - 1, rk, 0.0))
        y = tl.sum(tl.where(idx == KK - 1, rk1, 0.0))
        if NR == 3:
            rk2 = tl.load(A + base + (KK + 2) * STRIDE + idx, mask=mask, other=0.0)
            z = tl.sum(tl.where(idx == KK - 1, rk2, 0.0))
        else:
            z = 0.0
    tl.store(SC + pid * 3 + 0, x)
    tl.store(SC + pid * 3 + 1, y)
    tl.store(SC + pid * 3 + 2, z)


@triton.jit
def _h_vn_k(SC, V, KK, NR: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    """grid=batch. Unit Householder reflector of (x,y,z) placed at KK..KK+2."""
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    mask = idx < N
    xa = tl.load(SC + pid * 3 + 0)
    ya = tl.load(SC + pid * 3 + 1)
    za = tl.load(SC + pid * 3 + 2)
    xz = (idx < 0).to(tl.float64)
    xab = xa + xz
    yab = ya + xz
    zab = za + xz
    if NR == 3:
        xv = (
            tl.where(idx == KK, xab, xz)
            + tl.where(idx == KK + 1, yab, xz)
            + tl.where(idx == KK + 2, zab, xz)
        )
    else:
        xv = tl.where(idx == KK, xab, xz) + tl.where(idx == KK + 1, yab, xz)
    nrm2 = tl.sum(xv * xv)
    nrm_b = tl.sqrt(nrm2) + xz
    k1b = tl.where(idx == KK, (idx >= 0).to(tl.float64), xz)
    xp = tl.sum(k1b * xv)
    xp_b = xp + xz
    sign_v = tl.where(
        xp_b >= 0.0, (idx >= 0).to(tl.float64), xz - (idx >= 0).to(tl.float64)
    )
    alpha_v = -sign_v * nrm_b
    v = xv - k1b * alpha_v
    vn2 = tl.sum(v * v)
    v = v / (tl.sqrt(vn2 + _TINY) + xz)
    tl.store(V + pid * BLOCK + idx, v, mask=mask)


@triton.jit
def _h_wu_k(A, V, W, U, STRIDE, N: tl.constexpr, BLOCK: tl.constexpr):
    """grid=batch*BLOCK. W[j]=H^T v[j], U[j]=(H v)[j] (0-d dots, column j)."""
    pid = tl.program_id(0)
    b = pid // BLOCK
    j = pid % BLOCK
    idx = tl.arange(0, BLOCK)
    mask = idx < N
    v = tl.load(V + b * BLOCK + idx, mask=mask, other=0.0)
    col = tl.load(A + b * BLOCK * BLOCK + idx * STRIDE + j, mask=mask, other=0.0)
    w = tl.sum(col * v)
    row = tl.load(A + b * BLOCK * BLOCK + j * STRIDE + idx, mask=mask, other=0.0)
    u = tl.sum(row * v)
    tl.store(W + b * BLOCK + j, w)
    tl.store(U + b * BLOCK + j, u)


@triton.jit
def _h_t_k(V, U, T, BLOCK: tl.constexpr):
    """grid=batch. t = v . (H v)."""
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    v = tl.load(V + pid * BLOCK + idx)
    u = tl.load(U + pid * BLOCK + idx)
    tl.store(T + pid, tl.sum(v * u))


@triton.jit
def _h_apply_k(A, Q, V, W, U, T, STRIDE, N: tl.constexpr, BLOCK: tl.constexpr):
    """grid=batch*BLOCK, row i per program: H <- P H P, Q <- Q P (P=I-2vv^T)."""
    pid = tl.program_id(0)
    b = pid // BLOCK
    i = pid % BLOCK
    idx = tl.arange(0, BLOCK)
    mask = idx < N
    v = tl.load(V + b * BLOCK + idx, mask=mask, other=0.0)
    w = tl.load(W + b * BLOCK + idx, mask=mask, other=0.0)
    u = tl.load(U + b * BLOCK + idx, mask=mask, other=0.0)
    t = tl.load(T + b)
    vi = tl.broadcast_to(
        tl.sum(tl.where(idx == i, v, 0.0), axis=0, keep_dims=True), (BLOCK,)
    )
    ui = tl.broadcast_to(
        tl.sum(tl.where(idx == i, u, 0.0), axis=0, keep_dims=True), (BLOCK,)
    )
    Hi = tl.load(A + b * BLOCK * BLOCK + i * STRIDE + idx, mask=mask, other=0.0)
    Qi = tl.load(Q + b * BLOCK * BLOCK + i * STRIDE + idx, mask=mask, other=0.0)
    qi = tl.broadcast_to(tl.sum(Qi * v, axis=0, keep_dims=True), (BLOCK,))
    ov = (idx >= 0).to(tl.float64)
    two = ov + ov
    four = two + two
    tb = tl.broadcast_to(t, (BLOCK,))
    Hi = Hi - two * (ui * v + vi * w) + four * (tb * vi) * v
    Qi = Qi - two * (qi * v)
    Hi = tl.where(tl.abs(Hi) < 1.0e-13, 0.0, Hi)
    Qi = tl.where(tl.abs(Qi) < 1.0e-13, 0.0, Qi)
    tl.store(A + b * BLOCK * BLOCK + i * STRIDE + idx, Hi, mask=mask)
    tl.store(Q + b * BLOCK * BLOCK + i * STRIDE + idx, Qi, mask=mask)


@triton.jit
def _h_vh_k(A, V, KK, STRIDE, N: tl.constexpr, BLOCK: tl.constexpr):
    """grid=batch. Hessenberg reflector for column KK (rows KK+1..N-1)."""
    pid = tl.program_id(0)
    base = pid * BLOCK * BLOCK
    idx = tl.arange(0, BLOCK)
    mask = idx < N
    xz = (idx < 0).to(tl.float64)
    col = tl.load(A + base + idx * STRIDE + KK, mask=mask, other=0.0)
    xv = tl.where((idx >= KK + 1) & mask, col, xz)
    nrm2 = tl.sum(xv * xv)
    nrm_b = tl.sqrt(nrm2) + xz
    k1b = tl.where(idx == KK + 1, (idx >= 0).to(tl.float64), xz)
    xp = tl.sum(k1b * xv)
    xp_b = xp + xz
    sign_v = tl.where(
        xp_b >= 0.0, (idx >= 0).to(tl.float64), xz - (idx >= 0).to(tl.float64)
    )
    alpha_v = -sign_v * nrm_b
    v = xv - k1b * alpha_v
    vn2 = tl.sum(v * v)
    v = v / (tl.sqrt(vn2 + _TINY) + xz)
    tl.store(V + pid * BLOCK + idx, v, mask=mask)


@triton.jit
def _h_qr_scan(A, STATE, STRIDE, N: tl.constexpr, BLOCK: tl.constexpr):
    """grid=batch. Zero negligible subdiagonals; flags[2+j]=1 when deflatable."""
    pid = tl.program_id(0)
    base = pid * BLOCK * BLOCK
    idx = tl.arange(0, BLOCK)
    rr = idx[:, None]
    cc = idx[None, :]
    H = tl.load(A + base + rr * STRIDE + cc, mask=(rr < N) & (cc < N), other=0.0).to(
        tl.float64
    )
    subv = tl.sum(tl.where(rr == cc + 1, H, 0.0), axis=1)
    diagv = tl.sum(tl.where(rr == cc, H, 0.0), axis=1)
    smallv = tl.abs(subv) <= _DFLAT * (tl.abs(diagv) + tl.abs(diagv) + _TINY)
    m2 = (rr == cc + 1) & tl.broadcast_to(smallv[:, None], (BLOCK, BLOCK))
    H = tl.where(m2, 0.0, H)
    tl.store(A + base + rr * STRIDE + cc, H, mask=(rr < N) & (cc < N))
    subv2 = tl.sum(tl.where(rr == cc + 1, H, 0.0), axis=1)
    diagv2 = tl.sum(tl.where(rr == cc, H, 0.0), axis=1)
    p = tl.load(STATE + pid * 2 + 0)
    zeroj = (
        (tl.abs(subv2) <= _DFLAT * (tl.abs(diagv2) + tl.abs(diagv2) + _TINY))
        & (idx >= 1)
        & (idx < p)
    )
    tl.store(STATE + pid * (2 + BLOCK) + 2 + idx, zeroj.to(tl.int32), mask=idx < N)


@triton.jit
def _h_eig_j_k(Hb, LR, LI, BLKB, J, STRIDE, N: tl.constexpr, BLOCK: tl.constexpr):
    """grid=batch. Extract 1x1 or 2x2 block eigenvalue(s) at column J."""
    pid = tl.program_id(0)
    base = pid * BLOCK * BLOCK
    idx = tl.arange(0, BLOCK)
    mask = idx < N
    rj = tl.load(Hb + base + J * STRIDE + idx, mask=mask, other=0.0)
    rj1 = tl.load(Hb + base + (J + 1) * STRIDE + idx, mask=mask, other=0.0)
    dj = tl.sum(tl.where(idx == J, rj, 0.0))
    dj1 = tl.sum(tl.where(idx == J + 1, rj1, 0.0))
    sub = tl.sum(tl.where(idx == J, rj1, 0.0))
    b = tl.sum(tl.where(idx == J + 1, rj, 0.0))
    is2 = (J < N - 1) & (tl.abs(sub) > _DFLAT * (tl.abs(dj) + tl.abs(dj1) + _TINY))
    if is2:
        tr = dj + dj1
        det = dj * dj1 - b * sub
        disc = tr * tr - 4.0 * det
        sq = tl.sqrt(tl.abs(disc))
        re = tr * 0.5
        pos = disc >= 0.0
        r0 = tl.where(pos, (tr + sq) * 0.5, re)
        r1 = tl.where(pos, (tr - sq) * 0.5, re)
        im0 = tl.where(pos, 0.0, sq * 0.5)
        tl.store(LR + pid * BLOCK + J, r0)
        tl.store(LR + pid * BLOCK + J + 1, r1)
        tl.store(LI + pid * BLOCK + J, im0)
        tl.store(LI + pid * BLOCK + J + 1, -im0)
        tl.store(BLKB + pid * BLOCK + J, 1)
    else:
        tl.store(LR + pid * BLOCK + J, dj)
        tl.store(LI + pid * BLOCK + J, 0.0)


@triton.jit
def _bs_init_k(
    A, LR, LI, BLKB, XR, XI, J, STRIDE, N: tl.constexpr, BLOCK: tl.constexpr
):
    """grid=batch. Seed eigenvector x for column J."""
    pid = tl.program_id(0)
    base = pid * BLOCK * BLOCK
    idx = tl.arange(0, BLOCK)
    mask = idx < N
    zr = (idx < 0).to(tl.float64)
    Lr = tl.load(LR + pid * BLOCK + J)
    Li = tl.load(LI + pid * BLOCK + J)
    b2 = tl.maximum(J - 1, 0)
    rb = tl.load(A + base + b2 * STRIDE + idx, mask=mask, other=0.0)
    rj = tl.load(A + base + J * STRIDE + idx, mask=mask, other=0.0)
    tbb1 = tl.sum(tl.where(idx == b2 + 1, rb, 0.0))
    tbb = tl.sum(tl.where(idx == b2, rb, 0.0))
    tjj1 = tl.sum(tl.where(idx == J + 1, rj, 0.0))
    tjj = tl.sum(tl.where(idx == J, rj, 0.0))
    seed_prev = tl.where(idx == J - 1, tbb1 + zr, zr) + tl.where(
        idx == J, Lr - tbb + zr, zr
    )
    si_prev = tl.where(idx == J, Li + zr, zr)
    seed_this = tl.where(idx == J, tjj1 + zr, zr) + tl.where(
        idx == J + 1, Lr - tjj + zr, zr
    )
    si_this = tl.where(idx == J + 1, Li + zr, zr)
    seed_else = tl.where(idx == J, 1.0 + zr, zr)
    si_else = zr
    is_prev = tl.where(J > 0, tl.load(BLKB + pid * BLOCK + (J - 1)), 0) == 1
    is_this = tl.load(BLKB + pid * BLOCK + J) == 1
    xr = tl.where(is_prev, seed_prev, tl.where(is_this, seed_this, seed_else))
    xi = tl.where(is_prev, si_prev, tl.where(is_this, si_this, si_else))
    tl.store(XR + pid * BLOCK + idx, xr, mask=mask)
    tl.store(XI + pid * BLOCK + idx, xi, mask=mask)


@triton.jit
def _bs_step_k(
    A,
    LR,
    LI,
    XR,
    XI,
    J,
    II,
    DOLBLK: tl.constexpr,
    STRIDE,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """grid=batch. Back-substitute row II (or the 2x2 (II-1,II)) of (H - L I) x = 0."""
    pid = tl.program_id(0)
    base = pid * BLOCK * BLOCK
    idx = tl.arange(0, BLOCK)
    mask = idx < N
    zr = (idx < 0).to(tl.float64)
    xr = tl.load(XR + pid * BLOCK + idx, mask=mask, other=0.0)
    xi = tl.load(XI + pid * BLOCK + idx, mask=mask, other=0.0)
    Lr = tl.load(LR + pid * BLOCK + J)
    Li = tl.load(LI + pid * BLOCK + J)
    colmask = idx > II
    rowi = tl.load(A + base + II * STRIDE + idx, mask=mask, other=0.0)
    r_r = -tl.sum(tl.where(colmask, rowi * xr, 0.0))
    r_i = -tl.sum(tl.where(colmask, rowi * xi, 0.0))
    if DOLBLK:
        rowim = tl.load(A + base + (II - 1) * STRIDE + idx, mask=mask, other=0.0)
        r0r = -tl.sum(tl.where(colmask, rowim * xr, 0.0))
        r0i = -tl.sum(tl.where(colmask, rowim * xi, 0.0))
        a00 = tl.sum(tl.where(idx == II - 1, rowim, 0.0)) - Lr
        a01 = tl.sum(tl.where(idx == II, rowim, 0.0))
        a10 = tl.sum(tl.where(idx == II - 1, rowi, 0.0))
        a11 = tl.sum(tl.where(idx == II, rowi, 0.0)) - Lr
        detr = (a00 * a11 - (-Li) * (-Li)) - a01 * a10
        deti = a00 * (-Li) + (-Li) * a11
        n0r = r0r * a11 - r0i * (-Li) - a01 * r_r
        n0i = r0r * (-Li) + r0i * a11 - a01 * r_i
        n1r = a00 * r_r - r_i * (-Li) - a10 * r0r
        n1i = a00 * r_i + r_r * (-Li) - a10 * r0i
        d = detr * detr + deti * deti
        d = tl.where(d > 1.0e-30, d, 1.0e-30)
        x0r = (n0r * detr + n0i * deti) / d
        x0i = (n0i * detr - n0r * deti) / d
        x1r = (n1r * detr + n1i * deti) / d
        x1i = (n1i * detr - n1r * deti) / d
        xr = (
            xr
            + tl.where(idx == II - 1, x0r + zr, zr)
            + tl.where(idx == II, x1r + zr, zr)
        )
        xi = (
            xi
            + tl.where(idx == II - 1, x0i + zr, zr)
            + tl.where(idx == II, x1i + zr, zr)
        )
    else:
        dr = tl.sum(tl.where(idx == II, rowi, 0.0)) - Lr
        di = -Li
        dd = dr * dr + di * di
        dd = tl.where(dd > 1.0e-30, dd, 1.0e-30)
        xvr = (r_r * dr + r_i * di) / dd
        xvi = (r_i * dr - r_r * di) / dd
        xr = xr + tl.where(idx == II, xvr + zr, zr)
        xi = xi + tl.where(idx == II, xvi + zr, zr)
    tl.store(XR + pid * BLOCK + idx, xr, mask=mask)
    tl.store(XI + pid * BLOCK + idx, xi, mask=mask)


@triton.jit
def _vq_k(Q, XR, XI, VR, VI, J, STRIDE, N: tl.constexpr, BLOCK: tl.constexpr):
    """grid=batch*BLOCK. VR[i,J],VI[i,J] = (Q @ x)[i] for column J."""
    pid = tl.program_id(0)
    b = pid // BLOCK
    i = pid % BLOCK
    idx = tl.arange(0, BLOCK)
    mask = idx < N
    base = b * BLOCK * BLOCK
    xr = tl.load(XR + b * BLOCK + idx, mask=mask, other=0.0)
    xi = tl.load(XI + b * BLOCK + idx, mask=mask, other=0.0)
    Qi = tl.load(Q + base + i * STRIDE + idx, mask=mask, other=0.0)
    vr = tl.sum(Qi * xr)
    vi = tl.sum(Qi * xi)
    tl.store(VR + base + i * STRIDE + J, vr)
    tl.store(VI + base + i * STRIDE + J, vi)


@triton.jit
def _norm_k(VR, VI, STRIDE, J, N: tl.constexpr, BLOCK: tl.constexpr):
    """grid=batch. Normalize column J of V."""
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    mask = idx < N
    base = pid * BLOCK * BLOCK
    colr = tl.load(VR + base + idx * STRIDE + J, mask=mask, other=0.0)
    coli = tl.load(VI + base + idx * STRIDE + J, mask=mask, other=0.0)
    n2 = tl.sum(colr * colr + coli * coli)
    nrm = tl.sqrt(tl.where(n2 > 1.0e-30, n2, 1.0e-30))
    colr = colr / (nrm + (idx < 0).to(tl.float64))
    coli = coli / (nrm + (idx < 0).to(tl.float64))
    tl.store(VR + base + idx * STRIDE + J, colr, mask=mask)
    tl.store(VI + base + idx * STRIDE + J, coli, mask=mask)


def _hess_host(Ap, Qp, n, block, V, W, U, T):
    for k in range(n - 2):
        _h_vh_k[(1,)](Ap, V, k, block, N=n, BLOCK=block)
        _h_wu_k[(block,)](Ap, V, W, U, block, N=n, BLOCK=block)
        _h_t_k[(1,)](V, U, T, BLOCK=block)
        _h_apply_k[(block,)](Ap, Qp, V, W, U, T, block, N=n, BLOCK=block)
        torch.cuda.synchronize()


def _qr_host(Ap, Qp, st, n, block, V, W, U, T, SC):
    p = n
    it = 0
    for it in range(_MAXIT_COEF * max(10, n) + 30):
        if p <= 2:
            break
        st[0] = p
        _h_qr_scan[(1,)](Ap, st, block, N=n, BLOCK=block)
        torch.cuda.synchronize()
        p = int(st[0].item())
        ll = 0
        for j in range(p - 1, 0, -1):
            if int(st[2 + j].item()) == 1:
                ll = j
                break
        if ll >= p - 2:
            p = ll
        else:
            m = p - 1
            for K in range(ll, m):
                nr = 3 if K < m - 1 else 2
                _h_xyz_k[(1,)](
                    Ap, SC, p, ll, K, 0 if K == ll else 1, nr, block, N=n, BLOCK=block
                )
                _h_vn_k[(1,)](SC, V, K, nr, N=n, BLOCK=block)
                _h_wu_k[(block,)](Ap, V, W, U, block, N=n, BLOCK=block)
                _h_t_k[(1,)](V, U, T, BLOCK=block)
                _h_apply_k[(block,)](Ap, Qp, V, W, U, T, block, N=n, BLOCK=block)
            torch.cuda.synchronize()
    return it


def _eigvals_host(Ap, LR, LI, BLKB, n, block):
    j = 0
    while j < n:
        _h_eig_j_k[(1,)](Ap, LR, LI, BLKB, j, block, N=n, BLOCK=block)
        torch.cuda.synchronize()
        if int(BLKB[j].item()) == 1:
            j += 2
        else:
            j += 1


def _eigvecs_host(Ap, Qp, LR, LI, BLKB, VR, VI, XR, XI, n, block):
    for J in range(n):
        _bs_init_k[(1,)](Ap, LR, LI, BLKB, XR, XI, J, block, N=n, BLOCK=block)
        torch.cuda.synchronize()
        top = J - 1
        if J > 0 and int(BLKB[J - 1].item()) == 1:
            top = J - 2
        ii = top
        while ii >= 0:
            doblk = ii > 0 and int(BLKB[ii - 1].item()) == 1
            _bs_step_k[(1,)](Ap, LR, LI, XR, XI, J, ii, doblk, block, N=n, BLOCK=block)
            torch.cuda.synchronize()
            ii -= 2 if doblk else 1
        _vq_k[(block,)](Qp, XR, XI, VR, VI, J, block, N=n, BLOCK=block)
        _norm_k[(1,)](VR, VI, block, J, N=n, BLOCK=block)
        torch.cuda.synchronize()


def linalg_eig(A):
    logger.debug("GEMS LINALG_EIG (kunlunxin vendor)")
    if A.dim() < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("linalg.eig expects batches of square matrices")
    if A.is_complex():
        raise NotImplementedError("FlagGems linalg.eig supports real inputs only")
    if A.dtype != torch.float32:
        raise NotImplementedError("FlagGems linalg.eig currently supports float32 only")

    n = A.shape[-1]
    batch_shape = A.shape[:-2]
    batch = 1
    for d in batch_shape:
        batch *= d

    A_work = A.reshape(batch, n, n).to(torch.float32).contiguous()

    lr = torch.empty(batch, n, dtype=torch.float32, device=A.device)
    li = torch.empty(batch, n, dtype=torch.float32, device=A.device)
    vr = torch.empty(batch, n, n, dtype=torch.float32, device=A.device)
    vi = torch.empty(batch, n, n, dtype=torch.float32, device=A.device)

    block = triton.next_power_of_2(n)
    if block < 4:
        block = 4  # XPU triton needs BLOCK >= 4 for the 1D kernels
    Ap = torch.zeros(block, block, device=A.device, dtype=torch.float64)
    Qp = torch.zeros(block, block, device=A.device, dtype=torch.float64)
    st = torch.zeros(2 + block, device=A.device, dtype=torch.int32)
    V = torch.zeros(block, device=A.device, dtype=torch.float64)
    W = torch.zeros(block, device=A.device, dtype=torch.float64)
    U = torch.zeros(block, device=A.device, dtype=torch.float64)
    T = torch.zeros(1, device=A.device, dtype=torch.float64)
    SC = torch.zeros(3, device=A.device, dtype=torch.float64)
    LR = torch.zeros(block, device=A.device, dtype=torch.float64)
    LI = torch.zeros(block, device=A.device, dtype=torch.float64)
    BLKB = torch.zeros(block, device=A.device, dtype=torch.int32)
    VR = torch.zeros(block, block, device=A.device, dtype=torch.float64)
    VI = torch.zeros(block, block, device=A.device, dtype=torch.float64)
    XR = torch.zeros(block, device=A.device, dtype=torch.float64)
    XI = torch.zeros(block, device=A.device, dtype=torch.float64)

    with torch.no_grad():
        for b in range(batch):
            Ap.zero_()
            Qp.zero_()
            LR.zero_()
            LI.zero_()
            BLKB.zero_()
            Ap[:n, :n] = A_work[b].to(torch.float64)
            Qp[:n, :n] = torch.eye(n, device=A.device, dtype=torch.float64)
            _hess_host(Ap, Qp, n, block, V, W, U, T)
            _qr_host(Ap, Qp, st, n, block, V, W, U, T, SC)
            _eigvals_host(Ap, LR, LI, BLKB, n, block)
            _eigvecs_host(Ap, Qp, LR, LI, BLKB, VR, VI, XR, XI, n, block)
            lr[b, :] = LR[:n].to(torch.float32)
            li[b, :] = LI[:n].to(torch.float32)
            vr[b, :n, :n] = VR[:n, :n].to(torch.float32)
            vi[b, :n, :n] = VI[:n, :n].to(torch.float32)

    eigvals = torch.complex(lr, li).reshape(*batch_shape, n)
    eigvecs = torch.complex(vr, vi).reshape(*batch_shape, n, n)
    return eigvals, eigvecs
