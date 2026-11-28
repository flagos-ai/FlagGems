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
# Kunlunxin (XPU) specialized linalg.eig: LAPACK-style (dlahqr) double-shift QR
# pipeline with all matrix state in global memory.
#
# XPU3 compiler constraints (all probed, see harness/perf_ir/probe_eig_caps_v*.py,
# probe_mem*.py, probe_grid*.py and probe_fw.py):
#   * 2-D tile arithmetic (broadcast rank-k updates, transposed-tile matvecs)
#     silently miscompiles; only 1-D loads + tl.sum on directly loaded
#     rows/columns are exact.
#   * a reduction may not sit under an `if` nested in a loop, nor under a
#     `while` nested in a `for` ("failed to legalize tt.reduce"): control flow
#     uses scalar `tl.where` and while-form loops only.
#   * vector stores in multi-program kernels must carry an explicit mask,
#     otherwise writes are dropped non-deterministically; masked scalar stores
#     inside while loops are dropped as well (redirected to a dummy slot here).
#   * in-place row update inside a nested loop corrupts the matrix and
#     loop-carried vectors miscompile at width 64: the matrix is
#     double-buffered and every reflector application is a
#     load -> reduce -> store to a different buffer.
#   * padded width 32 crashes the legalizer in the masked 2-D kernel form.
# Consequence: one kernel per phase (init / Hessenberg / QR sweeps /
# eigenvalue extraction / eigenvector back-substitution) with in-kernel loops,
# i.e. 5 launches per matrix instead of thousands of host-driven launches.
#
# The Householder similarity P H P is applied as two passes, H <- H P then
# H <- P H, so no full-length work vector has to live in registers across a
# loop: pass one needs only (row . v), pass two only (column . v).
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_EPS = tl.constexpr(3.8146973e-06)  # 32 * float32 eps, deflation threshold
_TINY = tl.constexpr(1.0e-30)
_MIN_BLOCK = 8
_MAXIT_COEF = 30  # LAPACK dhseqr ITMAX per active block size


@triton.jit
def _init_k(A, AB, PAR, N, BLOCK: tl.constexpr):
    """grid=(BLOCK,). Row i of the padded work matrix / identity accumulator.

    AB layout: [H0, H1, Q0, Q1], each BLOCK*BLOCK floats.
    PAR[0] = H parity, PAR[1] = Q parity.
    """
    i = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    row = tl.load(A + i * N + idx, mask=(i < N) & (idx < N), other=0.0)
    tl.store(AB + i * BLOCK + idx, row, mask=idx < BLOCK)
    qrow = tl.where(idx == i, 1.0, 0.0)
    qrow = tl.where(i < N, qrow, 0.0)
    tl.store(AB + 2 * BLOCK * BLOCK + i * BLOCK + idx, qrow, mask=idx < BLOCK)
    if i == 0:
        tl.store(PAR, 0)
        tl.store(PAR + 1, 0)


@triton.jit
def _hess_k(AB, PARH, PARQ, N, BLOCK: tl.constexpr):
    """grid=(1,). Householder reduction to Hessenberg form, accumulating Q."""
    idx = tl.arange(0, BLOCK)
    bs = BLOCK * BLOCK
    parh = tl.load(PARH)
    parq = tl.load(PARQ)
    for k in range(0, N - 2):
        ho = parh * bs
        hd = (1 - parh) * bs
        qo = 2 * bs + parq * bs
        qd = 2 * bs + (1 - parq) * bs
        # ---- reflector for column k, rows k+1 .. N-1 ----
        col = tl.load(AB + ho + idx * BLOCK + k)
        xv = tl.where(idx > k, col, 0.0)
        nrm = tl.sqrt(tl.sum(xv * xv))
        x1 = tl.sum(tl.where(idx == k + 1, xv, 0.0))
        sgn = tl.where(x1 >= 0.0, 1.0, -1.0)
        v = xv + tl.where(idx == k + 1, sgn * nrm, 0.0)
        v = v / (tl.sqrt(tl.sum(v * v)) + _TINY)
        # ---- first pass: H <- H P and Q <- Q P (row-wise) ----
        for i in range(0, N):
            rowi = tl.load(AB + ho + i * BLOCK + idx)
            hi = tl.sum(rowi * v)
            tl.store(AB + hd + i * BLOCK + idx, rowi - 2.0 * hi * v, mask=idx < BLOCK)
            qi = tl.load(AB + qo + i * BLOCK + idx)
            qv = tl.sum(qi * v)
            tl.store(AB + qd + i * BLOCK + idx, qi - 2.0 * qv * v, mask=idx < BLOCK)
        # ---- second pass: H <- P H (column-wise); result back into `ho` ----
        for j in range(0, N):
            colj = tl.load(AB + hd + idx * BLOCK + j)
            wj = tl.sum(colj * v)
            tl.store(AB + ho + idx * BLOCK + j, colj - 2.0 * wj * v, mask=idx < BLOCK)
        parq = 1 - parq
    tl.store(PARQ, parq)


@triton.jit
def _qr_k(AB, PARH, PARQ, N, MAXEV, BLOCK: tl.constexpr):
    """grid=(1,). Francis double-shift QR sweeps (dlahqr).

    Per sweep: locate negligible subdiagonals (read-only; the eigenvalue
    extraction uses the same criterion, so no explicit zeroing is needed),
    then chase the bulge from ``ll`` up to ``p - 2``.  Each reflector is
    applied as two passes (H <- H P then H <- P H) so no full-length work
    vector has to live in registers across a loop.  Data-dependent choices
    are scalar `tl.where` expressions (no `if`, which would sit between a loop
    and a reduction).
    """
    idx = tl.arange(0, BLOCK)
    bs = BLOCK * BLOCK
    parh = tl.load(PARH)
    parq = tl.load(PARQ)
    p = N
    ev = 0
    while (p > 2) and (ev < MAXEV):
        ev = ev + 1
        # ---- locate negligible subdiagonals (deflation candidates), find ll ----
        ll = 0
        for j in range(1, N):
            sub = tl.load(AB + parh * bs + j * BLOCK + j - 1, mask=j < p, other=0.0)
            d1 = tl.load(AB + parh * bs + j * BLOCK + j, mask=j < p, other=0.0)
            d2 = tl.load(
                AB + parh * bs + (j - 1) * BLOCK + j - 1, mask=j < p, other=0.0
            )
            small = tl.abs(sub) <= _EPS * (tl.abs(d1) + tl.abs(d2) + _TINY)
            ll = tl.where(small & (j < p), j, ll)
        deflate = ll >= p - 2
        steps = tl.where(deflate, 0, p - 1 - ll)
        m = p - 1
        # ---- bulge chasing: kk-th step acts on rows/cols (ll + kk)..+2 ----
        kk = 0
        while kk < steps:
            K = ll + kk
            nr3 = K < m - 1
            is_init = kk == 0
            ho = parh * bs
            hd = (1 - parh) * bs
            qo = 2 * bs + parq * bs
            qd = 2 * bs + (1 - parq) * bs
            # initial-shift vector from the bottom 2x2 (Francis double shift)
            h11 = tl.load(AB + ho + ll * BLOCK + ll)
            h12 = tl.load(AB + ho + ll * BLOCK + tl.minimum(ll + 1, N - 1))
            h21 = tl.load(AB + ho + tl.minimum(ll + 1, N - 1) * BLOCK + ll)
            h22 = tl.load(
                AB + ho + tl.minimum(ll + 1, N - 1) * BLOCK + tl.minimum(ll + 1, N - 1)
            )
            h32 = tl.load(
                AB + ho + tl.minimum(ll + 2, N - 1) * BLOCK + tl.minimum(ll + 1, N - 1)
            )
            aa = tl.load(AB + ho + (p - 2) * BLOCK + p - 2)
            bb = tl.load(AB + ho + (p - 2) * BLOCK + p - 1)
            cc = tl.load(AB + ho + (p - 1) * BLOCK + p - 2)
            dd = tl.load(AB + ho + (p - 1) * BLOCK + p - 1)
            ss = aa + dd
            tt = aa * dd - bb * cc
            disc = ss * ss - 4.0 * tt
            sq = tl.sqrt(tl.abs(disc))
            half = tl.where(disc >= 0.0, sq, 0.0)
            rt1r = (ss + half) * 0.5
            rt2r = (ss - half) * 0.5
            im2 = tl.where(disc >= 0.0, 0.0, -disc * 0.25)
            x_init = h21 * h12 + (h11 - rt1r) * (h11 - rt2r) + im2
            y_init = h21 * (h11 + h22 - ss)
            z_init = h21 * h32
            # settle step: the column below the bulge
            km1 = tl.maximum(K - 1, 0)
            x_set = tl.load(AB + ho + K * BLOCK + km1)
            y_set = tl.load(AB + ho + tl.minimum(K + 1, N - 1) * BLOCK + km1)
            z_set = tl.load(AB + ho + tl.minimum(K + 2, N - 1) * BLOCK + km1)
            x = tl.where(is_init, x_init, x_set)
            y = tl.where(is_init, y_init, y_set)
            z = tl.where(is_init, z_init, tl.where(nr3, z_set, 0.0))
            # ---- unit reflector from (x, y, z) at K..K+2 ----
            xv = tl.where(idx == K, x, 0.0) + tl.where(idx == K + 1, y, 0.0)
            xv = xv + tl.where(nr3 & (idx == K + 2), z, 0.0)
            nrm = tl.sqrt(tl.sum(xv * xv))
            x1 = tl.sum(tl.where(idx == K, xv, 0.0))
            sgn = tl.where(x1 >= 0.0, 1.0, -1.0)
            v = xv + tl.where(idx == K, sgn * nrm, 0.0)
            v = v / (tl.sqrt(tl.sum(v * v)) + _TINY)
                # ---- first pass: H <- H P, Q <- Q P ----
            for i in range(0, N):
                rowi = tl.load(AB + ho + i * BLOCK + idx)
                hi = tl.sum(rowi * v)
                tl.store(
                    AB + hd + i * BLOCK + idx, rowi - 2.0 * hi * v, mask=idx < BLOCK
                )
                qi = tl.load(AB + qo + i * BLOCK + idx)
                qv = tl.sum(qi * v)
                tl.store(
                    AB + qd + i * BLOCK + idx, qi - 2.0 * qv * v, mask=idx < BLOCK
                )
            # ---- second pass: H <- P H (result back into `ho`) ----
            for j in range(0, N):
                colj = tl.load(AB + hd + idx * BLOCK + j)
                wj = tl.sum(colj * v)
                tl.store(
                    AB + ho + idx * BLOCK + j, colj - 2.0 * wj * v, mask=idx < BLOCK
                )
            parq = 1 - parq
            kk = kk + 1
        p = tl.where(deflate, ll, p)
    tl.store(PARQ, parq)


@triton.jit
def _eigvals_k(AB, LR, LI, BLKB, PARH, N, BLOCK: tl.constexpr):
    """grid=(1,). Extract 1x1 / 2x2 block eigenvalues of the quasi-triangular H."""
    bs = BLOCK * BLOCK
    so = tl.load(PARH) * bs
    j = 0
    while j < N:
        dj = tl.load(AB + so + j * BLOCK + j)
        jn = tl.minimum(j + 1, N - 1)
        sub = tl.load(AB + so + jn * BLOCK + j)
        dj1 = tl.load(AB + so + jn * BLOCK + jn)
        b = tl.load(AB + so + j * BLOCK + jn)
        is2 = (j < N - 1) & (
            tl.abs(sub) > _EPS * (tl.abs(dj) + tl.abs(dj1) + _TINY)
        )
        tr = dj + dj1
        det = dj * dj1 - b * sub
        disc = tr * tr - 4.0 * det
        sq = tl.sqrt(tl.abs(disc))
        re = tr * 0.5
        pos = disc >= 0.0
        r0 = tl.where(pos, (tr + sq) * 0.5, re)
        r1 = tl.where(pos, (tr - sq) * 0.5, re)
        im0 = tl.where(pos, 0.0, sq * 0.5)
        tl.store(LR + j, tl.where(is2, r0, dj))
        tl.store(LI + j, tl.where(is2, im0, 0.0))
        tl.store(BLKB + j, tl.where(is2, 1, 0))
        # second entry of a 2x2 block; without a 2x2 block it goes to the
        # dummy slot BLOCK (masked scalar stores are dropped inside while
        # loops by the XPU legalizer)
        jd = tl.where(is2, jn, BLOCK)
        tl.store(LR + jd, tl.where(is2, r1, 0.0))
        tl.store(LI + jd, tl.where(is2, -im0, 0.0))
        j = tl.where(is2, j + 2, j + 1)


@triton.jit
def _eigvec_k(AB, LR, LI, BLKB, XR, XI, VR, VI, PARH, PARQ, N, BLOCK: tl.constexpr):
    """grid=(1,). Back-substitute each eigenvector, then map through Q.

    The 2x2 / 1x1 step choice is selected with `tl.where` on the freshly
    computed candidates instead of an `if` (reductions may not sit under an
    `if` inside a loop).  The `jj` loop is a while loop: a `for` containing a
    reduction-bearing `while` is rejected by the legalizer.
    """
    idx = tl.arange(0, BLOCK)
    bs = BLOCK * BLOCK
    ho = tl.load(PARH) * bs  # H slot
    qo = 2 * bs + tl.load(PARQ) * bs  # Q slot
    jj = 0
    while jj < N:
        lr = tl.load(LR + jj)
        li = tl.load(LI + jj)
        b2 = tl.maximum(jj - 1, 0)
        tbb1 = tl.load(AB + ho + b2 * BLOCK + tl.minimum(b2 + 1, N - 1))
        tbb = tl.load(AB + ho + b2 * BLOCK + b2)
        jn = tl.minimum(jj + 1, N - 1)
        tjj1 = tl.load(AB + ho + jj * BLOCK + jn)
        tjj = tl.load(AB + ho + jj * BLOCK + jj)
        is_prev = (jj > 0) & (tl.load(BLKB + b2) == 1)
        is_this = tl.load(BLKB + jj) == 1
        seed_prev = tl.where(idx == jj - 1, tbb1, 0.0) + tl.where(
            idx == jj, lr - tbb, 0.0
        )
        si_prev = tl.where(idx == jj, li, 0.0)
        seed_this = tl.where(idx == jj, tjj1, 0.0) + tl.where(
            idx == jj + 1, lr - tjj, 0.0
        )
        si_this = tl.where(idx == jj + 1, li, 0.0)
        seed_else = tl.where(idx == jj, 1.0, 0.0)
        xr = tl.where(is_prev, seed_prev, tl.where(is_this, seed_this, seed_else))
        xi = tl.where(is_prev, si_prev, tl.where(is_this, si_this, 0.0))
        tl.store(XR + jj * BLOCK + idx, xr, mask=idx < BLOCK)
        tl.store(XI + jj * BLOCK + idx, xi, mask=idx < BLOCK)
        ii = tl.where(is_prev, jj - 2, jj - 1)
        while ii >= 0:
            # running vector comes from global memory (no loop-carried vector)
            xrv = tl.load(XR + jj * BLOCK + idx)
            xiv = tl.load(XI + jj * BLOCK + idx)
            iim = tl.maximum(ii - 1, 0)
            # 1x1 candidate: solve (H[ii,ii] - L) x[ii] = -sum_{j>ii} H[ii,j] x[j]
            rowi = tl.load(AB + ho + ii * BLOCK + idx)
            r_r = -tl.sum(tl.where(idx > ii, rowi * xrv, 0.0))
            r_i = -tl.sum(tl.where(idx > ii, rowi * xiv, 0.0))
            dr = tl.load(AB + ho + ii * BLOCK + ii) - lr
            di = -li
            dd = dr * dr + di * di
            dd = tl.where(dd > _TINY, dd, _TINY)
            xvr = (r_r * dr + r_i * di) / dd
            xvi = (r_i * dr - r_r * di) / dd
            # 2x2 candidate
            rowim = tl.load(AB + ho + iim * BLOCK + idx)
            r0r = -tl.sum(tl.where(idx > ii, rowim * xrv, 0.0))
            r0i = -tl.sum(tl.where(idx > ii, rowim * xiv, 0.0))
            a00 = tl.load(AB + ho + iim * BLOCK + iim) - lr
            a01 = tl.load(AB + ho + iim * BLOCK + ii)
            a10 = tl.load(AB + ho + ii * BLOCK + iim)
            a11 = tl.load(AB + ho + ii * BLOCK + ii) - lr
            mli = -li  # imaginary part of (H[ii,ii] - L), L = lr + i*li
            detr = (a00 * a11 - mli * mli) - a01 * a10
            deti = a00 * mli + mli * a11
            n0r = r0r * a11 - r0i * mli - a01 * r_r
            n0i = r0r * mli + r0i * a11 - a01 * r_i
            n1r = a00 * r_r - r_i * mli - a10 * r0r
            n1i = a00 * r_i + r_r * mli - a10 * r0i
            d = detr * detr + deti * deti
            d = tl.where(d > _TINY, d, _TINY)
            x0r = (n0r * detr + n0i * deti) / d
            x0i = (n0i * detr - n0r * deti) / d
            x1r = (n1r * detr + n1i * deti) / d
            x1i = (n1i * detr - n1r * deti) / d
            doblk = (ii > 0) & (tl.load(BLKB + iim) == 1)
            nrv = tl.where(
                doblk,
                xrv + tl.where(idx == ii - 1, x0r, 0.0) + tl.where(idx == ii, x1r, 0.0),
                xrv + tl.where(idx == ii, xvr, 0.0),
            )
            niv = tl.where(
                doblk,
                xiv + tl.where(idx == ii - 1, x0i, 0.0) + tl.where(idx == ii, x1i, 0.0),
                xiv + tl.where(idx == ii, xvi, 0.0),
            )
            tl.store(XR + jj * BLOCK + idx, nrv, mask=idx < BLOCK)
            tl.store(XI + jj * BLOCK + idx, niv, mask=idx < BLOCK)
            ii = tl.where(doblk, ii - 2, ii - 1)
        # ---- V[:, jj] = Q @ x, then normalize ----
        xrv = tl.load(XR + jj * BLOCK + idx)
        xiv = tl.load(XI + jj * BLOCK + idx)
        for i in range(0, N):
            qrow = tl.load(AB + qo + i * BLOCK + idx)
            tl.store(VR + i * BLOCK + jj, tl.sum(qrow * xrv))
            tl.store(VI + i * BLOCK + jj, tl.sum(qrow * xiv))
        colr = tl.load(VR + idx * BLOCK + jj, mask=idx < N, other=0.0)
        coli = tl.load(VI + idx * BLOCK + jj, mask=idx < N, other=0.0)
        n2 = tl.sum(colr * colr + coli * coli)
        nrm = tl.sqrt(tl.where(n2 > _TINY, n2, _TINY))
        tl.store(VR + idx * BLOCK + jj, colr / nrm, mask=idx < N)
        tl.store(VI + idx * BLOCK + jj, coli / nrm, mask=idx < N)
        jj = jj + 1


def _pick_block(n):
    block = triton.next_power_of_2(max(n, _MIN_BLOCK))
    if block == 32:
        # XPU3 legalizer crash for the padded width 32 in masked 2-D kernels;
        # probe_eig_caps_v2.py shows 8/16/64/128 are safe.
        block = 64
    return block


def linalg_eig(A):
    logger.debug("GEMS_KUNLUNXIN LINALG_EIG")
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
    dev = A.device

    lr = torch.empty(batch, n, dtype=torch.float32, device=dev)
    li = torch.empty(batch, n, dtype=torch.float32, device=dev)
    vr = torch.empty(batch, n, n, dtype=torch.float32, device=dev)
    vi = torch.empty(batch, n, n, dtype=torch.float32, device=dev)

    block = _pick_block(n)
    bs = block * block
    # double buffer: [H0, H1, Q0, Q1]
    AB = torch.zeros(4 * bs, device=dev, dtype=torch.float32)
    PAR = torch.zeros(2, device=dev, dtype=torch.int32)
    LR = torch.zeros(block + 1, device=dev, dtype=torch.float32)
    LI = torch.zeros(block + 1, device=dev, dtype=torch.float32)
    BLKB = torch.zeros(block + 1, device=dev, dtype=torch.int32)
    VR = torch.zeros(block, block, device=dev, dtype=torch.float32)
    VI = torch.zeros(block, block, device=dev, dtype=torch.float32)
    XR = torch.zeros(block, block, device=dev, dtype=torch.float32)
    XI = torch.zeros(block, block, device=dev, dtype=torch.float32)

    maxev = _MAXIT_COEF * max(10, n) + 30

    with torch.no_grad():
        for b in range(batch):
            _init_k[(block,)](A_work[b], AB, PAR, n, BLOCK=block)
            if n > 2:
                _hess_k[(1,)](AB, PAR, PAR + 1, n, BLOCK=block)
            _qr_k[(1,)](AB, PAR, PAR + 1, n, maxev, BLOCK=block)
            _eigvals_k[(1,)](AB, LR, LI, BLKB, PAR, n, BLOCK=block)
            _eigvec_k[(1,)](
                AB, LR, LI, BLKB, XR, XI, VR, VI, PAR, PAR + 1, n, BLOCK=block
            )
            lr[b, :] = LR[:n]
            li[b, :] = LI[:n]
            vr[b, :n, :n] = VR[:n, :n]
            vi[b, :n, :n] = VI[:n, :n]

    eigvals = torch.complex(lr, li).reshape(*batch_shape, n)
    eigvecs = torch.complex(vr, vi).reshape(*batch_shape, n, n)
    return eigvals, eigvecs
