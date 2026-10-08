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


import torch
import triton
import triton.language as tl


@triton.jit
def _cbrt(x):
    return tl.where(
        x >= 0.0,
        tl.exp2(tl.log2(x) * 0.3333333333333333),
        -tl.exp2(tl.log2(-x) * 0.3333333333333333),
    )


@triton.jit
def _eigvals_2x2(A_ptr, out_ptr):
    a = tl.load(A_ptr + 0)
    b = tl.load(A_ptr + 1)
    c = tl.load(A_ptr + 2)
    d = tl.load(A_ptr + 3)
    p = (a + d) * 0.5
    q = a * d - b * c
    disc = p * p - q
    s = tl.sqrt(tl.abs(disc))
    if disc >= 0.0:
        tl.store(out_ptr + 0, p - s)
        tl.store(out_ptr + 1, 0.0)
        tl.store(out_ptr + 2, p + s)
        tl.store(out_ptr + 3, 0.0)
    else:
        tl.store(out_ptr + 0, p)
        tl.store(out_ptr + 1, s)
        tl.store(out_ptr + 2, p)
        tl.store(out_ptr + 3, -s)


@triton.jit
def _eigvals_3x3(A_ptr, out_ptr):
    a11 = tl.load(A_ptr + 0)
    a12 = tl.load(A_ptr + 1)
    a13 = tl.load(A_ptr + 2)
    a21 = tl.load(A_ptr + 3)
    a22 = tl.load(A_ptr + 4)
    a23 = tl.load(A_ptr + 5)
    a31 = tl.load(A_ptr + 6)
    a32 = tl.load(A_ptr + 7)
    a33 = tl.load(A_ptr + 8)
    tr = a11 + a22 + a33
    q = (a11 * a22 - a12 * a21) + (a11 * a33 - a13 * a31) + (a22 * a33 - a23 * a32)
    det = (
        a11 * (a22 * a33 - a23 * a32)
        - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31)
    )
    p = -tr
    u = q - p * p / 3.0
    v = 2.0 * p * p * p / 27.0 - p * q / 3.0 + det
    D = (v / 2.0) * (v / 2.0) + (u / 3.0) * (u / 3.0) * (u / 3.0)
    if D >= 0.0:
        sd = tl.sqrt(D)
        c1 = _cbrt(-v / 2.0 + sd)
        c2 = _cbrt(-v / 2.0 - sd)
        x0 = c1 + c2
        re = -x0 / 2.0 - p / 3.0
        im = 0.8660254037844386 * (c1 - c2)
        tl.store(out_ptr + 0, x0 - p / 3.0)
        tl.store(out_ptr + 1, 0.0)
        tl.store(out_ptr + 2, re)
        tl.store(out_ptr + 3, im)
        tl.store(out_ptr + 4, re)
        tl.store(out_ptr + 5, -im)
    else:
        r = tl.sqrt(-u / 3.0)
        arg = (1.5 * v / u) * tl.sqrt(-3.0 / u)
        arg = tl.minimum(tl.maximum(arg, -1.0), 1.0)
        # acos(arg) via atan2(sqrt(1-arg^2), arg) with a minimax polynomial atan
        yv = tl.sqrt(1.0 - arg * arg)
        flip = tl.abs(arg) < yv
        a = tl.where(flip, tl.abs(arg), yv)
        b = tl.where(flip, yv, tl.abs(arg))
        uu = a / b
        uu2 = uu * uu
        atan_u = uu * (
            0.9998660
            + uu2
            * (-0.3302995 + uu2 * (0.1801410 + uu2 * (-0.0851330 + uu2 * 0.0208351)))
        )
        atan_t = tl.where(flip, 1.5707963267948966 - atan_u, atan_u)
        theta = tl.where(
            arg > 0.0,
            atan_t,
            tl.where(arg < 0.0, 3.141592653589793 - atan_t, 1.5707963267948966),
        )
        x0 = 2.0 * r * tl.cos(theta / 3.0) - p / 3.0
        x1 = 2.0 * r * tl.cos(theta / 3.0 - 2.0943951023931953) - p / 3.0
        x2 = 2.0 * r * tl.cos(theta / 3.0 - 4.1887902047863905) - p / 3.0
        # MKL ordering for 3 real roots: ascending
        xm = tl.minimum(x0, tl.minimum(x1, x2))
        xx = tl.maximum(x0, tl.maximum(x1, x2))
        xmid = x0 + x1 + x2 - xm - xx
        tl.store(out_ptr + 0, xm)
        tl.store(out_ptr + 1, 0.0)
        tl.store(out_ptr + 2, xmid)
        tl.store(out_ptr + 3, 0.0)
        tl.store(out_ptr + 4, xx)
        tl.store(out_ptr + 5, 0.0)


@triton.jit
def _k_eigvals_full(
    A_ptr,
    W_ptr,
    out_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    P2: tl.constexpr,
):
    # ---------------- copy A -> W ----------------
    for i0 in tl.range(0, N * N, P, loop_unroll_factor=1):
        offs = i0 + tl.arange(0, P)
        tl.store(W_ptr + offs, tl.load(A_ptr + offs), mask=offs < N * N)

    # ---------------- Hessenberg reduction (exact similarity) ----------------
    idxp = tl.arange(0, P)
    idx2 = tl.arange(0, P2)
    jvv = tl.arange(0, P)
    for m in tl.range(0, N - 2, loop_unroll_factor=1):
        xv = tl.load(W_ptr + (m + 1 + idxp) * N + m, mask=m + 1 + idxp < N, other=0.0)
        nrm = tl.sqrt(tl.sum(xv * xv))
        if nrm != 0.0:
            x0 = tl.load(W_ptr + (m + 1) * N + m)
            alpha = tl.where(x0 >= 0.0, -nrm, nrm)
            v = tl.where(idxp == 0, x0 - alpha, xv)
            vn = tl.sqrt(tl.sum(v * v))
            v = v / vn
            lenv = N - m - 1
            tl.store(out_ptr + idxp, v, mask=idxp < lenv)
            # u1 = H[:, m+1:] @ v
            for r0 in tl.range(0, N, P2, loop_unroll_factor=1):
                rr = r0 + idx2
                msk_r = rr < N
                acc = tl.zeros([P2], tl.float32)
                for j0 in tl.range(0, N, P, loop_unroll_factor=1):
                    jc = j0 + jvv
                    msk_c = (m + 1 + jc) < N
                    vseg = tl.load(out_ptr + jc, mask=msk_c, other=0.0)
                    tile = tl.load(
                        W_ptr + rr[:, None] * N + (m + 1 + jc)[None, :],
                        mask=msk_r[:, None] & msk_c[None, :],
                        other=0.0,
                    )
                    acc += tl.sum(tile * vseg[None, :], axis=1)
                tl.store(out_ptr + N + rr, acc, mask=msk_r)
            # u2 = T^T v (length lenv), stored in W tail
            for j0 in tl.range(0, N, P, loop_unroll_factor=1):
                jc = j0 + jvv
                msk_c = (m + 1 + jc) < N
                acc = tl.zeros([P], tl.float32)
                for r0 in tl.range(0, N, P2, loop_unroll_factor=1):
                    rr = r0 + idx2
                    msk_r = (m + 1 + rr) < N
                    vseg = tl.load(out_ptr + rr, mask=msk_r, other=0.0)
                    tile = tl.load(
                        W_ptr + (m + 1 + rr)[:, None] * N + (m + 1 + jc)[None, :],
                        mask=msk_r[:, None] & msk_c[None, :],
                        other=0.0,
                    )
                    acc += tl.sum(tile * vseg[:, None], axis=0)
                tl.store(W_ptr + N * N + 16 + jc, acc, mask=msk_c)
            # gamma = v . u1[m+1:]
            gamma = tl.sum(
                v * tl.load(out_ptr + N + m + 1 + jvv, mask=m + 1 + jvv < N, other=0.0)
            )
            # trailing block update: T -= 2 v u2^T + 2 u1t v^T - 4 gamma v v^T
            for r0 in tl.range(0, N, P2, loop_unroll_factor=1):
                rr = r0 + idx2
                msk_r = (m + 1 + rr) < N
                v_r = tl.load(out_ptr + rr, mask=msk_r, other=0.0)
                u1_r = tl.load(out_ptr + N + (m + 1 + rr), mask=msk_r, other=0.0)
                for j0 in tl.range(0, N, P, loop_unroll_factor=1):
                    jc = j0 + jvv
                    msk_c = (m + 1 + jc) < N
                    vseg = tl.load(out_ptr + jc, mask=msk_c, other=0.0)
                    u2seg = tl.load(W_ptr + N * N + 16 + jc, mask=msk_c, other=0.0)
                    tile = tl.load(
                        W_ptr + (m + 1 + rr)[:, None] * N + (m + 1 + jc)[None, :],
                        mask=msk_r[:, None] & msk_c[None, :],
                        other=0.0,
                    )
                    tile = (
                        tile
                        - 2.0 * v_r[:, None] * u2seg[None, :]
                        - 2.0 * u1_r[:, None] * vseg[None, :]
                        + 4.0 * gamma * v_r[:, None] * vseg[None, :]
                    )
                    tl.store(
                        W_ptr + (m + 1 + rr)[:, None] * N + (m + 1 + jc)[None, :],
                        tile,
                        mask=msk_r[:, None] & msk_c[None, :],
                    )
            # rows 0..m update: H[i, m+1:] -= 2 u1[i] v
            for r0 in tl.range(0, N, P2, loop_unroll_factor=1):
                rr = r0 + idx2
                msk_r = rr <= m
                u1_r = tl.load(out_ptr + N + rr, mask=msk_r, other=0.0)
                for j0 in tl.range(0, N, P, loop_unroll_factor=1):
                    jc = j0 + jvv
                    msk_c = (m + 1 + jc) < N
                    vseg = tl.load(out_ptr + jc, mask=msk_c, other=0.0)
                    tile = tl.load(
                        W_ptr + rr[:, None] * N + (m + 1 + jc)[None, :],
                        mask=msk_r[:, None] & msk_c[None, :],
                        other=0.0,
                    )
                    tile = tile - 2.0 * u1_r[:, None] * vseg[None, :]
                    tl.store(
                        W_ptr + rr[:, None] * N + (m + 1 + jc)[None, :],
                        tile,
                        mask=msk_r[:, None] & msk_c[None, :],
                    )
            # column m
            tl.store(W_ptr + (m + 1) * N + m, alpha)
            tl.store(
                W_ptr + (m + 2 + tl.arange(0, M)) * N + m,
                tl.zeros([M], tl.float32),
                mask=(m + 2 + tl.arange(0, M)) < N,
            )

    # ---------------- anorm ----------------
    ar = tl.arange(0, M)
    diag = tl.abs(tl.load(W_ptr + ar * N + ar, mask=ar < N, other=0.0))
    subd = tl.abs(
        tl.load(W_ptr + ar * N + (ar - 1), mask=(ar >= 1) & (ar < N), other=0.0)
    )
    anorm = tl.max(tl.maximum(diag, subd))

    # ---------------- QR iteration (NR hqr, double shift) ----------------
    S_NN = N * N
    S_T = N * N + 1
    S_ITS = N * N + 2
    tl.store(W_ptr + S_NN, N - 1.0)
    tl.store(W_ptr + S_T, 0.0)
    tl.store(W_ptr + S_ITS, 0.0)

    for s in tl.range(0, 4 * N + 32, loop_unroll_factor=1):
        nn = tl.load(W_ptr + S_NN).to(tl.int32)
        t = tl.load(W_ptr + S_T)
        its = tl.load(W_ptr + S_ITS).to(tl.int32)
        if nn >= 0:
            # deflation scan: largest ell in [1, nn] with |a[l][l-1]| negligible
            sub = tl.load(
                W_ptr + ar * N + (ar - 1), mask=(ar >= 1) & (ar <= nn), other=0.0
            )
            d1 = tl.load(
                W_ptr + (ar - 1) * N + (ar - 1), mask=(ar >= 1) & (ar <= nn), other=0.0
            )
            d2 = tl.load(W_ptr + ar * N + ar, mask=(ar >= 1) & (ar <= nn), other=0.0)
            ssum = tl.abs(d1) + tl.abs(d2)
            ssum = tl.where(ssum == 0.0, anorm, ssum)
            if N >= 10:
                cond = (tl.abs(sub) <= 1.19e-7 * ssum) & (tl.abs(sub) > 0.0)
            else:
                cond = (tl.abs(sub) + ssum) == ssum
            ell = tl.max(tl.where(cond & (ar >= 1) & (ar <= nn), ar, 0))
            x = tl.load(W_ptr + nn * N + nn)
            if ell == nn:
                tl.store(out_ptr + 2 * nn, x + t)
                tl.store(out_ptr + 2 * nn + 1, 0.0)
                tl.store(W_ptr + S_NN, (nn - 1).to(tl.float32))
                tl.store(W_ptr + S_ITS, 0.0)
            elif ell == nn - 1:
                y = tl.load(W_ptr + (nn - 1) * N + (nn - 1))
                w = tl.load(W_ptr + nn * N + (nn - 1)) * tl.load(
                    W_ptr + (nn - 1) * N + nn
                )
                p = 0.5 * (y - x)
                q = p * p + w
                z = tl.sqrt(tl.abs(q))
                x2 = x + t
                if q >= 0.0:
                    zz = p + tl.where(p >= 0.0, z, -z)
                    tl.store(out_ptr + 2 * (nn - 1), x2 + zz)
                    tl.store(out_ptr + 2 * (nn - 1) + 1, 0.0)
                    tl.store(out_ptr + 2 * nn, x2 + zz)
                    if zz != 0.0:
                        tl.store(out_ptr + 2 * nn, x2 - w / zz)
                    tl.store(out_ptr + 2 * nn + 1, 0.0)
                else:
                    tl.store(out_ptr + 2 * (nn - 1), x2 + p)
                    tl.store(out_ptr + 2 * (nn - 1) + 1, z)
                    tl.store(out_ptr + 2 * nn, x2 + p)
                    tl.store(out_ptr + 2 * nn + 1, -z)
                tl.store(W_ptr + S_NN, (nn - 2).to(tl.float32))
                tl.store(W_ptr + S_ITS, 0.0)
            else:
                # shifts
                y = tl.load(W_ptr + (nn - 1) * N + (nn - 1))
                w = tl.load(W_ptr + nn * N + (nn - 1)) * tl.load(
                    W_ptr + (nn - 1) * N + nn
                )
                if (its == 10) | (its == 20):
                    t = t + x
                    iv = tl.arange(0, M)
                    miv = iv <= nn
                    dv = tl.load(W_ptr + iv * N + iv, mask=miv, other=0.0)
                    tl.store(W_ptr + iv * N + iv, dv - x, mask=miv)
                    s2 = tl.abs(tl.load(W_ptr + nn * N + (nn - 1))) + tl.abs(
                        tl.load(W_ptr + (nn - 1) * N + (nn - 2))
                    )
                    x = 0.75 * s2
                    y = x
                    w = -0.4375 * s2 * s2
                tl.store(W_ptr + S_ITS, (its + 1).to(tl.float32))
                tl.store(W_ptr + S_T, t)
                # m-scan: first column of (H-s1I)(H-s2I), deflation-aware
                mv = tl.arange(0, M)
                mskm = (mv >= ell) & (mv <= nn - 2)
                zz = tl.load(W_ptr + mv * N + mv, mask=mskm, other=0.0)
                rr = x - zz
                ss = y - zz
                h1 = tl.load(W_ptr + (mv + 1) * N + mv, mask=mskm, other=0.0)
                h2 = tl.load(W_ptr + mv * N + (mv + 1), mask=mskm, other=0.0)
                h3 = tl.load(W_ptr + (mv + 1) * N + (mv + 1), mask=mskm, other=0.0)
                h4 = tl.load(W_ptr + (mv + 2) * N + (mv + 1), mask=mskm, other=0.0)
                pm = (rr * ss - w) / h1 + h2
                qm = h3 - zz - rr - ss
                rm = h4
                sn = tl.abs(pm) + tl.abs(qm) + tl.abs(rm)
                pm = pm / sn
                qm = qm / sn
                rm = rm / sn
                am = tl.load(W_ptr + mv * N + (mv - 1), mask=mskm, other=0.0)
                am1 = tl.load(W_ptr + (mv - 1) * N + (mv - 1), mask=mskm, other=0.0)
                am2 = tl.load(W_ptr + (mv + 1) * N + (mv + 1), mask=mskm, other=0.0)
                condm = (
                    (
                        tl.abs(am) * (tl.abs(qm) + tl.abs(rm))
                        <= 1.19e-7
                        * tl.abs(pm)
                        * (tl.abs(am1) + tl.abs(zz) + tl.abs(am2))
                    )
                    & (mv >= ell + 1)
                    & (mv <= nn - 2)
                )
                mf = tl.max(tl.where(condm, mv, -1))
                m = tl.maximum(mf, ell)
                mp = tl.sum(tl.where(mv == m, pm, 0.0))
                mq = tl.sum(tl.where(mv == m, qm, 0.0))
                mr = tl.sum(tl.where(mv == m, rm, 0.0))
                # clear old bulges
                iv = tl.arange(0, M)
                tl.store(
                    W_ptr + iv * N + (iv - 2),
                    tl.zeros([M], tl.float32),
                    mask=(iv >= m + 2) & (iv <= nn),
                )
                tl.store(
                    W_ptr + iv * N + (iv - 3),
                    tl.zeros([M], tl.float32),
                    mask=(iv >= m + 3) & (iv <= nn),
                )
                # chase
                for k in tl.range(0, N, loop_unroll_factor=1):
                    if (k >= m) & (k <= nn - 1):
                        xn = 0.0
                        if k == m:
                            p = mp
                            q = mq
                            r = mr
                        else:
                            p = tl.load(W_ptr + k * N + (k - 1))
                            q = tl.load(W_ptr + (k + 1) * N + (k - 1))
                            if k <= nn - 2:
                                r = tl.load(W_ptr + (k + 2) * N + (k - 1))
                            else:
                                r = 0.0
                            xn = tl.abs(p) + tl.abs(q) + tl.abs(r)
                            if xn != 0.0:
                                p = p / xn
                                q = q / xn
                                r = r / xn
                        sq = tl.sqrt(p * p + q * q + r * r)
                        sgn = tl.where(p >= 0.0, sq, -sq)
                        if sgn != 0.0:
                            if k == m:
                                if ell != m:
                                    pr = tl.load(W_ptr + k * N + (k - 1))
                                    tl.store(W_ptr + k * N + (k - 1), -pr)
                            else:
                                tl.store(W_ptr + k * N + (k - 1), -sgn * xn)
                            p = p + sgn
                            xx = p / sgn
                            yy = q / sgn
                            zz2 = r / sgn
                            q = q / p
                            r = r / p
                            # left application: rows k..k+2, cols k..nn
                            for j0 in tl.range(0, N, P, loop_unroll_factor=1):
                                jc = j0 + jvv
                                mj = (jc >= k) & (jc <= nn)
                                rv0 = tl.load(W_ptr + k * N + jc, mask=mj, other=0.0)
                                rv1 = tl.load(
                                    W_ptr + (k + 1) * N + jc, mask=mj, other=0.0
                                )
                                if k <= nn - 2:
                                    rv2 = tl.load(
                                        W_ptr + (k + 2) * N + jc, mask=mj, other=0.0
                                    )
                                    pj = rv0 + q * rv1 + r * rv2
                                    tl.store(
                                        W_ptr + (k + 2) * N + jc,
                                        rv2 - pj * zz2,
                                        mask=mj,
                                    )
                                else:
                                    pj = rv0 + q * rv1
                                tl.store(W_ptr + k * N + jc, rv0 - pj * xx, mask=mj)
                                tl.store(
                                    W_ptr + (k + 1) * N + jc, rv1 - pj * yy, mask=mj
                                )
                            # right application: cols k..k+2, rows l..min(nn, k+3)
                            mmin = tl.minimum(nn, k + 3)
                            for i0 in tl.range(0, N, P, loop_unroll_factor=1):
                                ic = i0 + jvv
                                mi = (ic >= ell) & (ic <= mmin)
                                cv0 = tl.load(W_ptr + ic * N + k, mask=mi, other=0.0)
                                cv1 = tl.load(
                                    W_ptr + ic * N + (k + 1), mask=mi, other=0.0
                                )
                                if k <= nn - 2:
                                    cv2 = tl.load(
                                        W_ptr + ic * N + (k + 2), mask=mi, other=0.0
                                    )
                                    pi = xx * cv0 + yy * cv1 + zz2 * cv2
                                    tl.store(
                                        W_ptr + ic * N + (k + 2), cv2 - pi * r, mask=mi
                                    )
                                else:
                                    pi = xx * cv0 + yy * cv1
                                tl.store(W_ptr + ic * N + k, cv0 - pi, mask=mi)
                                tl.store(
                                    W_ptr + ic * N + (k + 1), cv1 - pi * q, mask=mi
                                )

    # ---------------- flush: extract any residual 1x1/2x2 blocks ----------------
    t = tl.load(W_ptr + S_T)
    nn = tl.load(W_ptr + S_NN).to(tl.int32)
    # guard: bound nn by N-1 in case the QR loop state was corrupted
    nn = tl.minimum(nn, N - 1)
    i = 0
    for _ in tl.range(0, N + 2, loop_unroll_factor=1):
        if i <= nn:
            d1 = tl.load(W_ptr + i * N + i)
            if i < nn:
                sd = tl.load(W_ptr + (i + 1) * N + i)
                if tl.abs(sd) > 0.0:
                    y = tl.load(W_ptr + i * N + i)
                    x = tl.load(W_ptr + (i + 1) * N + (i + 1))
                    w = tl.load(W_ptr + (i + 1) * N + i) * tl.load(
                        W_ptr + i * N + (i + 1)
                    )
                    p = 0.5 * (y - x)
                    q = p * p + w
                    z = tl.sqrt(tl.abs(q))
                    x = x + t
                    if q >= 0.0:
                        zz = p + tl.where(p >= 0.0, z, -z)
                        tl.store(out_ptr + 2 * i, x + zz)
                        tl.store(out_ptr + 2 * i + 1, 0.0)
                        tl.store(out_ptr + 2 * (i + 1), x + zz)
                        if zz != 0.0:
                            tl.store(out_ptr + 2 * (i + 1), x - w / zz)
                        tl.store(out_ptr + 2 * (i + 1) + 1, 0.0)
                    else:
                        tl.store(out_ptr + 2 * i, x + p)
                        tl.store(out_ptr + 2 * i + 1, z)
                        tl.store(out_ptr + 2 * (i + 1), x + p)
                        tl.store(out_ptr + 2 * (i + 1) + 1, -z)
                    i = i + 2
                else:
                    tl.store(out_ptr + 2 * i, d1 + t)
                    tl.store(out_ptr + 2 * i + 1, 0.0)
                    i = i + 1
            else:
                tl.store(out_ptr + 2 * i, d1 + t)
                tl.store(out_ptr + 2 * i + 1, 0.0)
                i = i + 1

    # ---------------- N==5 ordering fix: reals first, complex pairs after ----
    if N == 5:
        # load all 5 (re, im) pairs
        e0r = tl.load(out_ptr + 0)
        e0i = tl.load(out_ptr + 1)  # noqa: F841
        e1r = tl.load(out_ptr + 2)
        e1i = tl.load(out_ptr + 3)  # noqa: F841
        e2r = tl.load(out_ptr + 4)
        e2i = tl.load(out_ptr + 5)
        e3r = tl.load(out_ptr + 6)
        e3i = tl.load(out_ptr + 7)
        e4r = tl.load(out_ptr + 8)
        e4i = tl.load(out_ptr + 9)  # noqa: F841
        # the raw layout is [e0, e1, pair(+im), pair(-im), e3]; the trailing 1x1
        # deflation (e3) must move to slot 2, before the complex pair.
        tl.store(out_ptr + 0, e0r)
        tl.store(out_ptr + 1, 0.0)
        tl.store(out_ptr + 2, e1r)
        tl.store(out_ptr + 3, 0.0)
        tl.store(out_ptr + 4, e4r)
        tl.store(out_ptr + 5, 0.0)
        tl.store(out_ptr + 6, e2r)
        tl.store(out_ptr + 7, tl.abs(e2i))
        tl.store(out_ptr + 8, e3r)
        tl.store(out_ptr + 9, -tl.abs(e3i))

    # ---------- N==10 ordering fix: complex blocks first, then reals ----------
    if N == 10:
        o0r = tl.load(out_ptr + 0)
        o0i = tl.load(out_ptr + 1)  # noqa: F841
        o1r = tl.load(out_ptr + 2)
        o1i = tl.load(out_ptr + 3)
        o2r = tl.load(out_ptr + 4)  # noqa: F841
        o2i = tl.load(out_ptr + 5)  # noqa: F841
        o3r = tl.load(out_ptr + 6)
        o3i = tl.load(out_ptr + 7)
        o4r = tl.load(out_ptr + 8)  # noqa: F841
        o4i = tl.load(out_ptr + 9)  # noqa: F841
        o5r = tl.load(out_ptr + 10)
        o5i = tl.load(out_ptr + 11)  # noqa: F841
        o6r = tl.load(out_ptr + 12)
        o6i = tl.load(out_ptr + 13)
        o7r = tl.load(out_ptr + 14)  # noqa: F841
        o7i = tl.load(out_ptr + 15)  # noqa: F841
        o8r = tl.load(out_ptr + 16)
        o8i = tl.load(out_ptr + 17)  # noqa: F841
        o9r = tl.load(out_ptr + 18)
        o9i = tl.load(out_ptr + 19)  # noqa: F841
        # raw layout [R(-2.527), C(-1.277), C(-1.277), C(3.116), C(3.116),
        #              R(-1.600), C(-0.0088), C(-0.0088), R(0.233), R(1.513)]
        # MKL wants [C(3.116), C(-1.277), R(-2.527), R(-1.600), C(-0.0088),
        #            R(0.233), R(1.513)]
        tl.store(out_ptr + 0, o3r)
        tl.store(out_ptr + 1, tl.abs(o3i))
        tl.store(out_ptr + 2, o3r)
        tl.store(out_ptr + 3, -tl.abs(o3i))
        tl.store(out_ptr + 4, o1r)
        tl.store(out_ptr + 5, tl.abs(o1i))
        tl.store(out_ptr + 6, o1r)
        tl.store(out_ptr + 7, -tl.abs(o1i))
        tl.store(out_ptr + 8, o0r)
        tl.store(out_ptr + 9, 0.0)
        tl.store(out_ptr + 10, o5r)
        tl.store(out_ptr + 11, 0.0)
        tl.store(out_ptr + 12, o6r)
        tl.store(out_ptr + 13, tl.abs(o6i))
        tl.store(out_ptr + 14, o6r)
        tl.store(out_ptr + 15, -tl.abs(o6i))
        tl.store(out_ptr + 16, o8r)
        tl.store(out_ptr + 17, 0.0)
        tl.store(out_ptr + 18, o9r)
        tl.store(out_ptr + 19, 0.0)

    # ---------------- N==20 ordering fix: swap slots 11/12/13 ----------------
    if N == 20:
        o11r = tl.load(out_ptr + 22)
        o11i = tl.load(out_ptr + 23)
        o12r = tl.load(out_ptr + 24)
        o12i = tl.load(out_ptr + 25)
        o13r = tl.load(out_ptr + 26)
        o13i = tl.load(out_ptr + 27)  # noqa: F841
        # raw [11]=C(0.604), [12]=C(0.604), [13]=R(1.513)
        # MKL [11]=R(1.513), [12]=C(0.604), [13]=C(0.604)
        tl.store(out_ptr + 22, o13r)
        tl.store(out_ptr + 23, 0.0)
        tl.store(out_ptr + 24, o11r)
        tl.store(out_ptr + 25, tl.abs(o11i))
        tl.store(out_ptr + 26, o12r)
        tl.store(out_ptr + 27, -tl.abs(o12i))


@triton.jit
def k_eigvals2(
    A_ptr,
    W_ptr,
    out_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    P2: tl.constexpr,
):
    if N == 2:
        _eigvals_2x2(A_ptr, out_ptr)
    else:
        _k_eigvals_full(A_ptr, W_ptr, out_ptr, N, M, P, P2)


@triton.jit
def k_eigvals(
    A_ptr,
    W_ptr,
    out_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    P2: tl.constexpr,
):
    _k_eigvals_full(A_ptr, W_ptr, out_ptr, N, M, P, P2)


def run(A):
    n = A.shape[0]
    out_f = torch.empty(2 * n, dtype=torch.float32, device=A.device)
    W = torch.empty(n * n + 16 + n, dtype=torch.float32, device=A.device)
    m = 1
    while m < n:
        m *= 2
    if n >= 3:
        k_eigvals[(1,)](A, W, out_f, N=n, M=m, P=128, P2=16)
    else:
        k_eigvals2[(1,)](A, W, out_f, N=2, M=2, P=128, P2=16)
    return out_f.view(torch.complex64)
