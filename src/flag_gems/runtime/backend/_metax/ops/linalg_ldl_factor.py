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

"""linalg_ldl_factor in Triton (real dtypes).

Replicates the reference (torch.linalg.ldl_factor on the metax/MACA cuSOLVER
backend) exactly: Bunch-Kaufman-style pivoting with the metax decision rule

    s   = |a[k,k]|
    c   = max_{i>k} |a[i,k]|            at row imax   (colmax)
    r   = max_{j>k, j!=imax} |a[imax,j]|              (rowmax, diag excluded)
    d   = |a[imax,imax]|
    alpha = (1+sqrt(17))/8

    nop   if c==0 or s >= alpha*c or s*r >= alpha*c*c
    swap  elif d >= alpha*max(c, r)
    2x2   else

Elimination is the standard LDL^T update; LD stores L columns at write time,
the diagonal and 2x2 off-diagonals from the final working matrix, and pivots
are 1-based with -(imax+1) written at both positions of a 2x2 block.
Loop-carried WK dependencies are guarded with tl.debug_barrier().
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


_ALPHA = (1.0 + (17.0**0.5)) / 8.0


@triton.jit
def _ldl_kernel(
    A,
    LD,
    PIV,
    WK,
    N,
    ALPHA: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    W = tl.arange(0, BLOCK)
    base = pid * N * N
    pbase = pid * N

    # copy A -> WK (working matrix; input must not be mutated), chunked
    cstart = 0
    while cstart < N:
        Wc = tl.arange(0, CHUNK)
        offs = base + W[:, None] * N + (cstart + Wc[None, :])
        m0 = (W[:, None] < N) & ((cstart + Wc)[None, :] < N)
        a = tl.load(A + offs, mask=m0, other=0.0)
        tl.store(WK + offs, a, mask=m0)
        cstart = cstart + CHUNK
    tl.debug_barrier()

    k = 0
    while k < N:
        # ---- column k: c = max |a[i,k]| ----
        col_off = base + (k + 1 + W) * N + k
        col_mask = (k + 1 + W) < N
        col = tl.load(WK + col_off, mask=col_mask, other=0.0)
        acol = tl.abs(col)
        c = tl.max(acol, axis=0)
        s_val = tl.load(WK + base + k * N + k)
        s = tl.abs(s_val)
        # If the first nop condition already holds (c==0 or s >= alpha*c), the
        # pivot is a nop regardless of r or imax, so the row-imax load, its
        # reduction, the d load, and the argmin reduction are all skipped. nop/
        # swap/imax are assigned in both branches (constant tensors in the skip
        # branch: c >= 0.0 is true whenever nop1 holds with a finite matrix;
        # s < 0.0 is always false since s is an absolute value; imax is unused
        # on the nop path which writes piv1 = k+1).
        nop1 = (c == 0.0) | (s >= ALPHA * c)
        if nop1:
            nop = c >= 0.0
            swap = s < 0.0
            imax = k
        else:
            idx = tl.where(acol == c, W, BLOCK)
            imax = tl.min(idx, axis=0) + k + 1
            imax = tl.where(c == 0.0, k, imax)
            # ---- row imax: r = max |a[imax,j]|, j>k, j!=imax ----
            row_off = base + imax * N + (k + 1 + W)
            row_mask = ((k + 1 + W) < N) & ((k + 1 + W) != imax)
            row = tl.load(WK + row_off, mask=row_mask, other=0.0)
            r = tl.max(tl.abs(row), axis=0)
            d = tl.abs(tl.load(WK + base + imax * N + imax))
            nop = s * r >= ALPHA * c * c
            swap = (~nop) & (d >= ALPHA * tl.maximum(c, r))

        if nop | swap:
            # ---------- 1x1 pivot ----------
            if CHUNK == BLOCK and swap:
                # fused register swap (single barrier): all reads happen before any
                # write. The trailing tile load redirects its row-imax entries to
                # the old row k; the tile store excludes column imax which the
                # cimax store covers. Row k is dead after this step, so no row-k
                # store is needed - this also removes the load/store barrier the
                # row-k store forced (the remaining tile-store vs tile-load
                # ordering is the same pattern the nop path already uses).
                lseg = tl.load(
                    WK + base + (k + 1 + W) * N + imax, mask=col_mask, other=0.0
                )
                ckseg = tl.load(
                    WK + base + (k + 1 + W) * N + k, mask=col_mask, other=0.0
                )
                d0 = tl.load(WK + base + imax * N + imax)
                s_kim = tl.load(WK + base + k * N + imax)
                s_kk = tl.load(WK + base + k * N + k)
                rel = imax - k - 1
                # trailing tile; row rel (imax) comes from the old row k
                offs_load = tl.where(
                    (k + 1 + W[:, None]) == imax,
                    base + k * N + (k + 1 + W[None, :]),
                    base + (k + 1 + W[:, None]) * N + (k + 1 + W[None, :]),
                )
                offs_store = base + (k + 1 + W[:, None]) * N + (k + 1 + W[None, :])
                m2 = (
                    ((k + 1 + W[:, None]) < N)
                    & ((k + 1 + W[None, :]) < N)
                    & ((k + 1 + W[None, :]) != imax)
                )
                t = tl.load(WK + offs_load, mask=m2, other=0.0)
                # L column (rows k+1..N-1 of the swapped column k), divided by d0
                lower_col = tl.where(W == rel, s_kim, lseg) / d0
                l_imax = s_kim / d0
                tl.store(PIV + pbase + k, imax + 1)
                tl.store(LD + base + k * N + k, d0)
                if d0 != 0.0:
                    tl.store(LD + base + (k + 1 + W) * N + k, lower_col, mask=col_mask)
                    t = t - d0 * (lower_col[:, None] * lower_col[None, :])
                    tl.store(WK + offs_store, t, mask=m2)
                    # swapped column imax: old column k (row-imax entry = old
                    # WK[k,k]) minus the rank-1 correction
                    ck_corr = tl.where(W == rel, s_kk, ckseg)
                    cimax = ck_corr - d0 * (lower_col * l_imax)
                    tl.store(WK + base + (k + 1 + W) * N + imax, cimax, mask=col_mask)
                    tl.debug_barrier()
                else:
                    # d0 == 0: no elimination; store the swapped column imax = old
                    # column k (corrected)
                    ck_corr = tl.where(W == rel, s_kk, ckseg)
                    tl.store(WK + base + (k + 1 + W) * N + imax, ck_corr, mask=col_mask)
                    tl.debug_barrier()
            else:
                if swap:
                    # swap rows k <-> imax
                    rk = tl.load(WK + base + k * N + W, mask=W < N, other=0.0)
                    ri = tl.load(WK + base + imax * N + W, mask=W < N, other=0.0)
                    tl.store(WK + base + k * N + W, ri, mask=W < N)
                    tl.store(WK + base + imax * N + W, rk, mask=W < N)
                    tl.debug_barrier()
                    # swap columns k <-> imax (must see the row swap)
                    ck = tl.load(WK + base + W * N + k, mask=W < N, other=0.0)
                    ci = tl.load(WK + base + W * N + imax, mask=W < N, other=0.0)
                    tl.store(WK + base + W * N + k, ci, mask=W < N)
                    tl.store(WK + base + W * N + imax, ck, mask=W < N)
                    tl.debug_barrier()
                    piv1 = imax + 1
                    d0 = tl.load(WK + base + k * N + k)
                    # post-swap L column: new column k, rows k+1..N-1
                    lower_col = tl.load(
                        WK + base + (k + 1 + W) * N + k, mask=col_mask, other=0.0
                    )
                else:
                    piv1 = k + 1
                    d0 = s_val
                    # no swap: the decision-time column-k load IS the L column
                    lower_col = col
                tl.store(PIV + pbase + k, piv1)
                tl.store(LD + base + k * N + k, d0)
                if d0 != 0.0:
                    lower_col = lower_col / d0
                    tl.store(LD + base + (k + 1 + W) * N + k, lower_col, mask=col_mask)
                    if CHUNK == BLOCK:
                        # single-chunk trailing update: reuse the register L column
                        # for both axes (rows k+1+W and cols k+1+W are identical),
                        # avoiding a second strided column load per step.
                        offs2 = base + (k + 1 + W[:, None]) * N + (k + 1 + W[None, :])
                        m2 = ((k + 1 + W[:, None]) < N) & ((k + 1 + W[None, :]) < N)
                        t = tl.load(WK + offs2, mask=m2, other=0.0)
                        t = t - d0 * (lower_col[:, None] * lower_col[None, :])
                        tl.store(WK + offs2, t, mask=m2)
                        tl.debug_barrier()
                    else:
                        # chunked rank-1 update of the trailing (n-k-1) x (n-k-1) block
                        cstart = k + 1
                        while cstart < N:
                            Wc = tl.arange(0, CHUNK)
                            cmask = (cstart + Wc) < N
                            offs2 = (
                                base + (k + 1 + W[:, None]) * N + (cstart + Wc[None, :])
                            )
                            m2 = ((k + 1 + W[:, None]) < N) & cmask[None, :]
                            t = tl.load(WK + offs2, mask=m2, other=0.0)
                            lc = tl.load(
                                WK + base + (cstart + Wc) * N + k, mask=cmask, other=0.0
                            )
                            lc = lc / d0
                            t = t - d0 * (lower_col[:, None] * lc[None, :])
                            tl.store(WK + offs2, t, mask=m2)
                            tl.debug_barrier()
                            cstart = cstart + CHUNK
            k = k + 1
        else:
            # ---------- 2x2 pivot ----------
            p2 = -(imax + 1)
            tl.store(PIV + pbase + k, p2)
            tl.store(PIV + pbase + k + 1, p2)
            mrow = (k + 2 + W) < N
            if imax != (k + 1):
                # row (k+1) is dead after this 2x2 step, so the swap only needs to
                # materialize the live row imax = old row k+1; the old d12/d22
                # (row imax, cols k and k+1) are captured pre-swap as scalars.
                d12 = tl.load(WK + base + imax * N + k)
                d22 = tl.load(WK + base + imax * N + imax)
                rk = tl.load(WK + base + (k + 1) * N + W, mask=W < N, other=0.0)
                tl.store(WK + base + imax * N + W, rk, mask=W < N)
                tl.debug_barrier()
                # column swap restricted to rows k+2..N-1 (the row store above
                # already permuted row imax; rows <= k are dead). ci doubles as the
                # L column 2: new column k+1 at rows k+2..N-1 is the post-swap
                # column imax, and the (imax,imax) corner lands correctly because
                # ck at row imax holds old WK[k+1,k+1] after the row store.
                ck = tl.load(
                    WK + base + (k + 2 + W) * N + (k + 1), mask=mrow, other=0.0
                )
                ci = tl.load(WK + base + (k + 2 + W) * N + imax, mask=mrow, other=0.0)
                tl.store(WK + base + (k + 2 + W) * N + (k + 1), ci, mask=mrow)
                tl.store(WK + base + (k + 2 + W) * N + imax, ck, mask=mrow)
                tl.debug_barrier()
                c1 = ci
            else:
                c1 = tl.load(
                    WK + base + (k + 2 + W) * N + (k + 1), mask=mrow, other=0.0
                )
                d12 = tl.load(WK + base + (k + 1) * N + k)
                d22 = tl.load(WK + base + (k + 1) * N + (k + 1))
            d11 = tl.load(WK + base + k * N + k)
            det = d11 * d22 - d12 * d12
            tl.store(LD + base + k * N + k, d11)
            tl.store(LD + base + (k + 1) * N + (k + 1), d22)
            tl.store(LD + base + (k + 1) * N + k, d12)
            if (k + 2) < N:
                c0 = tl.load(WK + base + (k + 2 + W) * N + k, mask=mrow, other=0.0)
                l1 = (c0 * d22 - c1 * d12) / det
                l2 = (c1 * d11 - c0 * d12) / det
                tl.store(LD + base + (k + 2 + W) * N + k, l1, mask=mrow)
                tl.store(LD + base + (k + 2 + W) * N + (k + 1), l2, mask=mrow)
                u = l1 * d11 + l2 * d12
                v = l1 * d12 + l2 * d22
                if CHUNK == BLOCK:
                    # single-chunk trailing update: reuse the register L columns
                    # for both axes (rows k+2+W and cols k+2+W are identical).
                    offs2 = base + (k + 2 + W[:, None]) * N + (k + 2 + W[None, :])
                    m2 = ((k + 2 + W[:, None]) < N) & ((k + 2 + W[None, :]) < N)
                    t = tl.load(WK + offs2, mask=m2, other=0.0)
                    t = t - (u[:, None] * l1[None, :] + v[:, None] * l2[None, :])
                    tl.store(WK + offs2, t, mask=m2)
                    tl.debug_barrier()
                else:
                    # chunked rank-2 update of the trailing block
                    cstart = k + 2
                    while cstart < N:
                        Wc = tl.arange(0, CHUNK)
                        cmask = (cstart + Wc) < N
                        offs2 = base + (k + 2 + W[:, None]) * N + (cstart + Wc[None, :])
                        m2 = ((k + 2 + W[:, None]) < N) & cmask[None, :]
                        t = tl.load(WK + offs2, mask=m2, other=0.0)
                        c0c = tl.load(
                            WK + base + (cstart + Wc) * N + k, mask=cmask, other=0.0
                        )
                        c1c = tl.load(
                            WK + base + (cstart + Wc) * N + (k + 1),
                            mask=cmask,
                            other=0.0,
                        )
                        l1c = (c0c * d22 - c1c * d12) / det
                        l2c = (c1c * d11 - c0c * d12) / det
                        t = t - (u[:, None] * l1c[None, :] + v[:, None] * l2c[None, :])
                        tl.store(WK + offs2, t, mask=m2)
                        tl.debug_barrier()
                        cstart = cstart + CHUNK
            k = k + 2


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def ldl_factor(self, *, hermitian=False):
    hermitian = bool(hermitian)
    self = self.contiguous()
    n = self.shape[-1]
    batch = self.numel() // (n * n)
    LD = torch.zeros_like(self)
    PIV = torch.zeros((batch, n), dtype=torch.int32, device=self.device)
    WK = torch.empty_like(self)
    if n == 0:
        return LD, PIV.view(self.shape[:-1])
    block = min(256, _next_pow2(max(n, 1)))
    chunk = 32 if block > 128 else block
    if block <= 32:
        nw = 4
    elif block == 64:
        nw = 8
    else:
        nw = 16
    _ldl_kernel[(batch,)](
        self,
        LD,
        PIV,
        WK,
        n,
        _ALPHA,
        BLOCK=block,
        CHUNK=chunk,
        num_warps=nw,
    )
    piv = PIV.view(self.shape[:-1])
    return LD, piv
