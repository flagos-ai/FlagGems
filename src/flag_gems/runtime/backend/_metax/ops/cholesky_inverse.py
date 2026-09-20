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


def _tri_inv_col_kernel(
    L_ptr,
    X_ptr,
    n,
    L_s0,
    L_s1,
    L_s2,
    X_s0,
    X_s1,
    X_s2,
    UPPER: tl.constexpr,
    N_PAD: tl.constexpr,
    DT: tl.constexpr,
):
    # Column-parallel forward substitution (one CTA per matrix column).
    # Column k of X = M^{-1} solves M x = e_k:
    #   x[k] = 1/M[k,k];  x[i] = -(1/M[i,i]) * sum_{j=k}^{i-1} M[i,j] x[j]
    pid = tl.program_id(0)  # column k
    b = tl.program_id(1)  # batch
    k = pid
    rows = tl.arange(0, N_PAD)
    cols = tl.arange(0, N_PAD)
    lbase = L_ptr + b * L_s0
    xbase = X_ptr + b * X_s0

    d_k = tl.load(lbase + k * L_s1 + k * L_s2)
    x = tl.where(rows == k, 1.0 / d_k, tl.zeros([N_PAD], dtype=DT))
    # Steps i <= k are pure waste (x[i] = 0 for i < k and x[k] is initialized);
    # start the substitution chain at k+1.  The Cholesky factor is exactly
    # triangular, so entries with cols > i are already zero and need no mask.
    for i in range(k + 1, n):
        if UPPER:
            mrow = tl.load(lbase + cols * L_s1 + i * L_s2, mask=cols < n, other=0.0)
        else:
            mrow = tl.load(lbase + i * L_s1 + cols * L_s2, mask=cols < n, other=0.0)
        S = tl.sum(mrow * x, axis=0)
        d_i = tl.load(lbase + i * L_s1 + i * L_s2)
        xi = -S / d_i
        x = tl.where(rows == i, xi, x)
    tl.store(xbase + rows * X_s1 + k * X_s2, x, mask=rows < n)


@triton.jit
def _tri_inv_col2_kernel(
    L_ptr,
    X_ptr,
    n,
    L_s0,
    L_s1,
    L_s2,
    X_s0,
    X_s1,
    X_s2,
    UPPER: tl.constexpr,
    N_PAD: tl.constexpr,
    G: tl.constexpr,
    DT: tl.constexpr,
):
    # Grouped column-parallel substitution: one CTA per G columns, sharing
    # each matrix-row load across the group (cheaper per-column chain latency).
    pid = tl.program_id(0)
    b = tl.program_id(1)
    k0 = pid * G
    rows = tl.arange(0, N_PAD)
    cols = tl.arange(0, N_PAD)
    gs = tl.arange(0, G)
    lbase = L_ptr + b * L_s0
    xbase = X_ptr + b * X_s0

    xs = tl.zeros([G, N_PAD], dtype=DT)
    for g in tl.static_range(G):
        k = k0 + g
        d_k = tl.load(lbase + k * L_s1 + k * L_s2)
        xrow = tl.where(rows == k, 1.0 / d_k, 0.0)
        xs = tl.where(gs[:, None] == g, xrow[None, :], xs)
    for i in range(0, n):
        if UPPER:
            mrow = tl.load(lbase + cols * L_s1 + i * L_s2, mask=cols < n, other=0.0)
        else:
            mrow = tl.load(lbase + i * L_s1 + cols * L_s2, mask=cols < n, other=0.0)
        mrow = tl.where(cols <= i, mrow, 0.0)
        S = tl.sum(mrow[None, :] * xs, axis=1)
        d_i = tl.load(lbase + i * L_s1 + i * L_s2)
        upd = tl.where(rows[None, :] == i, -S[:, None] / d_i, 0.0)
        active_g = gs < (i - k0)
        xs = tl.where((rows[None, :] == i) & active_g[:, None], upd, xs)
    tl.store(
        xbase + (k0 + gs)[:, None] * X_s2 + rows[None, :] * X_s1,
        xs,
        mask=rows[None, :] < n,
    )


@triton.jit
def _blk_diag_kernel(
    L_ptr,
    X_ptr,
    n,
    L_s0,
    L_s1,
    L_s2,
    X_s0,
    X_s1,
    X_s2,
    UPPER: tl.constexpr,
    T: tl.constexpr,
    DT: tl.constexpr,
):
    # Diagonal block inverses X_ii = inv(M_ii) via T-step row sweep in [T,T]
    # registers; also zero the strictly-upper blocks of X (invalid entries).
    i = tl.program_id(0)
    b = tl.program_id(1)
    m = tl.num_programs(0)
    rows = tl.arange(0, T)
    cols = tl.arange(0, T)
    lbase = L_ptr + b * L_s0
    xbase = X_ptr + b * X_s0

    Xb = tl.zeros([T, T], dtype=DT)
    for r in range(0, T):
        if UPPER:
            mrow = tl.load(
                lbase + (i * T + cols) * L_s1 + (i * T + r) * L_s2,
                mask=cols < T,
                other=0.0,
            )
        else:
            mrow = tl.load(
                lbase + (i * T + r) * L_s1 + (i * T + cols) * L_s2,
                mask=cols < T,
                other=0.0,
            )
        d = tl.load(lbase + (i * T + r) * L_s1 + (i * T + r) * L_s2)
        S = tl.sum(mrow[:, None] * Xb, axis=0)
        xr = (tl.where(cols == r, 1.0, 0.0) - S) / d
        Xb = tl.where(rows[:, None] == r, xr[None, :], Xb)
    tl.store(
        xbase + (i * T + rows)[:, None] * X_s1 + (i * T + cols)[None, :] * X_s2, Xb
    )
    Z = tl.zeros([T, T], dtype=DT)
    for j in range(i + 1, m):
        tl.store(
            xbase + (i * T + rows)[:, None] * X_s1 + (j * T + cols)[None, :] * X_s2, Z
        )


@triton.jit
def _blk_col_kernel(
    L_ptr,
    X_ptr,
    n,
    L_s0,
    L_s1,
    L_s2,
    X_s0,
    X_s1,
    X_s2,
    UPPER: tl.constexpr,
    T: tl.constexpr,
    KC: tl.constexpr,
    DT: tl.constexpr,
):
    # Block-column forward substitution (one CTA per matrix column block).
    # Column k: X_ik = -D_i @ sum_{j=k}^{i-1} M_ij X_jk, i = k+1..m-1 sequential.
    k = tl.program_id(0)
    b = tl.program_id(1)
    m = tl.num_programs(0)
    rows = tl.arange(0, T)
    cols = tl.arange(0, T)
    kkc = tl.arange(0, KC)
    lbase = L_ptr + b * L_s0
    xbase = X_ptr + b * X_s0
    for i in range(k + 1, m):
        S = tl.zeros([T, T], dtype=DT)
        for j in range(k, i):
            for kc in range(0, T, KC):
                if UPPER:
                    mch = tl.load(
                        lbase
                        + (j * T + kc + kkc)[None, :] * L_s1
                        + (i * T + rows)[:, None] * L_s2,
                        mask=(kc + kkc)[None, :] < T,
                        other=0.0,
                    )
                else:
                    mch = tl.load(
                        lbase
                        + (i * T + rows)[:, None] * L_s1
                        + (j * T + kc + kkc)[None, :] * L_s2,
                        mask=(kc + kkc)[None, :] < T,
                        other=0.0,
                    )
                xch = tl.load(
                    xbase
                    + (j * T + kc + kkc)[:, None] * X_s1
                    + (k * T + cols)[None, :] * X_s2,
                    mask=(kc + kkc)[:, None] < T,
                    other=0.0,
                )
                S += tl.sum(mch[:, :, None] * xch[None, :, :], axis=1)
        D = tl.load(
            xbase + (i * T + rows)[:, None] * X_s1 + (i * T + cols)[None, :] * X_s2
        )
        Xik = -tl.sum(D[:, :, None] * S[None, :, :], axis=1)
        tl.store(
            xbase + (i * T + rows)[:, None] * X_s1 + (k * T + cols)[None, :] * X_s2, Xik
        )


@triton.jit
def _xtx_kernel(
    X_ptr,
    out_ptr,
    n,
    X_s0,
    X_s1,
    X_s2,
    O_s0,
    O_s1,
    O_s2,
    KC: tl.constexpr,
    BLOCK: tl.constexpr,
    DT: tl.constexpr,
):
    # out = X^T @ X via exact accumulation:
    # out[i, j] = sum_k X[k, i] * X[k, j]
    # X is lower triangular, so terms with k < max(i,j) are exactly zero;
    # start each tile's k-loop at kstart = max(i0, j0).
    pid = tl.program_id(0)
    b = tl.program_id(1)
    ncol = (n + BLOCK - 1) // BLOCK
    i0 = (pid // ncol) * BLOCK
    j0 = (pid % ncol) * BLOCK
    rows = i0 + tl.arange(0, BLOCK)
    cols = j0 + tl.arange(0, BLOCK)
    kk = tl.arange(0, KC)
    kstart = tl.maximum(i0, j0)
    xbase = X_ptr + b * X_s0
    acc = tl.zeros([BLOCK, BLOCK], dtype=DT)
    for k0 in range(kstart, n, KC):
        a = tl.load(
            xbase + (k0 + kk)[:, None] * X_s1 + rows[None, :] * X_s2,
            mask=((k0 + kk)[:, None] < n) & (rows[None, :] < n),
            other=0.0,
        )
        c = tl.load(
            xbase + (k0 + kk)[:, None] * X_s1 + cols[None, :] * X_s2,
            mask=((k0 + kk)[:, None] < n) & (cols[None, :] < n),
            other=0.0,
        )
        acc += tl.sum(a[:, :, None] * c[:, None, :], axis=0)
    tl.store(
        out_ptr + b * O_s0 + rows[:, None] * O_s1 + cols[None, :] * O_s2,
        acc,
        mask=(rows[:, None] < n) & (cols[None, :] < n),
    )


@triton.jit
def _fused_kernel(
    L_ptr,
    out_ptr,
    n,
    L_s0,
    L_s1,
    L_s2,
    O_s0,
    O_s1,
    O_s2,
    UPPER: tl.constexpr,
    N_PAD: tl.constexpr,
    DT: tl.constexpr,
):
    # Single-CTA fused tiny-n: X = M^{-1} row sweep in registers while
    # accumulating out = X^T @ X in registers (one launch, no X round-trip).
    pid = tl.program_id(0)
    rows = tl.arange(0, N_PAD)
    cols = tl.arange(0, N_PAD)
    lbase = L_ptr + pid * L_s0
    obase = out_ptr + pid * O_s0
    X = tl.zeros([N_PAD, N_PAD], dtype=DT)
    acc = tl.zeros([N_PAD, N_PAD], dtype=DT)
    for i in range(0, n):
        if UPPER:
            mrow = tl.load(lbase + cols * L_s1 + i * L_s2, mask=cols < n, other=0.0)
        else:
            mrow = tl.load(lbase + i * L_s1 + cols * L_s2, mask=cols < n, other=0.0)
        mrow = tl.where(cols <= i, mrow, 0.0)
        d = tl.load(lbase + i * L_s1 + i * L_s2)
        S = tl.sum(mrow[:, None] * X, axis=0)
        xrow = (tl.where(cols == i, 1.0, 0.0) - S) / d
        X = tl.where(rows[:, None] == i, xrow[None, :], X)
        acc += xrow[:, None] * xrow[None, :]
    tl.store(
        obase + rows[:, None] * O_s1 + cols[None, :] * O_s2,
        acc,
        mask=(rows[:, None] < n) & (cols[None, :] < n),
    )


def cholesky_inverse(L, upper=False):
    dev = L.device
    dt = L.dtype
    n = L.shape[-1]
    if L.numel() == 0:
        return torch.empty_like(L)
    if L.dim() < 2 or L.shape[-2] != n:
        raise ValueError("cholesky_inverse: input must be square and at least 2D")
    if dt == torch.float32:
        tl_dt = tl.float32
    elif dt == torch.float64:
        tl_dt = tl.float64
    else:
        raise TypeError(f"cholesky_inverse: unsupported dtype {dt}")

    batch = L.numel() // (n * n)
    if L.dim() == 2:
        ls0, ls1, ls2 = 0, L.stride(0), L.stride(1)
    else:
        ls0, ls1, ls2 = L.stride(-3), L.stride(-2), L.stride(-1)

    out = torch.empty_like(L)
    if out.dim() == 2:
        os0, os1, os2 = 0, out.stride(0), out.stride(1)
    else:
        os0, os1, os2 = out.stride(-3), out.stride(-2), out.stride(-1)

    if n <= 16:
        N_PAD = max(16, triton.next_power_of_2(n))
        _fused_kernel[(batch,)](
            L,
            out,
            n,
            ls0,
            ls1,
            ls2,
            os0,
            os1,
            os2,
            UPPER=bool(upper),
            N_PAD=N_PAD,
            DT=tl_dt,
            num_warps=1,
        )
        return out

    X = torch.empty((batch, n, n), dtype=dt, device=dev)
    xs = X.stride()

    if n <= 32 or dt == torch.float64 or n % 16 != 0:
        N_PAD = max(16, triton.next_power_of_2(n))
        _tri_inv_col_kernel[(n, batch)](
            L,
            X,
            n,
            ls0,
            ls1,
            ls2,
            xs[0],
            xs[1],
            xs[2],
            UPPER=bool(upper),
            N_PAD=N_PAD,
            DT=tl_dt,
            num_warps=1,
        )
    else:
        T = 16
        m = n // T
        _blk_diag_kernel[(m, batch)](
            L,
            X,
            n,
            ls0,
            ls1,
            ls2,
            xs[0],
            xs[1],
            xs[2],
            UPPER=bool(upper),
            T=T,
            DT=tl_dt,
            num_warps=4,
        )
        _blk_col_kernel[(m, batch)](
            L,
            X,
            n,
            ls0,
            ls1,
            ls2,
            xs[0],
            xs[1],
            xs[2],
            UPPER=bool(upper),
            T=T,
            KC=16,
            DT=tl_dt,
            num_warps=4,
        )

    BLOCK, KC = 16, 16
    grid = (triton.cdiv(n, BLOCK) * triton.cdiv(n, BLOCK), batch)
    _xtx_kernel[grid](
        X,
        out,
        n,
        xs[0],
        xs[1],
        xs[2],
        os0,
        os1,
        os2,
        KC=KC,
        BLOCK=BLOCK,
        DT=tl_dt,
        num_warps=4,
    )
    return out
