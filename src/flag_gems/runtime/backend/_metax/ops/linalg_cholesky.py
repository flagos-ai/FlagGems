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


@triton.jit
def _cholesky_small_kernel(
    A_ptr,
    W_ptr,
    N,
    stride_ab,
    stride_am,
    stride_an,
    stride_wb,
    stride_wm,
    stride_wn,
    BLOCK: tl.constexpr,
):
    """Single-block Cholesky for N <= BLOCK: load A, factor in registers, store L.

    One program per matrix.  The whole matrix fits in one BLOCK x BLOCK tile, so no
    panel solve / trailing update is needed and the strict upper triangle is stored
    as zeros.
    """
    pid = tl.program_id(0)
    a0 = A_ptr + pid * stride_ab
    w0 = W_ptr + pid * stride_wb

    ar = tl.arange(0, BLOCK)
    rows = ar[:, None]
    cols = ar[None, :]
    mask = (rows < N) & (cols < N)

    D = tl.load(
        a0 + rows * stride_am + cols * stride_an,
        mask=mask & (rows >= cols),
        other=0.0,
    )
    R = D
    L = tl.zeros((BLOCK, BLOCK), dtype=D.dtype)
    for i in range(BLOCK):
        col = tl.sum(tl.where(cols == i, R, 0.0), axis=1)  # residual column i
        diag = tl.sum(tl.where(ar == i, col, 0.0))  # R[i, i]
        l_ii = tl.sqrt(tl.maximum(diag, 0.0))
        col_new = tl.where(ar == i, l_ii, tl.where(ar > i, col / l_ii, 0.0))
        L = tl.where(cols == i, col_new[:, None], L)
        R = tl.where(
            (rows > i) & (cols > i), R - col_new[:, None] * col_new[None, :], R
        )

    tl.store(
        w0 + rows * stride_wm + cols * stride_wn,
        tl.where(rows >= cols, L, 0.0),
        mask=mask,
    )


@triton.jit
def _init_lower_kernel(
    A_ptr,
    W_ptr,
    N,
    stride_ab,
    stride_am,
    stride_an,
    stride_wb,
    stride_wm,
    stride_wn,
    BLOCK: tl.constexpr,
):
    """Copy the lower triangle of A into W (workspace/output), zero the upper."""
    pid_b = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK + tl.arange(0, BLOCK)
    cols = pid_n * BLOCK + tl.arange(0, BLOCK)
    rows = rows[:, None]
    cols = cols[None, :]
    mask = (rows < N) & (cols < N)
    a = A_ptr + pid_b * stride_ab + rows * stride_am + cols * stride_an
    v = tl.load(a, mask=mask, other=0.0)
    v = tl.where(rows >= cols, v, 0.0)
    w = W_ptr + pid_b * stride_wb + rows * stride_wm + cols * stride_wn
    tl.store(w, v, mask=mask)


@triton.jit
def _cholesky_lower_kernel(
    W_ptr,
    N,
    stride_wb,
    stride_wm,
    stride_wn,
    BLOCK: tl.constexpr,
    RB: tl.constexpr,
):
    """In-place blocked right-looking Cholesky of the lower triangle of W.

    Each program handles one matrix.  BLOCK is the diagonal block size,
    RB the trailing-update tile size.
    """
    pid = tl.program_id(0)
    w0 = W_ptr + pid * stride_wb

    ar = tl.arange(0, BLOCK)
    rows = ar[:, None]
    cols = ar[None, :]
    lt = rows >= cols

    for k in range(0, N, BLOCK):
        kk = k + ar
        kmask = kk < N

        # ---- diagonal block: in-register column-Cholesky with rank-1 residuals ----
        D = tl.load(
            w0 + kk[:, None] * stride_wm + kk[None, :] * stride_wn,
            mask=kmask[:, None] & kmask[None, :] & lt,
            other=0.0,
        )
        R = D
        Ld = tl.zeros((BLOCK, BLOCK), dtype=D.dtype)
        for i in range(BLOCK):
            col = tl.sum(tl.where(cols == i, R, 0.0), axis=1)  # (B,) residual column i
            diag = tl.sum(tl.where(ar == i, col, 0.0))  # R[i, i]
            l_ii = tl.sqrt(tl.maximum(diag, 0.0))
            col_new = tl.where(ar == i, l_ii, tl.where(ar > i, col / l_ii, 0.0))
            Ld = tl.where(cols == i, col_new[:, None], Ld)
            R = tl.where(
                (rows > i) & (cols > i), R - col_new[:, None] * col_new[None, :], R
            )

        # store diagonal block factor (lower triangle)
        tl.store(
            w0 + kk[:, None] * stride_wm + kk[None, :] * stride_wn,
            Ld,
            mask=kmask[:, None] & kmask[None, :] & lt,
        )

        # ---- panel solve + trailing update ----
        m = N - (k + BLOCK)
        if m > 0:
            ld_diag = tl.sum(tl.where(rows == cols, Ld, 0.0), axis=1)  # (B,)
            # pass 1: solve panel blocks (forward substitution) and store
            for r in range(0, m, RB):
                rr = r + tl.arange(0, RB)
                rmask = rr < m
                rr_g = k + BLOCK + rr
                P = tl.load(
                    w0 + rr_g[:, None] * stride_wm + kk[None, :] * stride_wn,
                    mask=rmask[:, None],
                    other=0.0,
                )
                Xb = P
                for j in range(BLOCK):
                    col_j = tl.sum(
                        tl.where(cols == j, Ld, 0.0), axis=1
                    )  # column j of Ld
                    l_jj = tl.sum(tl.where(ar == j, ld_diag, 0.0))
                    xj = tl.sum(tl.where(cols == j, Xb, 0.0), axis=1)  # (RB,)
                    xj = xj / l_jj
                    Xb = tl.where(cols == j, xj[:, None], Xb)
                    Xb = tl.where(cols > j, Xb - xj[:, None] * col_j[None, :], Xb)
                tl.store(
                    w0 + rr_g[:, None] * stride_wm + kk[None, :] * stride_wn,
                    Xb,
                    mask=rmask[:, None],
                )
            # pass 2: trailing Schur-complement update  T -= Xr @ Xc^T
            for r in range(0, m, RB):
                rr = r + tl.arange(0, RB)
                rmask = rr < m
                rr_g = k + BLOCK + rr
                Xr = tl.load(
                    w0 + rr_g[:, None] * stride_wm + kk[None, :] * stride_wn,
                    mask=rmask[:, None],
                    other=0.0,
                )
                for c in range(0, m, RB):
                    cc = c + tl.arange(0, RB)
                    cmask = cc < m
                    cc_g = k + BLOCK + cc
                    Xc = tl.load(
                        w0 + cc_g[:, None] * stride_wm + kk[None, :] * stride_wn,
                        mask=cmask[:, None],
                        other=0.0,
                    )
                    T = tl.load(
                        w0 + rr_g[:, None] * stride_wm + cc_g[None, :] * stride_wn,
                        mask=rmask[:, None] & cmask[None, :],
                        other=0.0,
                    )
                    upd = tl.dot(Xr, tl.trans(Xc), input_precision="ieee")
                    T = T - upd
                    sm = (
                        rmask[:, None]
                        & cmask[None, :]
                        & (rr_g[:, None] >= cc_g[None, :])
                    )
                    tl.store(
                        w0 + rr_g[:, None] * stride_wm + cc_g[None, :] * stride_wn,
                        T,
                        mask=sm,
                    )


@triton.jit
def _transpose_kernel(
    L_ptr,
    U_ptr,
    N,
    stride_lb,
    stride_lm,
    stride_ln,
    stride_ub,
    stride_um,
    stride_un,
    BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(1)
    pid_m = tl.program_id(0)
    offs = pid_m * BLOCK + tl.arange(0, BLOCK)
    rows = offs[:, None]
    cols = offs[None, :]
    mask = (rows < N) & (cols < N)
    v = tl.load(
        L_ptr + pid_b * stride_lb + rows * stride_lm + cols * stride_ln,
        mask=mask,
        other=0.0,
    )
    tl.store(
        U_ptr + pid_b * stride_ub + cols * stride_um + rows * stride_un,
        v,
        mask=mask,
    )


def linalg_cholesky(A, upper=False):
    """Cholesky decomposition matching torch.linalg.cholesky(A, upper=upper)."""
    if isinstance(upper, torch.Tensor):
        up = bool(upper.item())
    else:
        up = bool(upper)

    n = A.shape[-1]
    batch = 1
    for s in A.shape[:-2]:
        batch *= s

    if A.dim() == 2:
        stride_ab, stride_am, stride_an = 0, A.stride(0), A.stride(1)
    else:
        stride_ab, stride_am, stride_an = A.stride(-3), A.stride(-2), A.stride(-1)

    W = torch.empty(A.shape, dtype=A.dtype, device=A.device)
    swb = n * n

    if n <= 32:
        # single-kernel path: whole matrix fits in one diagonal block
        SB = 16 if n <= 16 else 32
        _cholesky_small_kernel[(batch,)](
            A,
            W,
            n,
            stride_ab,
            stride_am,
            stride_an,
            swb,
            n,
            1,
            BLOCK=SB,
            num_warps=(1 if n <= 16 else 2),
        )
    else:
        IB = 64
        _init_lower_kernel[(triton.cdiv(n, IB), triton.cdiv(n, IB), batch)](
            A,
            W,
            n,
            stride_ab,
            stride_am,
            stride_an,
            swb,
            n,
            1,
            BLOCK=IB,
        )

        CB = 32
        RB = 64
        _cholesky_lower_kernel[(batch,)](
            W,
            n,
            swb,
            n,
            1,
            BLOCK=CB,
            RB=RB,
            num_warps=8,
        )

    if not up:
        return W

    U = torch.empty(A.shape, dtype=A.dtype, device=A.device)
    TB = 32
    _transpose_kernel[(triton.cdiv(n, TB), batch)](
        W,
        U,
        n,
        swb,
        n,
        1,
        swb,
        n,
        1,
        BLOCK=TB,
    )
    return U
