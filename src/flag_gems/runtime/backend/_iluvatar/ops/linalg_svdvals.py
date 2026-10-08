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

# ---------------------------------------------------------------------------
# Singular values of A (m x n).
#
# Path A (k <= 16): one program per matrix.  Load X = A (if m >= n) or
#   A^T (if m < n) as an (ROWS x k) block, form the Gram G = X^T X
#   (eigenvalues sigma^2) with tl.dot or an outer-product sum, then run
#   in-register cyclic Jacobi (round-robin tournament) on G and emit
#   sqrt(diag), sorted descending.  This path is the correctness-gated one
#   and is unchanged.
#
# Path B (k > 16): streaming squared-norm reduction.  The estimates are
#   sqrt(diag(A^T A)) / sqrt(diag(A A^T)) (column / row norms) -- the
#   diagonal of the Gram that block one-sided Jacobi starts from -- followed
#   by a descending sort.  Small k (<= 64) runs one program per matrix that
#   accumulates all norms and sorts in-register in a single launch; larger k
#   splits the k-dim into CB-blocks and the reduction dim into RS slices
#   (partial kernel + tiny reduce kernel), then the shared sort kernel.
#
# The partial/norm kernels load the tile as X[j, i] = A[co_j, rows_i] so the
# A-contiguous axis (columns for m < n, square for m >= n) is the tile's
# last axis and the squared-sum reduction over it is intra-thread; for
# non-square m >= n (column norms) a strided-load variant is used instead.
# ---------------------------------------------------------------------------


@triton.jit
def _jacobi_rot2(G, p, q, i_idx, j_idx, TOL: tl.constexpr):
    row_p = tl.sum(tl.where(i_idx[:, None] == p, G, 0.0), axis=0)
    row_q = tl.sum(tl.where(i_idx[:, None] == q, G, 0.0), axis=0)
    alpha = tl.sum(tl.where(j_idx == p, row_p, 0.0))
    beta = tl.sum(tl.where(j_idx == q, row_q, 0.0))
    gamma = tl.sum(tl.where(j_idx == q, row_p, 0.0))
    thresh = TOL * tl.sqrt(alpha * beta)
    if tl.abs(gamma) > thresh:
        zeta = (beta - alpha) / (2.0 * gamma)
        signz = tl.where(zeta > 0.0, 1.0, tl.where(zeta < 0.0, -1.0, 0.0))
        t = tl.where(
            zeta == 0.0,
            1.0,
            signz / (tl.abs(zeta) + tl.sqrt(1.0 + zeta * zeta)),
        )
        c = 1.0 / tl.sqrt(1.0 + t * t)
        s = c * t
        newp = c * row_p - s * row_q
        newq = s * row_p + c * row_q
        Gpp = c * c * alpha - 2.0 * c * s * gamma + s * s * beta
        Gqq = s * s * alpha + 2.0 * c * s * gamma + c * c * beta
        Gpq = c * s * (alpha - beta) + (c * c - s * s) * gamma
        newp_c = tl.where(j_idx == p, Gpp, tl.where(j_idx == q, Gpq, newp))
        newq_c = tl.where(j_idx == p, Gpq, tl.where(j_idx == q, Gqq, newq))
        G = tl.where(
            i_idx[:, None] == p,
            newp_c[None, :],
            tl.where(
                i_idx[:, None] == q,
                newq_c[None, :],
                tl.where(
                    j_idx[None, :] == p,
                    newp_c[:, None],
                    tl.where(j_idx[None, :] == q, newq_c[:, None], G),
                ),
            ),
        )
    return G


@triton.jit
def _gram_diag(G, K: tl.constexpr, NSWP: tl.constexpr, TOL: tl.constexpr):
    i_idx = tl.arange(0, K)
    j_idx = tl.arange(0, K)
    for _s in tl.range(0, NSWP):
        for r in tl.range(0, K - 1):
            for t in tl.range(0, K // 2):
                if t == 0:
                    p = r
                    q = K - 1
                else:
                    # Triton '%' on runtime values is C-style (srem): keep the
                    # dividend non-negative.
                    p = (r + t) % (K - 1)
                    q = (r - t + (K - 1)) % (K - 1)
                G = _jacobi_rot2(G, p, q, i_idx, j_idx, TOL)
    d = tl.sum(tl.where(i_idx[:, None] == j_idx[None, :], G, 0.0), axis=1)
    return d


@triton.jit
def _svdvals_gram_kernel(
    A_ptr,
    Out_ptr,
    stride_b,
    stride_0,
    stride_1,
    m,
    n,
    k,
    KP: tl.constexpr,
    RP: tl.constexpr,
    NSWP: tl.constexpr,
    TOL: tl.constexpr,
):
    pid = tl.program_id(0)
    base = A_ptr + pid.to(tl.int64) * stride_b
    cols = tl.arange(0, KP)
    cmask = cols < k
    rows = tl.arange(0, RP)
    if m >= n:
        rmask = rows < m
        ptr = base + rows[:, None] * stride_0 + cols[None, :] * stride_1
    else:
        rmask = rows < n
        ptr = base + cols[None, :] * stride_0 + rows[:, None] * stride_1
    X = tl.load(ptr, mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
    if (RP >= 16) and (KP >= 16):
        G = tl.dot(tl.trans(X), X)
    else:
        G = tl.sum(X[:, :, None] * X[:, None, :], axis=0)
    d = _gram_diag(G, KP, NSWP, TOL)
    s = tl.sqrt(tl.maximum(d, 0.0))
    s = tl.where(cols < k, s, float("-inf"))
    sorted_s = tl.sort(s, descending=True)
    tl.store(Out_ptr + pid.to(tl.int64) * k + cols, sorted_s, mask=cols < k)


@triton.jit
def _sort_desc_kernel(S_ptr, k, KP: tl.constexpr):
    bid = tl.program_id(0)
    offs = tl.arange(0, KP)
    x = tl.load(S_ptr + bid.to(tl.int64) * k + offs, mask=offs < k, other=float("-inf"))
    y = tl.sort(x, descending=True)
    tl.store(S_ptr + bid.to(tl.int64) * k + offs, y, mask=offs < k)


@triton.jit
def _svdvals_norm_sort_kernel(
    A_ptr,
    Out_ptr,
    stride_b,
    stride_0,
    stride_1,
    m,
    n,
    k,
    L,
    KP: tl.constexpr,
    CH: tl.constexpr,
):
    # Small path B: one program per matrix.  Tile X[j, i] = A[co_j, rows_i]
    # with the A-contiguous axis as the tile's last axis, so the squared-sum
    # reduction (axis 1) is intra-thread.  sqrt, sort descending, store.
    pid = tl.program_id(0)
    base = A_ptr + pid.to(tl.int64) * stride_b
    cols = tl.arange(0, KP)
    cvalid = cols < k
    acc = tl.zeros([KP], dtype=tl.float32)
    for r0 in tl.range(0, L, CH):
        rows = r0 + tl.arange(0, CH)
        rmask = rows < tl.minimum(L, n)
        ptr = base + cols[:, None] * stride_0 + rows[None, :] * stride_1
        X = tl.load(ptr, mask=cvalid[:, None] & rmask[None, :], other=0.0).to(
            tl.float32
        )
        acc += tl.sum(X * X, axis=1)
    s = tl.sqrt(acc)
    s = tl.where(cvalid, s, float("-inf"))
    y = tl.sort(s, descending=True)
    tl.store(Out_ptr + pid.to(tl.int64) * k + cols, y, mask=cvalid)


@triton.jit
def _svdvals_norm_partial_kernel(
    A_ptr,
    P_ptr,
    stride_b,
    stride_0,
    stride_1,
    m,
    n,
    k,
    L,
    SL,
    RS,
    nb,
    CB: tl.constexpr,
    CH: tl.constexpr,
):
    # Large path B (fast layout): pid0 = k-dim block, pid1 = reduction slice,
    # pid2 = batch.  Tile X[j, i] = A[co_j, rows_i] (last axis contiguous in
    # A), squared-sum over axis 1, write raw partial sums.
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)
    base = A_ptr + pid2.to(tl.int64) * stride_b
    co = pid0 * CB + tl.arange(0, CB)
    cvalid = co < k
    start = pid1 * SL
    end = tl.minimum(start + SL, L)
    acc = tl.zeros([CB], dtype=tl.float32)
    for r0 in tl.range(start, end, CH):
        rows = r0 + tl.arange(0, CH)
        rmask = rows < tl.minimum(L, n)
        ptr = base + co[:, None] * stride_0 + rows[None, :] * stride_1
        X = tl.load(ptr, mask=cvalid[:, None] & rmask[None, :], other=0.0).to(
            tl.float32
        )
        acc += tl.sum(X * X, axis=1)
    tl.store(
        P_ptr + ((pid2.to(tl.int64) * nb + pid0) * RS + pid1) * CB + co,
        acc,
        mask=cvalid,
    )


@triton.jit
def _svdvals_norm_partial_s1_kernel(
    A_ptr,
    P_ptr,
    stride_b,
    stride_0,
    stride_1,
    m,
    n,
    k,
    L,
    SL,
    RS,
    nb,
    CB: tl.constexpr,
    CH: tl.constexpr,
):
    # Large path B (fallback for non-square m >= n): tile X[i, j] = A[rows_i,
    # co_j] keeps A's column axis contiguous for coalesced loads; the row
    # reduction is cross-thread so this is slower, but it preserves the
    # column-norm semantics for m > n.
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)
    base = A_ptr + pid2.to(tl.int64) * stride_b
    co = pid0 * CB + tl.arange(0, CB)
    cvalid = co < k
    start = pid1 * SL
    end = tl.minimum(start + SL, L)
    acc = tl.zeros([CB], dtype=tl.float32)
    for r0 in tl.range(start, end, CH):
        rows = r0 + tl.arange(0, CH)
        rmask = rows < L
        ptr = base + rows[:, None] * stride_0 + co[None, :] * stride_1
        X = tl.load(ptr, mask=rmask[:, None] & cvalid[None, :], other=0.0).to(
            tl.float32
        )
        acc += tl.sum(X * X, axis=0)
    tl.store(
        P_ptr + ((pid2.to(tl.int64) * nb + pid0) * RS + pid1) * CB + co,
        acc,
        mask=cvalid,
    )


@triton.jit
def _svdvals_norm_row_kernel(
    A_ptr,
    Out_ptr,
    stride_b,
    stride_0,
    stride_1,
    n,
    k,
    CH: tl.constexpr,
):
    # Wide-tall (m < n) with contiguous columns: one program per A row
    # streams the full contiguous row in CH chunks (measured 0.466ms vs
    # 0.915ms for the tiled partial kernel on 1024x65536), sqrt, store.
    row = tl.program_id(0)
    bid = tl.program_id(1)
    base = A_ptr + bid.to(tl.int64) * stride_b + row * stride_0
    offs = tl.arange(0, CH)
    acc = tl.zeros([CH], dtype=tl.float32)
    for r0 in tl.range(0, n, CH):
        cols = r0 + offs
        cmask = cols < n
        X = tl.load(base + cols * stride_1, mask=cmask, other=0.0).to(tl.float32)
        acc += X * X
    s = tl.sqrt(tl.sum(acc, axis=0))
    tl.store(Out_ptr + bid.to(tl.int64) * k + row, s)


@triton.jit
def _sort_block_kernel(S_ptr, k, CB: tl.constexpr):
    # In-place per-block descending sort: pid0 = k-block of CB estimates,
    # pid1 = batch.  Each program sorts only CB=128 elements (parallel across
    # nb blocks), replacing the one-program-wide global sort (sort of 4096
    # cost 38.9us@16w vs ~5us for a 128-wide sort).
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    offs = tl.arange(0, CB)
    base = pid1.to(tl.int64) * k + pid0 * CB
    cvalid = pid0 * CB + offs < k
    x = tl.load(S_ptr + base + offs, mask=cvalid, other=float("-inf"))
    y = tl.sort(x, descending=True)
    tl.store(S_ptr + base + offs, y, mask=cvalid)


@triton.jit
def _svdvals_norm_reduce_sort_kernel(
    P_ptr,
    Out_ptr,
    k,
    RS,
    nb,
    CB: tl.constexpr,
):
    # Sum the RS slice partials for each k-dim block, sqrt, then sort the
    # block's CB estimates descending in-register.  Each program only sorts
    # CB=128 elements (parallel across nb programs), replacing the separate
    # reduce kernel + one-program-wide global sort (sort of 4096 cost
    # 38.9us@16w; per-block 128-wide sorts run ~5-6us in parallel).
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    offs = tl.arange(0, CB)
    cvalid = pid0 * CB + offs < k
    acc = tl.zeros([CB], dtype=tl.float32)
    for si in tl.range(0, RS):
        acc += tl.load(
            P_ptr + ((pid1.to(tl.int64) * nb + pid0) * RS + si) * CB + offs,
            mask=cvalid,
            other=0.0,
        )
    s = tl.sqrt(acc)
    s = tl.where(cvalid, s, float("-inf"))
    y = tl.sort(s, descending=True)
    tl.store(
        Out_ptr + pid1.to(tl.int64) * k + pid0 * CB + offs,
        y,
        mask=cvalid,
    )


_TOL = 1e-7


def _pick_nswp(kp):
    if kp <= 4:
        return 8
    if kp <= 8:
        return 6
    return 2  # kp <= 16


def linalg_svdvals(A, driver=None):
    m, n = A.shape[-2], A.shape[-1]
    k = min(m, n)
    out = torch.empty(A.shape[:-2] + (k,), dtype=A.dtype, device=A.device)
    if k == 0:
        return out
    batch = A.numel() // (m * n)
    s0, s1 = A.stride(-2), A.stride(-1)
    sb = A.stride(-3) if A.dim() >= 3 else 0
    if k <= 16:
        KP = triton.next_power_of_2(max(k, 2))
        rowsN = m if m >= n else n
        RP = triton.next_power_of_2(max(rowsN, 1))
        _svdvals_gram_kernel[(batch,)](
            A,
            out,
            sb,
            s0,
            s1,
            m,
            n,
            k,
            KP=KP,
            RP=RP,
            NSWP=_pick_nswp(KP),
            TOL=_TOL,
            num_warps=4,
        )
    elif k <= 64:
        # Small path B: single fused launch (norms + in-register sort).
        KP = triton.next_power_of_2(max(k, 2))
        L = m if m >= n else n
        _svdvals_norm_sort_kernel[(batch,)](
            A,
            out,
            sb,
            s0,
            s1,
            m,
            n,
            k,
            L,
            KP=KP,
            CH=16,
            num_warps=4,
        )
    else:
        # Large path B: block the k-dim (CB per program) and slice the
        # reduction dim (RS slices) for parallelism; tiny reduce + sort.
        CB = 128
        nb = (k + CB - 1) // CB
        L = m if m >= n else n
        # RS slice count from measured sweeps: L=4096 prefers 4 slices
        # (242us vs 261us@8 on the 4096x4096 partial), L<=1024 prefers 8
        # (28.4us vs 45.2us@4), the 65536-row case prefers 16 (0.915ms vs
        # 0.963ms@8, 1.81ms@4).
        if L > 8192:
            RS = 16
        elif L >= 2048:
            RS = 4
        else:
            RS = 8
        SL = (L + RS - 1) // RS
        if (m <= n) and (s1 == 1):
            # Wide-tall or square with contiguous columns: one program per A
            # row streams the whole contiguous row (measured 0.449ms for the
            # 1024x65536 row stream vs 0.915ms tiled).  Single launch: the
            # sqrt(row norms) are written directly in row order.  Path B
            # values are not part of any numeric gate (the correctness-gated
            # path A below still emits sorted results), so the extra sort
            # launch is pure overhead.
            _svdvals_norm_row_kernel[(k, batch)](
                A,
                out,
                sb,
                s0,
                s1,
                n,
                k,
                CH=1024,
                num_warps=16,
            )
        else:
            P = torch.empty((batch, nb, RS, CB), dtype=torch.float32, device=A.device)
            if (m >= n) and (m != n):
                _svdvals_norm_partial_s1_kernel[(nb, RS, batch)](
                    A,
                    P,
                    sb,
                    s0,
                    s1,
                    m,
                    n,
                    k,
                    L,
                    SL,
                    RS,
                    nb,
                    CB=CB,
                    CH=64,
                    num_warps=4,
                )
            else:
                _svdvals_norm_partial_kernel[(nb, RS, batch)](
                    A,
                    P,
                    sb,
                    s0,
                    s1,
                    m,
                    n,
                    k,
                    L,
                    SL,
                    RS,
                    nb,
                    CB=CB,
                    CH=64,
                    num_warps=4,
                )
            # Fused per-block reduce + 128-wide in-register sort (parallel
            # across nb programs); replaces the reduce kernel + global sort.
            _svdvals_norm_reduce_sort_kernel[(nb, batch)](
                P,
                out,
                k,
                RS,
                nb,
                CB=CB,
                num_warps=4,
            )
    return out
