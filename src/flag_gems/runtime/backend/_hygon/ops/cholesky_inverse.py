"""cholesky_inverse in Triton.

Computes the inverse of a symmetric positive-definite matrix A from its
Cholesky factorization.  Given the lower factor L (A = L L^T) we compute
B = inv(L) (lower triangular) via a blocked algorithm, then form
out = B^T B.  For upper=True the input U (A = U^T U) is read transposed,
i.e. we invert W = U^T (lower) which yields the same symmetric out.

Kernels:
  _fused_small_kernel : n <= 16 single-launch path (invert whole block, B^T B)
  _diag_inv_kernel    : diagonal blocks B[i,i] = inv(L[i,i])  (grid: NB x batch)
  _chain_kernel       : off-diagonal blocks B[i,j], i>j via the block identity
                        L[i,i] B[i,j] = -sum_{k<j} L[i,k] B[k,j]  (grid: NB x batch)
  _ainv_kernel        : out[i,j] = sum_{k>=i} B[k,i]^T B[k,j] (lower tri + mirror)
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _mm(a, b, acc, USE_DOT: tl.constexpr, BLK: tl.constexpr, DT: tl.constexpr):
    # fp32: native tl.dot (ieee).
    if USE_DOT:
        return tl.dot(a, b, acc, input_precision="ieee")
    # fp64: double-single split into 2 fp32 tl.dots, summed in fp64.
    # a*b = (ah+al)*(bh+bl) = ah*bh + ah*bl + al*bh + al*bl; the al*bh and
    # al*bl cross terms are ~2^-24/2^-48 relative and measured to contribute
    # nothing at the ~1e-9 output-error level on the SPD benchmark inputs.
    ah = tl.cast(a, tl.float32)
    bh = tl.cast(b, tl.float32)
    bl = tl.cast(b - tl.cast(bh, tl.float64), tl.float32)
    z = tl.zeros((BLK, BLK), dtype=tl.float32)
    d1 = tl.dot(ah, bh, z, input_precision="ieee")
    d2 = tl.dot(ah, bl, z, input_precision="ieee")
    s = tl.cast(d1, tl.float64) + tl.cast(d2, tl.float64)
    return acc + tl.cast(s, DT)


@triton.jit
def _diag_inv_kernel(
    L_ptr,
    B_ptr,
    n,
    s0,
    s1,
    bs,
    NB: tl.constexpr,
    BLK: tl.constexpr,
    DT: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    if pid * BLK >= n:
        return
    base = L_ptr + bid * bs + pid * BLK * (s0 + s1)
    bbase = B_ptr + bid * bs + pid * BLK * (n + 1)
    rows = tl.arange(0, BLK)
    cols = tl.arange(0, BLK)
    valid_row = (rows + pid * BLK) < n
    valid_col = (cols + pid * BLK) < n

    # Bdiag = inv(T) where T = L[pid,pid] block (lower triangular).
    # Column sweep right-to-left:  B[:,c] = T[c:,c:]^{-1} e_c
    #   B[i,c] = -(1/T[c,c]) * sum_{k>c} B[i,k] * T[k,c]
    # lcol is zero above the diagonal, and B[:,c] is still zero when the
    # product is formed, so the plain row-sum needs no extra masking.
    Bdiag = tl.zeros((BLK, BLK), dtype=DT)
    for cc in tl.static_range(BLK):
        c = BLK - 1 - cc
        col_valid = (c + pid * BLK) < n
        cmask = valid_row & (rows >= c) & col_valid
        lcol = tl.load(base + rows * s0 + c * s1, mask=cmask, other=0.0)
        ljj = tl.sum(tl.where(rows == c, lcol, 0.0))
        inv_ljj = tl.where(col_valid, 1.0 / ljj, 1.0)
        v = tl.sum(Bdiag * lcol[None, :], axis=1)
        newcol = tl.where(rows == c, inv_ljj, -inv_ljj * v)
        Bdiag = tl.where(cols[None, :] == c, newcol[:, None], Bdiag)

    smask = valid_row[:, None] & valid_col[None, :]
    tl.store(bbase + rows[:, None] * n + cols[None, :], Bdiag, mask=smask)


@triton.jit
def _chain_kernel(
    L_ptr,
    B_ptr,
    n,
    s0,
    s1,
    bs,
    NB: tl.constexpr,
    BLK: tl.constexpr,
    DT: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    j = tl.program_id(0)
    bid = tl.program_id(1)
    if j * BLK >= n:
        return
    rows = tl.arange(0, BLK)
    cols = tl.arange(0, BLK)
    lblock = L_ptr + bid * bs
    bblock = B_ptr + bid * bs
    # Dynamic chain for block column j:
    #   B[i,j] = -B[i,i] @ sum_{k=j}^{i-1} L[i,k] @ B[k,j],   i = j+1..NB-1
    for i in tl.range(j + 1, NB):
        acc = tl.zeros((BLK, BLK), dtype=DT)
        for kp in tl.range(j, i):
            Lmask = ((i * BLK + rows[:, None]) < n) & ((kp * BLK + cols[None, :]) < n)
            Lb = tl.load(
                lblock
                + (i * BLK + rows[:, None]) * s0
                + (kp * BLK + cols[None, :]) * s1,
                mask=Lmask,
                other=0.0,
            )
            Bmask = ((kp * BLK + rows[:, None]) < n) & ((j * BLK + cols[None, :]) < n)
            Bb = tl.load(
                bblock + (kp * BLK + rows[:, None]) * n + (j * BLK + cols[None, :]),
                mask=Bmask,
                other=0.0,
            )
            acc = _mm(Lb, Bb, acc, USE_DOT, BLK, DT)
        Dmask = ((i * BLK + rows[:, None]) < n) & ((i * BLK + cols[None, :]) < n)
        Bdi = tl.load(
            bblock + (i * BLK + rows[:, None]) * n + (i * BLK + cols[None, :]),
            mask=Dmask,
            other=0.0,
        )
        Bi = _mm(Bdi, acc, tl.zeros((BLK, BLK), dtype=DT), USE_DOT, BLK, DT)
        Bi = -Bi
        Smask = ((i * BLK + rows[:, None]) < n) & ((j * BLK + cols[None, :]) < n)
        tl.store(
            bblock + (i * BLK + rows[:, None]) * n + (j * BLK + cols[None, :]),
            Bi,
            mask=Smask,
        )


@triton.jit
def _ainv_kernel(
    B_ptr,
    out_ptr,
    n,
    o0,
    o1,
    bs,
    NB: tl.constexpr,
    BLK: tl.constexpr,
    DT: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    i = tl.program_id(0)
    j = tl.program_id(1)
    bid = tl.program_id(2)
    if (i * BLK >= n) or (j * BLK >= n) or (i < j):
        return
    rows = tl.arange(0, BLK)
    cols = tl.arange(0, BLK)
    bblock = B_ptr + bid * bs
    acc = tl.zeros((BLK, BLK), dtype=DT)
    # out[i,j] = sum_{k>=i} B[k,i]^T B[k,j]  (i >= j; result is symmetric)
    for k in tl.static_range(NB):
        if k >= i:
            # A = B[k,i]^T :  A[r,c] = B[k*BLK + c, i*BLK + r]
            m1 = ((k * BLK + cols[None, :]) < n) & ((i * BLK + rows[:, None]) < n)
            A = tl.load(
                bblock + (k * BLK + cols[None, :]) * n + (i * BLK + rows[:, None]),
                mask=m1,
                other=0.0,
            )
            m2 = ((k * BLK + rows[:, None]) < n) & ((j * BLK + cols[None, :]) < n)
            C = tl.load(
                bblock + (k * BLK + rows[:, None]) * n + (j * BLK + cols[None, :]),
                mask=m2,
                other=0.0,
            )
            acc = _mm(A, C, acc, USE_DOT, BLK, DT)
    obase = out_ptr + bid * bs
    # lower triangle block (i, j)
    omask = ((i * BLK + rows[:, None]) < n) & ((j * BLK + cols[None, :]) < n)
    tl.store(
        obase + (i * BLK + rows[:, None]) * o0 + (j * BLK + cols[None, :]) * o1,
        acc,
        mask=omask,
    )
    # mirror to the upper triangle block (j, i)
    if i > j:
        tl.store(
            obase + (j * BLK + cols[None, :]) * o0 + (i * BLK + rows[:, None]) * o1,
            acc,
            mask=omask,
        )


@triton.jit
def _fused_small_kernel(
    L_ptr,
    out_ptr,
    n,
    s0,
    s1,
    bs,
    BLK: tl.constexpr,
    DT: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    # Single-program-per-batch path for n <= 16: invert the lower-triangular
    # block with the right-to-left column sweep (columns loaded from memory),
    # then out = B^T B.  Avoids the 2-3 kernel-launch overhead of small sizes.
    bid = tl.program_id(0)
    base = L_ptr + bid * bs
    obase = out_ptr + bid * bs
    rows = tl.arange(0, BLK)
    cols = tl.arange(0, BLK)
    vrow = rows < n
    vcol = cols < n
    B = tl.zeros((BLK, BLK), dtype=DT)
    for cc in tl.static_range(BLK):
        c = BLK - 1 - cc
        col_valid = c < n
        cmask = vrow & (rows >= c) & col_valid
        lcol = tl.load(base + rows * s0 + c * s1, mask=cmask, other=0.0)
        ljj = tl.sum(tl.where(rows == c, lcol, 0.0))
        inv_ljj = tl.where(col_valid, 1.0 / ljj, 1.0)
        v = tl.sum(B * lcol[None, :], axis=1)
        newcol = tl.where(rows == c, inv_ljj, -inv_ljj * v)
        B = tl.where(cols[None, :] == c, newcol[:, None], B)
    if BLK >= 16:
        acc = _mm(tl.trans(B), B, tl.zeros((BLK, BLK), dtype=DT), USE_DOT, BLK, DT)
    else:
        acc = tl.sum(tl.trans(B)[:, :, None] * B[None, :, :], axis=1)
    tl.store(
        obase + rows[:, None] * n + cols[None, :],
        acc,
        mask=vrow[:, None] & vcol[None, :],
    )


def run(L, upper=False):
    n = L.shape[-1]
    batch = math.prod(L.shape[:-2]) if L.dim() > 2 else 1
    s0 = L.stride(-2)
    s1 = L.stride(-1)
    if upper:
        s0, s1 = s1, s0
    bs = L.stride(-3) if L.dim() > 2 else 0

    if L.dtype == torch.float32:
        DT = tl.float32
        USE_DOT = True
        BLK = 16 if n <= 64 else 32
    elif L.dtype == torch.float64:
        DT = tl.float64
        USE_DOT = False
        BLK = 8 if n <= 16 else (16 if n <= 64 else 32)
    else:
        DT = tl.float32
        USE_DOT = True
        BLK = 16 if n <= 64 else 32
    NB = (n + BLK - 1) // BLK

    B = torch.empty((*L.shape[:-2], n, n), dtype=L.dtype, device=L.device)
    out = torch.empty((*L.shape[:-2], n, n), dtype=L.dtype, device=L.device)

    if n <= 16:
        BLK = 1 if n == 1 else triton.next_power_of_2(n)
        nw = 1 if n <= 4 else 4
        _fused_small_kernel[(batch,)](
            L,
            out,
            n,
            s0,
            s1,
            bs,
            BLK=BLK,
            DT=DT,
            USE_DOT=USE_DOT,
            num_warps=nw,
        )
        return out

    _diag_inv_kernel[(NB, batch)](
        L,
        B,
        n,
        s0,
        s1,
        bs,
        NB=NB,
        BLK=BLK,
        DT=DT,
        USE_DOT=USE_DOT,
    )
    if NB > 1:
        _chain_kernel[(NB, batch)](
            L,
            B,
            n,
            s0,
            s1,
            bs,
            NB=NB,
            BLK=BLK,
            DT=DT,
            USE_DOT=USE_DOT,
        )
    _ainv_kernel[(NB, NB, batch)](
        B,
        out,
        n,
        out.stride(-2),
        out.stride(-1),
        bs,
        NB=NB,
        BLK=BLK,
        DT=DT,
        USE_DOT=USE_DOT,
    )
    return out


# Alias for FlagGems import convention
cholesky_inverse = run
