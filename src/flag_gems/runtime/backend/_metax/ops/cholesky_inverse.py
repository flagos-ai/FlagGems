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
# cholesky_inverse (FlagGems / Gems pytest contract)
#
#   out = A^{-1} where A = L L^T (upper=False) or A = U^T U (upper=True)
#   computed WITHOUT mutating self.
#
# Identity used everywhere: with L_eff = L (lower) or L_eff = U^T (upper,
# obtained as a zero-copy stride swap of the input's last two dims),
#       Y = L_eff^{-1}   (lower triangular)   and   out = Y^T Y .
#
# Precision note: tl.dot is accurate for fp32 on this backend but returns
# garbage for fp64 operands (verified numerically), so every fp64 product is a
# tile FMA reduction.  Profiling also showed that an in-loop cross-lane
# reduction (tl.sum over a 3-D broadcast tile) costs ~9 us per step here while a
# rank-1 element-wise update costs ~0.2 us, so fp64 products use rank-1 loops:
#   * fused path   : out accumulates sum_k y_k y_k^T while Y is built
#   * off-diagonal : rank-1 loop over scaled L blocks (see below)
#   * gram         : rank-1 accumulation over k
#
# Dispatch (metadata only):
#   n <= 16          : single fused launch, one program per matrix
#   16 < n <= 64     : column-parallel substitution + tiled gram (2 launches)
#   n > 64           : blocked path
#     K0 diag inverse: one warp per column of each diagonal block inverse
#     K1 scale       : M_ik = D_i L_ik for k < i, fully parallel over (i,k)
#                      (fp64 only; it removes the per-block-row post-product)
#     K2 chain       : Y_ij = -sum_k M_ik Y_kj   (fp32: D_i applied by tl.dot)
#     K3 gram        : out = Y^T Y
# ---------------------------------------------------------------------------

_FUSED_MAX = 16
_COLSUB_MAX = 64


@triton.jit
def _fused_kernel(
    L,
    Out,
    n,
    sr,
    sc,
    sb,
    BLOCK_N: tl.constexpr,
    F64: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n
    col_mask = cols < n

    Y = tl.zeros((BLOCK_N, BLOCK_N), dtype=L.dtype.element_ty)
    if F64:
        out_acc = tl.zeros((BLOCK_N, BLOCK_N), dtype=L.dtype.element_ty)
    base = L + pid * sb

    for i in range(0, n):
        off = base + i * sr
        lrow = tl.load(off + cols * sc, mask=col_mask & (cols < i), other=0.0)
        lii = tl.load(off + i * sc)
        acc = tl.sum(lrow[:, None] * Y, axis=0)
        nr = tl.where(cols == i, 1.0, 0.0).to(L.dtype.element_ty) - acc
        nr = nr / lii
        nr = tl.where(col_mask, nr, 0.0)
        if F64:
            out_acc += nr[:, None] * nr[None, :]
        Y = tl.where(rows[:, None] == i, nr[None, :], Y)

    if F64:
        res = out_acc
    else:
        res = tl.dot(tl.trans(Y), Y, out_dtype=tl.float32, input_precision="ieee")

    out_off = Out + pid * n * n + rows[:, None] * n + cols[None, :]
    tl.store(out_off, res, mask=row_mask[:, None] & col_mask[None, :])


@triton.jit
def _colsub_kernel(
    L,
    Ysc,
    n,
    sr,
    sc,
    sb,
    BLOCK_N: tl.constexpr,
):
    """One program per column c of Y = L_eff^{-1}; entries i >= c only."""
    b = tl.program_id(0)
    c = tl.program_id(1)
    idx = tl.arange(0, BLOCK_N)
    base = L + b * sb
    y = tl.zeros((BLOCK_N,), dtype=L.dtype.element_ty)
    for i in range(c, n):
        lrow = tl.load(base + i * sr + idx * sc, mask=idx < i, other=0.0)
        lii = tl.load(base + i * sr + i * sc)
        acc = tl.sum(lrow * y, axis=0)
        val = (tl.where(i == c, 1.0, 0.0).to(L.dtype.element_ty) - acc) / lii
        y = tl.where(idx == i, val, y)
    tl.store(Ysc + b * n * n + idx * n + c, y, mask=idx < n)


@triton.jit
def _diag_inv_col_kernel(
    L,
    Ysc,
    n,
    sr,
    sc,
    sb,
    BK: tl.constexpr,
):
    """One program per column of each BK x BK diagonal block inverse."""
    b = tl.program_id(0)
    j = tl.program_id(1)
    c = tl.program_id(2)
    i0 = j * BK
    idx = tl.arange(0, BK)
    gc = i0 + c
    base = L + b * sb
    y = tl.zeros((BK,), dtype=L.dtype.element_ty)
    for i in range(c, BK):
        gt = i0 + i
        lrow = tl.load(
            base + gt * sr + (i0 + idx) * sc, mask=(idx < i) & (gc < n), other=0.0
        )
        lii = tl.load(base + gt * sr + gt * sc, mask=gt < n, other=1.0)
        acc = tl.sum(lrow * y, axis=0)
        val = (tl.where(i == c, 1.0, 0.0).to(L.dtype.element_ty) - acc) / lii
        y = tl.where(idx == i, val, y)
    tl.store(Ysc + b * n * n + (i0 + idx) * n + gc, y, mask=(gc < n) & (i0 + idx < n))


@triton.jit
def _premul_kernel(
    L,
    Ysc,
    Msc,
    n,
    sr,
    sc,
    sb,
    BK: tl.constexpr,
):
    """M_ik = D_i L_ik  (k < i) with D_i = inv(L_ii) read from Ysc.
    Fully parallel over (block row i, block col k); rank-1 element updates."""
    b = tl.program_id(0)
    i = tl.program_id(1)
    k = tl.program_id(2)
    r = tl.arange(0, BK)
    c = tl.arange(0, BK)
    gi = i * BK + r
    gkc = k * BK + c
    imask = gi < n
    kmask = gkc < n
    base = L + b * sb
    ybase = Ysc + b * n * n
    acc = tl.zeros((BK, BK), dtype=L.dtype.element_ty)
    for t in range(0, BK):
        gt = i * BK + t
        dv = tl.load(ybase + gi * n + gt, mask=imask & (gt < n), other=0.0)
        lv = tl.load(base + gt * sr + gkc * sc, mask=(gt < n) & kmask, other=0.0)
        acc += dv[:, None] * lv[None, :]
    tl.store(
        Msc + b * n * n + gkc[:, None] * n + gi[None, :],
        tl.trans(acc),
        mask=kmask[:, None] & imask[None, :] & (k < i),
    )


@triton.jit
def _offdiag_inv_kernel(
    L,
    Ysc,
    Msc,
    n,
    sr,
    sc,
    sb,
    nb,
    BK: tl.constexpr,
    F64: tl.constexpr,
):
    b = tl.program_id(0)
    j = tl.program_id(1)
    r = tl.arange(0, BK)
    c = tl.arange(0, BK)
    gj = j * BK + c
    jmask = gj < n
    base = L + b * sb
    ybase = Ysc + b * n * n
    mbase = Msc + b * n * n

    for i in range(j + 1, nb):
        gi = i * BK + r
        imask = gi < n
        acc = tl.zeros((BK, BK), dtype=L.dtype.element_ty)
        if F64:
            # Y_ij = -sum_k M_ik Y_kj, no post-multiplication needed.
            # M is stored transposed so the column of M is a contiguous row.
            for gk in range(j * BK, i * BK):
                mv = tl.load(mbase + gk * n + gi, mask=(gk < n) & imask, other=0.0)
                yv = tl.load(ybase + gk * n + gj, mask=(gk < n) & jmask, other=0.0)
                acc += mv[:, None] * yv[None, :]
            res = -acc
        else:
            for k in range(j, i):
                gkc = k * BK + c
                rk = k * BK + r
                lt = tl.load(
                    base + gi[:, None] * sr + gkc[None, :] * sc,
                    mask=imask[:, None] & (gkc < n)[None, :],
                    other=0.0,
                )
                yt = tl.load(
                    ybase + rk[:, None] * n + gj[None, :],
                    mask=(rk < n)[:, None] & jmask[None, :],
                    other=0.0,
                )
                acc = tl.dot(lt, yt, acc, out_dtype=tl.float32, input_precision="ieee")
            di = tl.load(
                ybase + gi[:, None] * n + gi[None, :],
                mask=imask[:, None] & imask[None, :],
                other=0.0,
            )
            zero = tl.zeros((BK, BK), dtype=L.dtype.element_ty)
            res = -tl.dot(di, acc, zero, out_dtype=tl.float32, input_precision="ieee")
        tl.store(
            ybase + gi[:, None] * n + gj[None, :],
            res,
            mask=imask[:, None] & jmask[None, :],
        )


@triton.jit
def _gram_kernel(
    Ysc,
    Out,
    n,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    F64: tl.constexpr,
):
    b = tl.program_id(0)
    pm = tl.program_id(1)
    pn = tl.program_id(2)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    yb = Ysc + b * n * n
    mmask = rm < n
    nmask = rn < n

    acc = tl.zeros((BM, BN), dtype=Ysc.dtype.element_ty)
    if F64:
        for k in range(0, n):
            av = tl.load(yb + k * n + rm, mask=mmask & (k >= rm), other=0.0)
            bv = tl.load(yb + k * n + rn, mask=nmask & (k >= rn), other=0.0)
            acc += av[:, None] * bv[None, :]
    else:
        for k0 in range(0, n, BK):
            kk = k0 + rk
            kmask = kk < n
            a = tl.load(
                yb + kk[:, None] * n + rm[None, :],
                mask=kmask[:, None] & (kk[:, None] >= rm[None, :]) & mmask[None, :],
                other=0.0,
            )
            bb = tl.load(
                yb + kk[:, None] * n + rn[None, :],
                mask=kmask[:, None] & (kk[:, None] >= rn[None, :]) & nmask[None, :],
                other=0.0,
            )
            acc = tl.dot(
                tl.trans(a), bb, acc, out_dtype=tl.float32, input_precision="ieee"
            )

    tl.store(
        Out + b * n * n + rm[:, None] * n + rn[None, :],
        acc,
        mask=mmask[:, None] & nmask[None, :],
    )


def _num_warps(block_n):
    if block_n >= 64:
        return 8
    if block_n >= 32:
        return 4
    return 2


def _gram_cfg(n, f64):
    if n <= 64:
        return 16, (8 if f64 else 16)
    if n <= 128:
        return 32, (4 if f64 else 32)
    return (32, 4) if f64 else (64, 32)


def cholesky_inverse(self, upper=False):
    if self.numel() == 0:
        return self

    n = self.size(-1)
    B = self.numel() // (n * n)
    sr = self.stride(-2)
    sc = self.stride(-1)
    sb = self.stride(-3) if self.dim() >= 3 else 0
    if upper:
        sr, sc = sc, sr

    out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    f64 = self.dtype == torch.float64

    if n <= _FUSED_MAX:
        block_n = max(16, triton.next_power_of_2(n))
        _fused_kernel[(B,)](
            self,
            out,
            n,
            sr,
            sc,
            sb,
            BLOCK_N=block_n,
            F64=f64,
            num_warps=_num_warps(block_n),
            num_stages=1,
        )
        return out

    bmn, bkc = _gram_cfg(n, f64)

    if n <= _COLSUB_MAX:
        block_n = max(16, triton.next_power_of_2(n))
        ysc = torch.empty((B, n, n), dtype=self.dtype, device=self.device)
        _colsub_kernel[(B, n)](
            self,
            ysc,
            n,
            sr,
            sc,
            sb,
            BLOCK_N=block_n,
            num_warps=1,
            num_stages=4,
        )
    else:
        bk = 16 if (not f64 and n <= 128) else 32
        nb = (n + bk - 1) // bk
        ysc = torch.empty((B, n, n), dtype=self.dtype, device=self.device)
        _diag_inv_col_kernel[(B, nb, bk)](
            self,
            ysc,
            n,
            sr,
            sc,
            sb,
            BK=bk,
            num_warps=1,
            num_stages=4,
        )
        if f64:
            msc = torch.empty((B, n, n), dtype=self.dtype, device=self.device)
            _premul_kernel[(B, nb, nb)](
                self,
                ysc,
                msc,
                n,
                sr,
                sc,
                sb,
                BK=bk,
                num_warps=4,
                num_stages=3,
            )
        else:
            msc = ysc
        _offdiag_inv_kernel[(B, nb)](
            self,
            ysc,
            msc,
            n,
            sr,
            sc,
            sb,
            nb,
            BK=bk,
            F64=f64,
            num_warps=4,
            num_stages=3,
        )

    _gram_kernel[(B, triton.cdiv(n, bmn), triton.cdiv(n, bmn))](
        ysc,
        out,
        n,
        BM=bmn,
        BN=bmn,
        BK=bkc,
        F64=f64,
        num_warps=2 if bmn <= 16 else 4,
        num_stages=3,
    )
    return out
