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
"""metax-private ``torch.linalg.qr`` (``aten::linalg_qr``).

The generic implementation (``flag_gems.ops.linalg_qr``) is tuned for NVIDIA
H20 and is fully correct there.  This backend override compensates for three
metax (MACA) platform quirks, each verified in isolation on the hardware:

1. **fp64 ``tl.dot`` returns garbage** (deterministically wrong values, all
   tile shapes / warp counts; fp32 dot is exact).  Kernels that use ``tl.dot``
   on the fp64 code paths get dot-free copies here
   (``_larft_kernel_mx`` / ``_larfb_kernel_mx`` / ``_tsqr_apply_kernel_mx``),
   with the matrix products computed as row-chunked broadcast (rank-CH)
   updates -- verified to ~1e-16 vs CPU references.  fp32 keeps ``tl.dot``
   everywhere.
2. **The fused multi-panel Q-assembly kernel miscompiles** (same class of
   failure as the iluvatar override): ``_assemble_q_fused_kernel``'s single
   launch does identity + reverse panel loop with cross-iteration global
   write->read of Q, and returns a wrong Q on this backend.  Q assembly uses
   the stream-ordered variant (identity kernel + one larfb per panel), so
   every cross-panel dependency is ordered by the stream.  All other generic
   pieces (fused small-matrix path, TSQR local/tree kernels, geqrt panels)
   are verified correct here and reused as-is.
3. **64 KB shared memory per CTA** (H20: 228 KB) and ~128 KB of private
   (register) tile budget.  The two limits bite different kernels, so the
   tile caps are recalibrated from on-hardware measurements
   (``_GEQRT_SRAM_MAX_M``, the ``_TSQR_MX_*`` budgets below); the
   H20-calibrated caps raise ``OutOfResources`` (or spill) at kernel launch.

Structure: this module is SELF-CONTAINED -- it never mutates the generic
module.  Generic kernels / helpers that are correct on this backend are
imported and launched directly; the fp64-broken or miscalibrated pieces get
``*_mx`` variants here, and the top-level routing (``_linalg_qr_mx``) mirrors
the generic ``_linalg_qr`` with those variants plugged in.  Only
``float32`` / ``float64`` are supported, matching the generic op.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_qr import (
    _ASSEMBLE_RM,
    _ASSEMBLE_TN,
    _ASSEMBLE_WARPS,
    _FUSED_DIM,
    _FUSED_ELEM,
    _FUSED_M,
    _FUSED_TALL_M,
    _GEQRT_SRAM_WARPS,
    _LARFB_RM,
    _LARFB_TN,
    _LARFB_WARPS,
    _LARFT_WARPS,
    _MCTA_MIN_NC,
    _MCTA_NC_MAX,
    _MCTA_RM,
    _MCTA_RM_FP64,
    _MCTA_WARPS,
    _PANEL_IB,
    _PANEL_RM,
    _TSQR_ASPECT,
    _TSQR_BR,
    _TSQR_FOLD_ELEM,
    _TSQR_MAX_N,
    _TSQR_MIN_M,
    _TSQR_TREE_FLAT_ROWS,
    _TSQR_TREE_WARPS,
    _assemble_q_single_panel_kernel,
    _fused_qr,
    _geqrt_mcta_kernel,
    _identity_kernel,
    _larfb_kernel,
    _larft_kernel,
    _launch_geqrt,
    _launch_geqrt_sram,
    _triu_copy,
    _tsqr_apply_kernel,
    _tsqr_local_sram_kernel,
    _tsqr_tree_kernel,
    _validate_mode,
    _validate_out,
)
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# dot-free matrix helpers (fp64 tl.dot is broken on this backend -- see #1)
# ---------------------------------------------------------------------------
@triton.jit
def _mm_row_col(A, B, dt, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr):
    """C[m, n] = sum_k A[m, k] * B[k, n] via outer-product accumulation.

    ``A`` is (M, K), ``B`` is (K, N); ``K`` serial steps of a rank-1 update.
    Only used on fp64 paths (small K = a panel / block width); fp32 keeps
    ``tl.dot``, which is exact on this backend.
    """
    ki = tl.arange(0, K)
    zero = tl.full((), 0.0, dtype=dt)
    C = tl.zeros((M, N), dtype=dt)
    for k in range(K):
        ak = tl.sum(tl.where(ki[None, :] == k, A, zero), axis=1)  # A[:, k]
        bk = tl.sum(tl.where(ki[:, None] == k, B, zero), axis=0)  # B[k, :]
        C += ak[:, None] * bk[None, :]
    return C


# ===========================================================================
# dot-free variant of the generic _larft_kernel (fp64 only; fp32 routes to
# the generic kernel, whose tl.dot is exact here).  The Gram matrix is built
# with row-chunked broadcast products (rank-CH updates) instead of tl.dot or
# the O(K^2) extraction loop: optimal M*ib^2 work.
# ===========================================================================
@libentry()
@triton.jit
def _larft_kernel_mx(
    V,
    TAU,
    MOUT,
    M,
    ib,
    sVb,
    sVm,
    sVn,
    sTauB,
    sTauN,
    sMb,
    sMm,
    sMn,
    RM: tl.constexpr,  # unused (kept for launch-signature parity)
    IBN: tl.constexpr,
    CH: tl.constexpr,
):
    pid = tle.program_id(0)
    Vb = V + pid * sVb
    TAUb = TAU + pid * sTauB
    Tb = MOUT + pid * sMb

    dt = Vb.dtype.element_ty
    idx = tl.arange(0, IBN)  # row/col index 0..IBN-1

    # ---- Gram G = V^H V  (IBN x IBN), rank-CH broadcast updates ----
    # (CH, IBN, IBN) fp64 broadcast tile: CH=4, IBN=64 -> 32 KB registers
    G = tl.zeros((IBN, IBN), dtype=dt)
    for t in range(0, (M + CH - 1) // CH):
        rows = t * CH + tl.arange(0, CH)
        rmask = rows < M
        v_off = rows[:, None] * sVm + idx[None, :] * sVn
        Vc = tl.load(Vb + v_off, mask=rmask[:, None] & (idx[None, :] < ib), other=0.0)
        G += tl.sum(Vc[:, :, None] * Vc[:, None, :], axis=0)

    # ---- M = T^{-1} = triu(G, 1) + diag(1/tau)  (upper triangular) ----
    tau_vec = tl.load(TAUb + idx * sTauN, mask=idx < ib, other=1.0)
    inv_tau = tl.where(tau_vec != 0.0, 1.0 / tau_vec, float("inf"))
    Mmat = tl.where(idx[:, None] < idx[None, :], G, 0.0)
    Mmat = tl.where(idx[:, None] == idx[None, :], inv_tau[:, None], Mmat)

    # fp64 stores M = T^{-1}; the fp64 larfb solves against it (SOLVE path)
    out = Mmat

    tl.store(
        Tb + idx[:, None] * sMm + idx[None, :] * sMn,
        out,
        mask=(idx[:, None] < ib) & (idx[None, :] < ib),
    )


# ===========================================================================
# dot-free variant of the generic _larfb_kernel (fp64 only; fp32 routes to
# the generic kernel).  Both products (W1 = V^H C and C -= V Y) are
# row-chunked broadcast reductions -- optimal M*ib*TN work, no fp64 tl.dot,
# no O(K^2) extraction loop.  Y comes from the in-kernel triangular solve on
# M=T^{-1} (SOLVE path), as in the generic fp64 kernel.
# ===========================================================================
@triton.jit
def _larfb_kernel_mx(
    V,
    TOUT,
    C,
    M,
    ib,
    P,
    sVb,
    sVm,
    sVn,
    sTb,
    sTm,
    sTn,
    sCb,
    sCm,
    sCn,
    RM: tl.constexpr,
    IBN: tl.constexpr,
    TN: tl.constexpr,
    UPPER: tl.constexpr,
    SOLVE: tl.constexpr,
    CH: tl.constexpr,
):
    pid_b = tle.program_id(0)
    pid_p = tle.program_id(1)
    Vb = V + pid_b * sVb
    Tb = TOUT + pid_b * sTb
    Cb = C + pid_b * sCb

    dt = Cb.dtype.element_ty
    col_idx = tl.arange(0, IBN)
    p_idx = pid_p * TN + tl.arange(0, TN)
    pmask = p_idx < P

    Msram = tl.load(
        Tb + col_idx[:, None] * sTm + col_idx[None, :] * sTn,
        mask=(col_idx[:, None] < ib) & (col_idx[None, :] < ib),
        other=0.0,
    )

    # ---- W1 = V^H C[:, p-tile]  (rank-CH broadcast updates) ----
    # (CH, IBN, TN) fp64 broadcast tile: CH=4, IBN=64, TN=32 -> 64 KB regs
    W1 = tl.zeros((IBN, TN), dtype=dt)
    for t in range(0, (M + CH - 1) // CH):
        rows = t * CH + tl.arange(0, CH)
        rmask = rows < M
        v_off = rows[:, None] * sVm + col_idx[None, :] * sVn
        c_off = rows[:, None] * sCm + p_idx[None, :] * sCn
        Vc = tl.load(
            Vb + v_off, mask=rmask[:, None] & (col_idx[None, :] < ib), other=0.0
        )
        Cc = tl.load(Cb + c_off, mask=rmask[:, None] & pmask[None, :], other=0.0)
        W1 += tl.sum(Vc[:, :, None] * Cc[:, None, :], axis=0)
    W1 = tl.where(col_idx[:, None] < ib, W1, 0.0)

    # ---- Y: triangular solve on M = T^{-1} (SOLVE path) ----
    Y = tl.zeros((IBN, TN), dtype=dt)
    if UPPER:
        for jj in range(IBN):
            i = IBN - 1 - jj
            if i < ib:
                Mrow = tl.sum(tl.where(col_idx[:, None] == i, Msram, 0.0), axis=0)
                W1row = tl.sum(tl.where(col_idx[:, None] == i, W1, 0.0), axis=0)
                Mii = tl.sum(tl.where(col_idx == i, Mrow, 0.0))
                contrib = tl.sum(
                    tl.where(col_idx[:, None] > i, Mrow[:, None] * Y, 0.0), axis=0
                )
                Yrow = (W1row - contrib) * (1.0 / Mii)
                Y = tl.where(col_idx[:, None] == i, Yrow[None, :], Y)
    else:
        for i in range(IBN):
            if i < ib:
                Mcol = tl.sum(tl.where(col_idx[None, :] == i, Msram, 0.0), axis=1)
                W1row = tl.sum(tl.where(col_idx[:, None] == i, W1, 0.0), axis=0)
                Mii = tl.sum(tl.where(col_idx == i, Mcol, 0.0))
                contrib = tl.sum(
                    tl.where(col_idx[:, None] < i, Mcol[:, None] * Y, 0.0), axis=0
                )
                Yrow = (W1row - contrib) * (1.0 / Mii)
                Y = tl.where(col_idx[:, None] == i, Yrow[None, :], Y)
    Y = tl.where(col_idx[:, None] < ib, Y, 0.0)

    # ---- C[:, p-tile] -= V Y  (rank-CH broadcast updates) ----
    for t in range(0, (M + CH - 1) // CH):
        rows = t * CH + tl.arange(0, CH)
        rmask = rows < M
        v_off = rows[:, None] * sVm + col_idx[None, :] * sVn
        c_off = rows[:, None] * sCm + p_idx[None, :] * sCn
        Vc = tl.load(
            Vb + v_off, mask=rmask[:, None] & (col_idx[None, :] < ib), other=0.0
        )
        Cc = tl.load(Cb + c_off, mask=rmask[:, None] & pmask[None, :], other=0.0)
        Cc = Cc - tl.sum(Vc[:, :, None] * Y[None, :, :], axis=1)
        tl.store(Cb + c_off, Cc, mask=rmask[:, None] & pmask[None, :])


# ===========================================================================
# dot-free variant of the generic _tsqr_apply_kernel.  Only the products on
# the fp64 path change: the final Q = X @ Qti and the two-level composition
# Qti = Qti @ Qt2i.  The FOLD_TREE branch is fp32-only by routing and keeps
# its selection dot.
# ===========================================================================
@libentry()
@triton.jit
def _tsqr_apply_kernel_mx(
    V_local,
    TAU_local,
    Qt,
    Qt2,
    Q,
    Rblocks,
    Racc,
    m,
    n,
    br,
    k_max,
    grp,
    num_blocks,
    sQb,
    sQm,
    sQn,
    sRAb,
    sRAm,
    sRAn,
    BM: tl.constexpr,
    IBN: tl.constexpr,
    TWO_LEVEL: tl.constexpr,
    FOLD_TREE: tl.constexpr,
    BRMt: tl.constexpr,
    NO_DOT: tl.constexpr,
):
    pid_b = tle.program_id(0)
    block_id = tle.program_id(1)
    Vb = V_local + pid_b * (m * n)
    TAUb = TAU_local + pid_b * (num_blocks * n) + block_id * n
    stack_batch = num_blocks * n * n

    dt = Vb.dtype.element_ty
    zero = tl.full((), 0.0, dtype=dt)
    one = tl.full((), 1.0, dtype=dt)

    rm = tl.arange(0, BM)
    cn = tl.arange(0, IBN)
    blk_start = block_id * br
    M = tl.minimum(br, m - blk_start)
    nr = tl.minimum(k_max, M)
    rmask = rm < M
    cmask = cn < n

    Vt = tl.load(
        Vb + (blk_start + rm)[:, None] * n + cn[None, :],
        mask=rmask[:, None] & cmask[None, :],
        other=zero,
    )
    tau_vec = tl.load(TAUb + cn, mask=cn < nr, other=zero)

    # Q_local = H_0 ... H_{nr-1} [I; 0] (reverse order application)
    X = tl.where(rm[:, None] == cn[None, :], one, zero)
    for jj in range(nr):
        j = nr - 1 - jj
        vj = tl.sum(tl.where(cn[None, :] == j, Vt, zero), axis=1)
        tau_j = tl.sum(tl.where(cn == j, tau_vec, zero))
        w = tau_j * tl.sum(vj[:, None] * X, axis=0)
        X = X - vj[:, None] * w[None, :]

    cq = tl.arange(0, IBN)
    if FOLD_TREE:
        # (fp32-only branch, unchanged from the generic kernel)
        rt = tl.arange(0, BRMt)
        Rm_t = num_blocks * n
        gmask = rt < Rm_t
        G = tl.load(
            Rblocks + pid_b * stack_batch + rt[:, None] * n + cn[None, :],
            mask=gmask[:, None] & cmask[None, :],
            other=zero,
        )
        taut = tl.zeros([IBN], dtype=dt)
        for j in range(n):
            col_j = tl.sum(tl.where(cn[None, :] == j, G, zero), axis=1)
            alpha = tl.sum(tl.where(rt == j, col_j, zero))
            xnorm_sq = tl.sum(tl.where(rt > j, col_j * col_j, zero))
            norm = tl.sqrt(alpha * alpha + xnorm_sq)
            beta = tl.where(alpha >= zero, -norm, norm)
            reflect = xnorm_sq > zero
            beta_eff = tl.where(reflect, beta, alpha)
            tau = tl.where(reflect, (beta - alpha) / beta, zero)
            denom = alpha - beta
            v_tail = tl.where(reflect, col_j / denom, zero)  # guard 0/0 NaN
            G = tl.where((rt[:, None] == j) & (cn[None, :] == j), beta_eff, G)
            G = tl.where((rt[:, None] > j) & (cn[None, :] == j), v_tail[:, None], G)
            taut = tl.where(cn == j, tau, taut)
            vj = tl.where(rt > j, v_tail, tl.where(rt == j, one, zero))
            pmask = cn[None, :] > j
            w = tau * tl.sum(tl.where(pmask, vj[:, None] * G, zero), axis=0)
            G = tl.where(pmask, G - vj[:, None] * w[None, :], G)
        if block_id == 0:
            R_tile = tl.where(rt[:, None] <= cn[None, :], G, zero)
            tl.store(
                Racc + pid_b * sRAb + rt[:, None] * sRAm + cn[None, :] * sRAn,
                R_tile,
                mask=(rt[:, None] < n) & cmask[None, :],
            )
        Xt = tl.where(rt[:, None] == cn[None, :], one, zero)
        for jj in range(n):
            j = n - 1 - jj
            v_tail = tl.sum(tl.where(cn[None, :] == j, G, zero), axis=1)
            vj = tl.where(rt > j, v_tail, tl.where(rt == j, one, zero))
            tau_j = tl.sum(tl.where(cn == j, taut, zero))
            w = tau_j * tl.sum(vj[:, None] * Xt, axis=0)
            Xt = Xt - vj[:, None] * w[None, :]
        S = tl.where(rt[None, :] == (block_id * n + cq)[:, None], one, zero)
        Qti = tl.dot(S, Xt, allow_tf32=False)
        Qti = tl.where(cq[:, None] < n, Qti, zero)
    else:
        Qti = tl.load(
            Qt + pid_b * stack_batch + (block_id * n + cq)[:, None] * n + cn[None, :],
            mask=(cq[:, None] < n) & cmask[None, :],
            other=zero,
        )
    if TWO_LEVEL:
        # compose with the group rows of the top-level factor first
        gid = block_id // grp
        num_groups = (num_blocks + grp - 1) // grp
        Qt2i = tl.load(
            Qt2
            + pid_b * (num_groups * n * n)
            + (gid * n + cq)[:, None] * n
            + cn[None, :],
            mask=(cq[:, None] < n) & cmask[None, :],
            other=zero,
        )
        if NO_DOT:
            Qti = _mm_row_col(Qti, Qt2i, dt, IBN, IBN, IBN)
        else:
            Qti = tl.dot(Qti, Qt2i, allow_tf32=False)
    if NO_DOT:
        out = _mm_row_col(X, Qti, dt, BM, IBN, IBN)
    else:
        out = tl.dot(X, Qti, allow_tf32=False)
    tl.store(
        Q + pid_b * sQb + (blk_start + rm)[:, None] * sQm + cn[None, :] * sQn,
        out,
        mask=rmask[:, None] & cmask[None, :],
    )


# ===========================================================================
# dot-free variant of the generic _assemble_q_single_panel_kernel (fp64).
# Single panel (k <= ib): Q = I - V T V^H, with W1 = T V[p,:]^H read straight
# off V's rows and each output tile written exactly once.  T is stored as
# M = T^{-1} on the fp64 path, so W1 comes from the same in-kernel
# upper-triangular solve as _larfb_kernel_mx; the V_t @ W1 product uses the
# outer-product extraction loop (fp64 tl.dot is broken here).
# ===========================================================================
@libentry()
@triton.jit
def _assemble_q_single_panel_kernel_mx(
    V,
    T,
    Q,
    m,
    k,
    qcols,
    sVb,
    sVm,
    sVn,
    sTb,
    sTm,
    sTn,
    sQb,
    sQm,
    sQn,
    RM: tl.constexpr,
    TN: tl.constexpr,
    IBN: tl.constexpr,
):
    pid_b = tle.program_id(0)
    pid_r = tle.program_id(1)
    pid_p = tle.program_id(2)
    Vb = V + pid_b * sVb
    Tb = T + pid_b * sTb
    Qb = Q + pid_b * sQb

    dt = Qb.dtype.element_ty
    zero = tl.zeros((), dtype=dt)
    one = tl.full((), 1.0, dtype=dt)
    col_idx = tl.arange(0, IBN)  # padded reflector columns
    p_idx = pid_p * TN + tl.arange(0, TN)  # Q columns of this CTA
    pmask = p_idx < qcols
    rm = tl.arange(0, RM)

    # Z = V[p_idx, :k]^H  (IBN, TN); masks use k (reflectors), not n
    Vrows = tl.load(
        Vb + p_idx[:, None] * sVm + col_idx[None, :] * sVn,
        mask=(p_idx[:, None] < m) & (col_idx[None, :] < k),
        other=zero,
    )
    Z = tl.trans(Vrows)
    # M = T^{-1} (upper triangular, IBN x IBN)
    Msram = tl.load(
        Tb + col_idx[:, None] * sTm + col_idx[None, :] * sTn,
        mask=(col_idx[:, None] < k) & (col_idx[None, :] < k),
        other=zero,
    )

    # W1 = T @ Z  <=>  solve M W1 = Z by back substitution.  Every row-tile
    # CTA redundantly solves for W1 (k serial steps on an (IBN, TN) tile) --
    # cheap next to the row-parallel Q write it buys, and rows/tau beyond k
    # are exactly zero (Mii=+inf convention as in _larfb_kernel_mx).
    W1 = tl.zeros((IBN, TN), dtype=dt)
    for jj in range(k):
        i = k - 1 - jj
        Mrow = tl.sum(tl.where(col_idx[:, None] == i, Msram, 0.0), axis=0)
        Zrow = tl.sum(tl.where(col_idx[:, None] == i, Z, 0.0), axis=0)
        Mii = tl.sum(tl.where(col_idx == i, Mrow, 0.0))
        contrib = tl.sum(
            tl.where(col_idx[:, None] > i, Mrow[:, None] * W1, 0.0), axis=0
        )
        W1row = (Zrow - contrib) * (1.0 / Mii)
        W1 = tl.where(col_idx[:, None] == i, W1row[None, :], W1)
    W1 = tl.where(col_idx[:, None] < k, W1, zero)

    # Q_tile = I_tile - V_t @ W1, written once; one row tile per CTA
    # (outer-product loop runs k steps, not IBN -- k << IBN for narrow panels)
    rows = pid_r * RM + rm
    rmask = rows < m
    Vt = tl.load(
        Vb + rows[:, None] * sVm + col_idx[None, :] * sVn,
        mask=rmask[:, None] & (col_idx[None, :] < k),
        other=zero,
    )
    Qt = tl.where(rows[:, None] == p_idx[None, :], one, zero)
    ki = tl.arange(0, IBN)
    for kk in range(k):
        ak = tl.sum(tl.where(ki[None, :] == kk, Vt, zero), axis=1)
        bk = tl.sum(tl.where(ki[:, None] == kk, W1, zero), axis=0)
        Qt -= ak[:, None] * bk[None, :]
    tl.store(
        Qb + rows[:, None] * sQm + p_idx[None, :] * sQn,
        Qt,
        mask=rmask[:, None] & pmask[None, :],
    )


# ===========================================================================
# Row-parallel fp64 larft/larfb for tall panels.  The dot-free serial kernels
# above loop over ALL M panel rows per CTA (larft: one CTA per batch; larfb:
# one CTA per column tile), which costs ~2.7/5.3 ms per panel at m=4096.  For
# M > _PAR_GRAN the row reductions are split across RB = ceil(M/_PAR_GRAN)
# row-chunk CTAs that write partial tiles; a second, stream-ordered launch
# reduces the partials (fixed order -> deterministic, no atomics, no
# cross-CTA barriers) and finishes the panel.
# ===========================================================================
@libentry()
@triton.jit
def _larft_gram_partial_kernel_mx(
    V,
    Gpart,
    M,
    ib,
    sVb,
    sVm,
    sVn,
    sGb,
    sGr,
    sGm,
    sGn,
    GRAN: tl.constexpr,
    IBN: tl.constexpr,
    CH: tl.constexpr,
):
    """Partial Gram of one GRAN-row chunk of V: G_r = V[lo:hi]^H V[lo:hi]."""
    pid = tle.program_id(0)
    rid = tle.program_id(1)
    Vb = V + pid * sVb
    Gb = Gpart + pid * sGb + rid * sGr

    dt = Vb.dtype.element_ty
    idx = tl.arange(0, IBN)
    lo = rid * GRAN
    hi = tl.minimum(lo + GRAN, M)

    G = tl.zeros((IBN, IBN), dtype=dt)
    for t in range(0, (hi - lo + CH - 1) // CH):
        rows = lo + t * CH + tl.arange(0, CH)
        rmask = rows < hi
        v_off = rows[:, None] * sVm + idx[None, :] * sVn
        Vc = tl.load(Vb + v_off, mask=rmask[:, None] & (idx[None, :] < ib), other=0.0)
        G += tl.sum(Vc[:, :, None] * Vc[:, None, :], axis=0)
    # columns >= ib are exactly zero (Vc masked), so an unmasked store is safe
    tl.store(Gb + idx[:, None] * sGm + idx[None, :] * sGn, G)


@libentry()
@triton.jit
def _larft_finalize_kernel_mx(
    Gpart,
    TAU,
    MOUT,
    RB,
    ib,
    sGb,
    sGr,
    sGm,
    sGn,
    sTauB,
    sTauN,
    sMb,
    sMm,
    sMn,
    IBN: tl.constexpr,
):
    """G = sum_r G_r;  M = T^{-1} = triu(G, 1) + diag(1/tau)  (as _larft_kernel_mx)."""
    pid = tle.program_id(0)
    Gb = Gpart + pid * sGb
    TAUb = TAU + pid * sTauB
    Tb = MOUT + pid * sMb

    dt = TAUb.dtype.element_ty
    idx = tl.arange(0, IBN)

    G = tl.zeros((IBN, IBN), dtype=dt)
    for r in range(RB):
        G += tl.load(Gb + r * sGr + idx[:, None] * sGm + idx[None, :] * sGn)

    tau_vec = tl.load(TAUb + idx * sTauN, mask=idx < ib, other=1.0)
    inv_tau = tl.where(tau_vec != 0.0, 1.0 / tau_vec, float("inf"))
    Mmat = tl.where(idx[:, None] < idx[None, :], G, 0.0)
    Mmat = tl.where(idx[:, None] == idx[None, :], inv_tau[:, None], Mmat)
    tl.store(
        Tb + idx[:, None] * sMm + idx[None, :] * sMn,
        Mmat,
        mask=(idx[:, None] < ib) & (idx[None, :] < ib),
    )


@libentry()
@triton.jit
def _larfb_w1_partial_kernel_mx(
    V,
    C,
    Wpart,
    M,
    ib,
    P,
    sVb,
    sVm,
    sVn,
    sCb,
    sCm,
    sCn,
    sWb,
    sWr,
    sWp,
    sWm,
    sWn,
    GRAN: tl.constexpr,
    IBN: tl.constexpr,
    TN: tl.constexpr,
    CH: tl.constexpr,
):
    """Partial W1_r = V[lo:hi]^H C[lo:hi, p-tile] for one row chunk."""
    pid_b = tle.program_id(0)
    rid = tle.program_id(1)
    pid_p = tle.program_id(2)
    Vb = V + pid_b * sVb
    Cb = C + pid_b * sCb

    dt = Cb.dtype.element_ty
    col_idx = tl.arange(0, IBN)
    tn_idx = tl.arange(0, TN)
    p_idx = pid_p * TN + tn_idx
    pmask = p_idx < P
    lo = rid * GRAN
    hi = tl.minimum(lo + GRAN, M)

    W1 = tl.zeros((IBN, TN), dtype=dt)
    for t in range(0, (hi - lo + CH - 1) // CH):
        rows = lo + t * CH + tl.arange(0, CH)
        rmask = rows < hi
        v_off = rows[:, None] * sVm + col_idx[None, :] * sVn
        c_off = rows[:, None] * sCm + p_idx[None, :] * sCn
        Vc = tl.load(
            Vb + v_off, mask=rmask[:, None] & (col_idx[None, :] < ib), other=0.0
        )
        Cc = tl.load(Cb + c_off, mask=rmask[:, None] & pmask[None, :], other=0.0)
        W1 += tl.sum(Vc[:, :, None] * Cc[:, None, :], axis=0)
    Wb = Wpart + pid_b * sWb + rid * sWr + pid_p * sWp
    tl.store(Wb + col_idx[:, None] * sWm + tn_idx[None, :] * sWn, W1)


@libentry()
@triton.jit
def _larfb_solve_kernel_mx(
    TOUT,
    Wpart,
    Yout,
    RB,
    ib,
    P,
    sTb,
    sTm,
    sTn,
    sWb,
    sWr,
    sWp,
    sWm,
    sWn,
    sYb,
    sYp,
    sYm,
    sYn,
    IBN: tl.constexpr,
    TN: tl.constexpr,
    UPPER: tl.constexpr,
):
    """Y = M\\W1 per (batch, p-tile): reduces the RB row-chunk partials into
    W1 and runs the serial triangular solve on M = T^{-1} exactly once
    (convention of _larfb_kernel_mx).  Hoisting this out of the apply kernel
    removes the per-(row-chunk, p-tile) redundant partial re-reads (up to
    RB*IBN*TN elements per CTA) and repeated k-step solves."""
    pid_b = tle.program_id(0)
    pid_p = tle.program_id(1)
    Tb = TOUT + pid_b * sTb

    dt = Tb.dtype.element_ty
    col_idx = tl.arange(0, IBN)
    tn_idx = tl.arange(0, TN)

    Wb = Wpart + pid_b * sWb + pid_p * sWp
    W1 = tl.zeros((IBN, TN), dtype=dt)
    for r in range(RB):
        W1 += tl.load(Wb + r * sWr + col_idx[:, None] * sWm + tn_idx[None, :] * sWn)
    W1 = tl.where(col_idx[:, None] < ib, W1, 0.0)

    Msram = tl.load(
        Tb + col_idx[:, None] * sTm + col_idx[None, :] * sTn,
        mask=(col_idx[:, None] < ib) & (col_idx[None, :] < ib),
        other=0.0,
    )

    # ---- Y: triangular solve on M = T^{-1} (same as _larfb_kernel_mx) ----
    Y = tl.zeros((IBN, TN), dtype=dt)
    if UPPER:
        for jj in range(IBN):
            i = IBN - 1 - jj
            if i < ib:
                Mrow = tl.sum(tl.where(col_idx[:, None] == i, Msram, 0.0), axis=0)
                W1row = tl.sum(tl.where(col_idx[:, None] == i, W1, 0.0), axis=0)
                Mii = tl.sum(tl.where(col_idx == i, Mrow, 0.0))
                contrib = tl.sum(
                    tl.where(col_idx[:, None] > i, Mrow[:, None] * Y, 0.0), axis=0
                )
                Yrow = (W1row - contrib) * (1.0 / Mii)
                Y = tl.where(col_idx[:, None] == i, Yrow[None, :], Y)
    else:
        for i in range(IBN):
            if i < ib:
                Mcol = tl.sum(tl.where(col_idx[None, :] == i, Msram, 0.0), axis=1)
                W1row = tl.sum(tl.where(col_idx[:, None] == i, W1, 0.0), axis=0)
                Mii = tl.sum(tl.where(col_idx == i, Mcol, 0.0))
                contrib = tl.sum(
                    tl.where(col_idx[:, None] < i, Mcol[:, None] * Y, 0.0), axis=0
                )
                Yrow = (W1row - contrib) * (1.0 / Mii)
                Y = tl.where(col_idx[:, None] == i, Yrow[None, :], Y)
    Y = tl.where(col_idx[:, None] < ib, Y, 0.0)
    Yb = Yout + pid_b * sYb + pid_p * sYp
    tl.store(Yb + col_idx[:, None] * sYm + tn_idx[None, :] * sYn, Y)


@libentry()
@triton.jit
def _larfb_apply_kernel_mx(
    V,
    Y,
    C,
    M,
    ib,
    P,
    sVb,
    sVm,
    sVn,
    sYb,
    sYp,
    sYm,
    sYn,
    sCb,
    sCm,
    sCn,
    GRAN: tl.constexpr,
    IBN: tl.constexpr,
    TN: tl.constexpr,
    CH: tl.constexpr,
):
    """C[lo:hi, p-tile] -= V[lo:hi] Y, with Y precomputed per (batch, p-tile)
    by _larfb_solve_kernel_mx (rank-CH broadcast updates, dot-free)."""
    pid_b = tle.program_id(0)
    rid = tle.program_id(1)
    pid_p = tle.program_id(2)
    Vb = V + pid_b * sVb
    Cb = C + pid_b * sCb

    col_idx = tl.arange(0, IBN)
    tn_idx = tl.arange(0, TN)
    p_idx = pid_p * TN + tn_idx
    pmask = p_idx < P
    lo = rid * GRAN
    hi = tl.minimum(lo + GRAN, M)

    Yb = Y + pid_b * sYb + pid_p * sYp
    Yt = tl.load(Yb + col_idx[:, None] * sYm + tn_idx[None, :] * sYn)
    Yt = tl.where(col_idx[:, None] < ib, Yt, 0.0)

    # ---- C[lo:hi, p-tile] -= V Y  (rank-CH broadcast updates) ----
    for t in range(0, (hi - lo + CH - 1) // CH):
        rows = lo + t * CH + tl.arange(0, CH)
        rmask = rows < hi
        v_off = rows[:, None] * sVm + col_idx[None, :] * sVn
        c_off = rows[:, None] * sCm + p_idx[None, :] * sCn
        Vc = tl.load(
            Vb + v_off, mask=rmask[:, None] & (col_idx[None, :] < ib), other=0.0
        )
        Cc = tl.load(Cb + c_off, mask=rmask[:, None] & pmask[None, :], other=0.0)
        Cc = Cc - tl.sum(Vc[:, :, None] * Yt[None, :, :], axis=1)
        tl.store(Cb + c_off, Cc, mask=rmask[:, None] & pmask[None, :])


# ===========================================================================
# T-fused variant of the generic _geqrt_sram_kernel (fp32 only): after the
# in-SRAM reflector chain the panel tile still holds every reflector tail, so
# the compact-WY factor T (INVERT form, identical math to _larft_kernel
# INVERT=True) is built in-register and stored with the panel -- removing one
# kernel launch + one global V re-read per panel.  The launcher below feeds
# TBUF at the panel's (kk, kk) offset; callers must skip the separate larft
# launch for panels factorised here (see _panel_has_fused_t).
# ===========================================================================
@libentry()
@triton.jit
def _geqrt_sram_t_kernel_mx(
    W,
    V,
    TAU,
    TBUF,
    M,
    ib,
    kk,
    n,
    k,
    sWb,
    sWm,
    sWn,
    sVb,
    sVm,
    sVn,
    sTauB,
    sTauN,
    sTb,
    sTm,
    sTn,
    BM: tl.constexpr,
    IBN: tl.constexpr,
):
    pid = tle.program_id(0)
    Wb = W + pid * sWb
    Vb = V + pid * sVb
    TAUb = TAU + pid * sTauB
    dt = Wb.dtype.element_ty
    zero = tl.zeros((), dtype=dt)
    one = tl.full((), 1.0, dtype=dt)

    rm = tl.arange(0, BM)  # panel-local row  (0..BM-1 -> rows kk..kk+M-1)
    cn = tl.arange(0, IBN)  # panel-local col   (0..IBN-1 -> cols kk..kk+IBN-1)
    rmask = rm < M
    cmask = cn < ib

    # load the panel W[kk:kk+M, kk:kk+ib] into one SRAM tile
    rows_g = kk + rm
    cols_g = kk + cn
    A = tl.load(
        Wb + rows_g[:, None] * sWm + cols_g[None, :] * sWn,
        mask=rmask[:, None] & cmask[None, :],
        other=zero,
    )

    # tau accumulates in registers and is flushed with ONE vector store after
    # the loop -- 0-d scalar stores are unreliable on some vendor backends
    tau_arr = tl.zeros([IBN], dtype=dt)
    two = tl.arange(0, 2)
    for j in range(ib):
        col_j = tl.sum(tl.where(cn[None, :] == j, A, zero), axis=1)
        # alpha (row j) and the tail's squared norm in ONE cross-lane
        # reduction over the row axis: two dependent (BM,) reduction passes
        # become a single (BM, 2) pass, then two register-local extracts.
        pair = tl.sum(
            tl.join(
                tl.where(rm == j, col_j, zero), tl.where(rm > j, col_j * col_j, zero)
            ),
            axis=0,
        )
        alpha = tl.sum(tl.where(two == 0, pair, zero))
        xnorm_sq = tl.sum(tl.where(two == 1, pair, zero))
        norm = tl.sqrt(alpha * alpha + xnorm_sq)
        beta = tl.where(alpha >= zero, -norm, norm)
        reflect = xnorm_sq > zero
        beta_eff = tl.where(reflect, beta, alpha)
        tau = tl.where(reflect, (beta - alpha) / beta, zero)
        denom = alpha - beta
        v_tail = tl.where(reflect, col_j / denom, zero)  # guard 0/0 NaN
        # Householder vector vj (0/<j, 1/==j, tail/>j)
        vj = tl.where(rm > j, v_tail, tl.where(rm == j, one, zero))
        vj = tl.where(reflect, vj, tl.where(rm == j, one, zero))
        # write vj to the V buffer and the R/V values into the in-SRAM panel
        # (independent of the trailing reduction below -- issue it first)
        tl.store(Vb + rows_g * sVm + (kk + j) * sVn, vj, mask=rmask)
        # trailing update within the panel (cols j+1..ib), in SRAM
        pmask = cn[None, :] > j
        w = tau * tl.sum(tl.where(pmask, vj[:, None] * A, zero), axis=0)
        A = tl.where((rm[:, None] == j) & (cn[None, :] == j), beta_eff, A)
        A = tl.where((rm[:, None] > j) & (cn[None, :] == j), v_tail[:, None], A)
        # write tau
        tau_arr = tl.where(cn == j, tau, tau_arr)
        A = tl.where(pmask, A - vj[:, None] * w[None, :], A)

    # flush tau with one vector store
    tl.store(TAUb + (kk + cn) * sTauN, tau_arr, mask=cmask)

    # write the panel back to W (upper triangle holds R; strict-lower holds the
    # reflector tails, harmless -- _triu_copy only reads the upper triangle)
    tl.store(
        Wb + rows_g[:, None] * sWm + cols_g[None, :] * sWn,
        A,
        mask=rmask[:, None] & cmask[None, :],
    )

    # ---- fused DLARFT: T = (triu(V^H V, 1) + diag(1/tau))^{-1} ----
    # V from the in-register tile: strict-lower tails + unit diagonal (columns
    # beyond ib stay zero, exactly like the explicit zeros in the V buffer).
    # BM >= 16 is guaranteed by the caller so the tl.dot K-dim is legal.
    Vt = tl.where(rm[:, None] > cn[None, :], A, zero)
    Vt = tl.where((rm[:, None] == cn[None, :]) & cmask[None, :], one, Vt)
    G = tl.dot(tl.trans(Vt), Vt, allow_tf32=False)
    # dinv = 1/M_ii = tau (resp. 0 for tau == 0): division-free, and the
    # zero-tau T row comes out exactly zero as in the generic kernel.
    U = tl.where(cn[:, None] < cn[None, :], G, zero)
    X = -U * tau_arr[:, None]  # N = D^{-1} U: ROW scaling
    eye = tl.where(cn[:, None] == cn[None, :], one, zero)
    S = eye + X
    # Neumann product: X is strictly upper hence nilpotent (X^IBN == 0);
    # 5 squarings cover IBN <= 64, extra iterations are no-ops.
    for _ in range(5):
        X = tl.dot(X, X, allow_tf32=False)
        S = tl.dot(S, eye + X, allow_tf32=False)
    Tmat = S * tau_arr[None, :]  # T = S D^{-1}: COLUMN scaling
    tl.store(
        TBUF + pid * sTb + (kk + cn)[:, None] * sTm + (kk + cn)[None, :] * sTn,
        Tmat,
        mask=cmask[:, None] & cmask[None, :],
    )


def _launch_geqrt_sram_t_mx(W, V, tau, Tbuf, m, n, k, kk, ib, B):
    """SRAM-resident panel factorisation with fused T build (fp32 only)."""
    ib_active = min(ib, k - kk)
    M = m - kk
    sWb, sWm, sWn = W.stride()
    sVb, sVm, sVn = V.stride()
    sTauB, sTauN = tau.stride()
    sTb, sTm, sTn = Tbuf.stride()
    # .fn bypasses the libentry wrapper's per-launch key building (~10us on
    # this CPU); the bare @libentry() carries no tuning policy, so the raw
    # JITFunction launch is semantically identical.
    _geqrt_sram_t_kernel_mx.fn[(B,)](
        W,
        V,
        tau,
        Tbuf,
        M,
        ib_active,
        kk,
        n,
        k,
        sWb,
        sWm,
        sWn,
        sVb,
        sVm,
        sVn,
        sTauB,
        sTauN,
        sTb,
        sTm,
        sTn,
        BM=triton.next_power_of_2(M),
        IBN=max(16, triton.next_power_of_2(ib_active)),
        num_warps=_GEQRT_SRAM_WARPS,
    )


def _panel_has_fused_t(esz, m, kk, ib):
    """True when the panel at ``kk`` is factorised by the T-fused SRAM kernel
    (fp32, 16 <= pow2(M) <= sram cap) and its T tile is already in Tbuf --
    callers must not run larft for it.  ``bm >= 16`` keeps the fused kernel's
    tl.dot K-dim legal; smaller tail panels take the plain kernel + larft.
    """
    if esz != 4:
        return False
    cap = _GEQRT_SRAM_MAX_M * _PANEL_IB // max(ib, _PANEL_IB)
    bm = triton.next_power_of_2(m - kk)
    return 16 <= bm <= cap


def _launch_geqrt_mcta_mx(
    W,
    V,
    tau,
    alpha_buf,
    xnorm_buf,
    w_sum,
    rowj_buf,
    ctr,
    m,
    k,
    kk,
    ib,
    p,
    NC,
    B,
    rm=_MCTA_RM,
):
    """Same launch geometry as the generic _launch_geqrt_mcta, but through the
    raw JITFunction handle (.fn) to skip libentry's per-launch key building --
    the panel loop is CPU-launch-bound at small/mid square sizes here."""
    M = m - kk
    nr = min(ib, k - kk)
    CHUNK = (M + NC - 1) // NC
    NUM_TILES = (CHUNK + rm - 1) // rm
    sWb, sWm, sWn = W.stride()
    sVb, sVm, sVn = V.stride()
    sTauB, sTauN = tau.stride()
    sAB, sAM, _ = alpha_buf.stride()
    sXB, sXM, _ = xnorm_buf.stride()
    sWB, sWM, sWN, _ = w_sum.stride()
    sRB, sRM2, sRN2, _ = rowj_buf.stride()
    sCtrB, sCtrM, _ = ctr.stride()
    _geqrt_mcta_kernel.fn[(B, NC)](
        W,
        V,
        tau,
        alpha_buf,
        xnorm_buf,
        w_sum,
        rowj_buf,
        ctr,
        M,
        kk,
        ib,
        nr,
        p,
        sWb,
        sWm,
        sWn,
        sVb,
        sVm,
        sVn,
        sTauB,
        sTauN,
        sAB,
        sAM,
        sXB,
        sXM,
        sWB,
        sWM,
        sWN,
        sRB,
        sRM2,
        sRN2,
        sCtrB,
        sCtrM,
        CHUNK=CHUNK,
        RM=rm,
        IBN=max(16, triton.next_power_of_2(ib)),
        NC=NC,
        NUM_TILES=NUM_TILES,
        num_warps=_MCTA_WARPS,
    )


# ---------------------------------------------------------------------------
# metax launch wrappers: same contracts as the generic _launch_* helpers, but
# routing fp64 to the dot-free broadcast kernels and fp32 to the generic ones.
# Row-chunk width for the fp64 broadcast products: sized so the 3D broadcast
# tile stays <= _BCAST_ELEM fp64 elements (8192 = 64 KB of registers per CTA
# -- measured safe and ~6-8% faster than the original 4096 budget on the
# blocked fp64 path; the larft Gram tile is (CH, IBN, IBN) so CH tops out at
# 2 for IBN=64, larfb's (CH, IBN, TN) at CH=4).  A fixed CH=4 at IBN=64 for
# the Gram tile (128 KB) silently corrupts results -- never hard-code CH.
# ---------------------------------------------------------------------------
_BCAST_ELEM = 8192
# fp64 larft/larfb row-chunk size for the row-parallel path (used when the
# panel is taller than one chunk).  Swept 16..2048 on (4096,4096)/(1024,1024)
# fp64: 32-64 are the sweet spot (4096^2 r-mode 255 ms vs 320 at 512, vs 502
# at 2048); 64 keeps a small edge on the largest shapes.
_PAR_GRAN = 64

# fp64 multi-CTA rows-per-CTA: 64 for very tall panels (fp64 4096^2 r-mode
# 255 -> 245 ms) but the generic 32 below the gate -- at mid sizes the larger
# per-CTA chunk's serial cost dominates (1024^2 fp64 reduced: 35.7 -> 43.4 ms
# regression with an ungated 64).
_MCTA_RM_FP64_BIG = 64
_MCTA_FP64_BIG_M = 2048
# fp32 multi-CTA panel rows-per-CTA for mid-size panels: 32 wins where the
# barrier participants are few (1024^2 r-mode 10.1 -> 9.0 ms) but loses once
# NC hits the 64-CTA cap (4096^2 r 71.7 -> 79.4 ms), so only panels with
# M <= _MCTA_SMALL_M use it.
_MCTA_RM_SMALL = 32
_MCTA_SMALL_M = 2048
# fp32 larfb narrow-trailing tile width (see _launch_larfb_mx).
_LARFB_TN_SMALL = 16
_LARFB_TN_SMALL_MAX_P = 1024


def _bcast_ch(ibn, other):
    return max(1, min(8, _BCAST_ELEM // (ibn * max(other, 1))))


def _launch_larft_mx(V, tau, Tout, m, kk, ib, B):
    M = m - kk
    sVb, sVm, sVn = V.stride()
    sTauB, sTauN = tau.stride()
    sTb, sTm, sTn = Tout.stride()
    if V.element_size() == 8:
        ibn = max(16, triton.next_power_of_2(ib))
        if M > _PAR_GRAN:
            # tall panel: row-parallel Gram partials + finalize (see above)
            rb = (M + _PAR_GRAN - 1) // _PAR_GRAN
            Gpart = torch.empty(B, rb, ibn, ibn, dtype=V.dtype, device=V.device)
            _larft_gram_partial_kernel_mx[(B, rb)](
                V,
                Gpart,
                M,
                ib,
                sVb,
                sVm,
                sVn,
                *Gpart.stride(),
                GRAN=_PAR_GRAN,
                IBN=ibn,
                CH=_bcast_ch(ibn, ibn),
                num_warps=_LARFT_WARPS,
                num_stages=1,
            )
            _larft_finalize_kernel_mx[(B,)](
                Gpart,
                tau,
                Tout,
                rb,
                ib,
                *Gpart.stride(),
                sTauB,
                sTauN,
                sTb,
                sTm,
                sTn,
                IBN=ibn,
                num_warps=_LARFT_WARPS,
                num_stages=1,
            )
            return
        _larft_kernel_mx[(B,)](
            V,
            tau,
            Tout,
            M,
            ib,
            sVb,
            sVm,
            sVn,
            sTauB,
            sTauN,
            sTb,
            sTm,
            sTn,
            RM=_PANEL_RM,
            IBN=ibn,
            CH=_bcast_ch(ibn, ibn),
            num_warps=_LARFT_WARPS,
            # explicit pipeliner depth: defensive against the default-depth
            # miscompiles this backend shows on load-loop kernels (proven for
            # the fp32 larfb, see _launch_larfb_mx); correctness-neutral here
            num_stages=1,
        )
    else:
        _larft_kernel.fn[(B,)](
            V,
            tau,
            Tout,
            M,
            ib,
            sVb,
            sVm,
            sVn,
            sTauB,
            sTauN,
            sTb,
            sTm,
            sTn,
            RM=_PANEL_RM,
            IBN=max(16, triton.next_power_of_2(ib)),
            INVERT=True,
            num_warps=_LARFT_WARPS,
        )


def _launch_larfb_mx(V, Tp, C, m, p, ib, B, upper):
    sVb, sVm, sVn = V.stride()
    sTb, sTm, sTn = Tp.stride()
    sCb, sCm, sCn = C.stride()
    # fp32 trailing tile width: TN=16 wins on narrow trailing updates (r-mode
    # small squares: 1024^2 r 10.1 -> 9.6 ms) but halves per-CTA throughput on
    # wide ones (4096^2 reduced regresses 111 -> 121 ms), so gate on p.
    tn = 32
    if V.element_size() == 4:
        tn = _LARFB_TN_SMALL if p <= _LARFB_TN_SMALL_MAX_P else _LARFB_TN
    grid_p = (p + tn - 1) // tn
    if V.element_size() == 8:
        ibn = max(16, triton.next_power_of_2(ib))
        if m > _PAR_GRAN:
            # tall panel: row-parallel W1 partials + one solve per p-tile
            # + row-parallel apply (see above)
            rb = (m + _PAR_GRAN - 1) // _PAR_GRAN
            Wpart = torch.empty(B, rb, grid_p, ibn, tn, dtype=V.dtype, device=V.device)
            sWb, sWr, sWp, sWm, sWn = Wpart.stride()
            Ybuf = torch.empty(B, grid_p, ibn, tn, dtype=V.dtype, device=V.device)
            sYb, sYp, sYm, sYn = Ybuf.stride()
            _larfb_w1_partial_kernel_mx[(B, rb, grid_p)](
                V,
                C,
                Wpart,
                m,
                ib,
                p,
                sVb,
                sVm,
                sVn,
                sCb,
                sCm,
                sCn,
                sWb,
                sWr,
                sWp,
                sWm,
                sWn,
                GRAN=_PAR_GRAN,
                IBN=ibn,
                TN=tn,
                CH=_bcast_ch(ibn, tn),
                num_warps=_LARFB_WARPS,
                num_stages=1,
            )
            _larfb_solve_kernel_mx[(B, grid_p)](
                Tp,
                Wpart,
                Ybuf,
                rb,
                ib,
                p,
                sTb,
                sTm,
                sTn,
                sWb,
                sWr,
                sWp,
                sWm,
                sWn,
                sYb,
                sYp,
                sYm,
                sYn,
                IBN=ibn,
                TN=tn,
                UPPER=upper,
                num_warps=_LARFB_WARPS,
                num_stages=1,
            )
            _larfb_apply_kernel_mx[(B, rb, grid_p)](
                V,
                Ybuf,
                C,
                m,
                ib,
                p,
                sVb,
                sVm,
                sVn,
                sYb,
                sYp,
                sYm,
                sYn,
                sCb,
                sCm,
                sCn,
                GRAN=_PAR_GRAN,
                IBN=ibn,
                TN=tn,
                CH=_bcast_ch(ibn, tn),
                num_warps=_LARFB_WARPS,
                num_stages=1,
            )
            return
        _larfb_kernel_mx[(B, grid_p)](
            V,
            Tp,
            C,
            m,
            ib,
            p,
            sVb,
            sVm,
            sVn,
            sTb,
            sTm,
            sTn,
            sCb,
            sCm,
            sCn,
            RM=_LARFB_RM,
            IBN=ibn,
            TN=tn,
            UPPER=upper,
            SOLVE=True,
            CH=_bcast_ch(ibn, tn),
            num_warps=_LARFB_WARPS,
            num_stages=1,
        )
    else:
        _larfb_kernel[(B, grid_p)](
            V,
            Tp,
            C,
            m,
            ib,
            p,
            sVb,
            sVm,
            sVn,
            sTb,
            sTm,
            sTn,
            sCb,
            sCm,
            sCn,
            RM=_LARFB_RM,
            IBN=max(16, triton.next_power_of_2(ib)),
            TN=tn,
            UPPER=upper,
            SOLVE=False,
            num_warps=_LARFB_WARPS,
            # an explicit pipeliner depth is REQUIRED on this backend: with the
            # default num_stages the fp32 larfb miscompiles for some (M, P)
            # combinations (verified by replaying captured pipeline inputs:
            # tn=32/warps=4/default -> err 5e-2, stages=1 -> 2e-7)
            num_stages=1,
        )


def _launch_tsqr_apply(
    grid,
    V_local,
    TAU_local,
    Qt,
    Qt2,
    Q,
    Rblocks,
    Racc,
    m,
    n,
    br,
    k_max,
    grp,
    num_blocks,
    sQb,
    sQm,
    sQn,
    sRAb,
    sRAm,
    sRAn,
    BM,
    IBN,
    TWO_LEVEL,
    FOLD_TREE,
    BRMt,
    num_warps,
):
    if Q.element_size() == 8:
        # fp64: dot-free variant (quirk #1)
        _tsqr_apply_kernel_mx[grid](
            V_local,
            TAU_local,
            Qt,
            Qt2,
            Q,
            Rblocks,
            Racc,
            m,
            n,
            br,
            k_max,
            grp,
            num_blocks,
            sQb,
            sQm,
            sQn,
            sRAb,
            sRAm,
            sRAn,
            BM=BM,
            IBN=IBN,
            TWO_LEVEL=TWO_LEVEL,
            FOLD_TREE=FOLD_TREE,
            BRMt=BRMt,
            NO_DOT=True,
            num_warps=num_warps,
        )
    else:
        # fp32: the generic kernel is verified correct here
        _tsqr_apply_kernel[grid](
            V_local,
            TAU_local,
            Qt,
            Qt2,
            Q,
            Rblocks,
            Racc,
            m,
            n,
            br,
            k_max,
            grp,
            num_blocks,
            sQb,
            sQm,
            sQn,
            sRAb,
            sRAm,
            sRAn,
            BM=BM,
            IBN=IBN,
            TWO_LEVEL=TWO_LEVEL,
            FOLD_TREE=FOLD_TREE,
            BRMt=BRMt,
            num_warps=num_warps,
        )


# ---------------------------------------------------------------------------
# metax-private tile caps (quirk #3).  Unlike H20 (228 KB shared/CTA), this
# backend has 64 KB shared and ~128 KB of private (register) tile budget per
# CTA, and the two limits bite DIFFERENT kernels (measured on hardware):
#   * register-resident kernels (geqrt_sram, tsqr local/tree, qr_fused) never
#     touch shared meaningfully -- their real bound is the private budget
#     (BM*IBN*itemsize <= ~128 KB, numerically fine even at BM=1024).  But on
#     this backend a BM=512 register-resident panel is ~1.5-2x SLOWER than the
#     multi-CTA factorisation (spill), so the cap stays small: 256 (measured
#     sweet spot; 64/128/256 tie, 512 clearly loses).
#   * the TSQR budgets (fp32 apply shared-bound to BM*IBN <= 8192; fp64
#     register-bound) are decoupled per dtype in _tsqr_mx below -- the single
#     generic constant cannot express both.
# Wider fp64 panels (ib=64 vs the generic 32) halve the panel count -- and
# with it the serial factor->T->trailing chain -- for a measured 7-15% win on
# fp64 squares and wide-blocked inputs on this backend.  fp32 keeps the
# generic 32 (its dot path is fast enough that wider tiles just spill).
# ---------------------------------------------------------------------------
_GEQRT_SRAM_MAX_M = 256
_PANEL_IB_FP64 = 64
# r-mode-only fp32 panel width: with no Q to assemble, the cost is the serial
# per-panel geqrt chain, whose per-reflector tile reductions scale with the
# panel width.  ib=16 measured ~16-21% faster than 32 on mid-size r-mode
# squares (256^2/512^2/1024^2); it LOSES on 2048^2+ and on wide small-k
# shapes, and for modes that assemble Q it would DOUBLE the number of
# stream-ordered assembly sweeps (4096^2 fp32 109->140 ms) -- hence the
# narrow gate below.
_PANEL_IB_R_FP32 = 16


def _r_mode_ib(esz, m, n, k, mode):
    """Panel width for the blocked factorisation (None = dtype default)."""
    if mode == "r" and esz == 4 and 128 <= k <= 1024 and n <= 1024:
        return _PANEL_IB_R_FP32
    return None


def _blocked_qr_mx(W, V, tau, Tbuf, m, n, k, ib=None):
    """In-place blocked Householder QR; leaves R in the upper triangle of W.

    Same panel loop as the generic ``_blocked_qr``, but the compact-WY
    build/apply go through the metax launch wrappers (fp64 dot-free kernels;
    fp32 larfb at a pinned num_stages=1) and the SRAM cap uses the
    metax-calibrated value.  fp64 panels are widened to ib=64 (see above);
    the factorisation and Q assembly below stay in sync on the panel width.
    """
    if W.element_size() == 8:
        ib = _PANEL_IB_FP64
    elif ib is None:
        ib = _PANEL_IB
    B = W.shape[0]
    P = (k + ib - 1) // ib
    dt = W.dtype
    dev = W.device
    # The SRAM kernel keeps both the A tile and V_panel live; fp64 doubles
    # register pressure and never wins on this backend -> disable it there.
    sram_max_m = _GEQRT_SRAM_MAX_M if W.element_size() == 4 else 0
    # the SRAM kernel keeps a (BM, IBN) register tile: wider panels (ib=64)
    # double its size, so halve the row cap to stay out of the spill regime.
    # NOTE: the cap is pinned at _GEQRT_SRAM_MAX_M for ib < _PANEL_IB on
    # purpose -- BM=512 single-CTA panels measure 1.5-2x SLOWER than the
    # multi-CTA path even at half the tile bytes (512x16 fp32), so a
    # tile-byte-based cap was tried and reverted.
    sram_max_m = sram_max_m * _PANEL_IB // max(ib, _PANEL_IB)
    # multi-CTA scratch (one slot per (panel, column), used once -> zeroed once);
    # allocated only when at least one panel actually takes the multi-CTA path.
    needs_mcta = any(
        triton.next_power_of_2(m - kk) > sram_max_m for kk in range(0, k, ib)
    )
    if needs_mcta:
        alpha_buf = torch.zeros(B, P, ib, dtype=dt, device=dev)
        xnorm_buf = torch.zeros(B, P, ib, dtype=dt, device=dev)
        w_sum = torch.zeros(B, P, ib, ib, dtype=dt, device=dev)
        rowj_buf = torch.zeros(B, P, ib, ib, dtype=dt, device=dev)
        ctr = torch.zeros(B, P, ib, dtype=torch.int32, device=dev)
    # Hoisted stride tuples + as_strided views: plain getitem slicing costs
    # ~4us per view on this host CPU and the loop below runs k/ib panels per
    # call, which is the CPU-bound regime for small/mid squares.
    sW = W.stride()
    sV = V.stride()
    sTau = tau.stride()
    sTb_, sTm_, sTn_ = Tbuf.stride()
    soW, soV, soTau, soTb = (t.storage_offset() for t in (W, V, tau, Tbuf))
    for kk in range(0, k, ib):
        ib_active = min(ib, k - kk)
        M = m - kk
        bm = triton.next_power_of_2(M)
        fused_t = False
        if _panel_has_fused_t(W.element_size(), m, kk, ib):
            # SRAM-resident factorisation with the WY factor T built in-kernel
            # (fp32): saves one larft launch + one global V re-read per panel.
            _launch_geqrt_sram_t_mx(W, V, tau, Tbuf, m, n, k, kk, ib_active, B)
            fused_t = True
        elif bm <= sram_max_m:
            # panel fits SRAM: single-CTA resident factorisation (no global re-reads)
            _launch_geqrt_sram(W, V, tau, m, k, kk, ib_active, B)
        else:
            # ceil(M/rm) CTAs -> CHUNK == rm -> the register-resident
            # fast path of _geqrt_mcta_kernel (each CTA loads its row chunk
            # once, no per-reflector global re-reads).
            rm = _MCTA_RM if W.element_size() == 4 else _MCTA_RM_FP64
            if W.element_size() == 4:
                if M <= _MCTA_SMALL_M:
                    rm = _MCTA_RM_SMALL
            elif M > _MCTA_FP64_BIG_M:
                rm = _MCTA_RM_FP64_BIG
            nc = max(1, min(_MCTA_NC_MAX, (M + rm - 1) // rm))
            if nc >= _MCTA_MIN_NC:
                _launch_geqrt_mcta_mx(
                    W,
                    V,
                    tau,
                    alpha_buf,
                    xnorm_buf,
                    w_sum,
                    rowj_buf,
                    ctr,
                    m,
                    k,
                    kk,
                    ib_active,
                    kk // ib,
                    nc,
                    B,
                    rm=rm,
                )
            else:
                _launch_geqrt(W, V, tau, m, k, kk, ib_active, B)
        Vp = torch.as_strided(V, (B, M, ib_active), sV, soV + kk * sV[1] + kk * sV[2])
        taup = torch.as_strided(tau, (B, ib_active), sTau, soTau + kk * sTau[1])
        Tp = torch.as_strided(
            Tbuf,
            (B, ib_active, ib_active),
            (sTb_, sTm_, sTn_),
            soTb + kk * sTm_ + kk * sTn_,
        )
        if kk + ib_active < n:
            if not fused_t:
                _launch_larft_mx(Vp, taup, Tp, m, kk, ib_active, B)
            C = torch.as_strided(
                W,
                (B, M, n - (kk + ib_active)),
                sW,
                soW + kk * sW[1] + (kk + ib_active) * sW[2],
            )
            _launch_larfb_mx(
                Vp, Tp, C, m - kk, n - (kk + ib_active), ib_active, B, upper=False
            )


# ---------------------------------------------------------------------------
# metax-private TSQR orchestration.  Same algorithm / phases as the generic
# _tsqr, but the register/shared budgets are decoupled per dtype (the generic
# code derives both from one fp32 constant divided by 4 for fp64, which on
# this backend cripples the fp64 routing):
#
#   * fp64 local/apply tiles are REGISTER-bound and tolerate spill (measured
#     correct up to 512x64), so the row block can be 128 rows -- halving the
#     block count vs the generic fp64 routing.  The generic budget (2048
#     elements after the /4) gives br = 65 at n = 64, i.e. 2x the blocks and
#     a tree stack that no longer fits the two-level budget -> every tall
#     fp64 input fell back to the (slow) blocked path.
#   * fp64 tree tiles are hard-limited to 32768 elements by private memory
#     (1024x64 fails to launch), fp32 to 65536.
#   * fp32 row blocks stay bounded by the APPLY kernel's shared memory
#     (BM*IBN <= 8192), as in the generic code.
# ---------------------------------------------------------------------------
_TSQR_MX_BR_FP64 = 128  # fp64 row-block cap (swept: 64/128/256 -> 128 wins
#                           #  on every tall-skinny fp64 shape; 256's spill
#                           #  costs more than the halved block count saves)
_TSQR_MX_LOCAL_FP64 = 16384  # fp64 local-tile element budget
_TSQR_MX_RED_FP64 = 32768  # fp64 tree-tile element budget (private mem wall)
_TSQR_MX_LOCAL_FP32 = 8192  # fp32: bounded by the apply kernel's shared mem
_TSQR_MX_RED_FP32 = 65536  # fp32 tree-tile budget (two-level group/top tiles)
# A flat reduction is ONE CTA factoring the whole stack serially -- profitable
# only while the stack is small; past that the two-level tree (parallel group
# reductions) wins despite the extra launch.  Measured on this backend: the
# crossover is ~16K fp32 elements (e.g. 2048x64 fp32: flat 24 ms vs two-level
# ~7 ms).  fp64 keeps the generic BRM <= _TSQR_TREE_FLAT_ROWS rule.
_TSQR_MX_FLAT_FP32 = 16384


def _tsqr_mx(W, m, n, k, mode, B, out_Q=None, out_R=None):
    """Returns (Q or None, R).  R is (B, n, n); Q (reduced) is (B, m, n)."""
    dt = W.dtype
    dev = W.device
    IBN = max(16, triton.next_power_of_2(n))
    sWb, sWm, sWn = W.stride()
    esz = W.element_size()
    if esz == 8:
        sram_elem = _TSQR_MX_LOCAL_FP64
        red_elem = _TSQR_MX_RED_FP64
        fin_br = _TSQR_MX_BR_FP64
    else:
        sram_elem = _TSQR_MX_LOCAL_FP32
        red_elem = _TSQR_MX_RED_FP32
        fin_br = _TSQR_BR
    write_Q = mode != "r"

    # Row-block size: the local kernel keeps a (pow2(br), IBN) register tile.
    # Every block must hold >= n+1 rows (a shorter block could not produce n
    # orthonormal local Q columns), so rebalance until the last block does.
    br = max(n + 1, min(m, fin_br, sram_elem // n))
    num_blocks = (m + br - 1) // br
    while num_blocks > 1 and m - (num_blocks - 1) * br < n + 1:
        num_blocks -= 1
        br = max(n + 1, (m + num_blocks - 1) // num_blocks)
    Rm = num_blocks * n

    R_blocks = torch.empty(B, num_blocks, n, n, dtype=dt, device=dev)
    Racc = out_R if out_R is not None else torch.empty(B, n, n, dtype=dt, device=dev)

    IBNt = triton.next_power_of_2(n)
    BRM = triton.next_power_of_2(Rm)
    flat = (
        BRM * IBNt <= red_elem
        and BRM * IBNt <= _TSQR_MX_FLAT_FP32
        and (esz == 4 or BRM <= _TSQR_TREE_FLAT_ROWS)
    )
    grp = num_blocks
    num_groups = 1
    if not flat:
        cap = max(1, red_elem // (IBNt * IBNt))  # group-tile budget, in blocks
        cap = 1 << (cap.bit_length() - 1)  # round down to a power of two
        g = 1
        while g * g < num_blocks:
            g <<= 1
        grp = min(g, cap)
        num_groups = (num_blocks + grp - 1) // grp
        while (
            grp < num_blocks
            and triton.next_power_of_2(num_groups * n) * IBNt > red_elem
        ):
            grp <<= 1
            num_groups = (num_blocks + grp - 1) // grp
        two_level = (
            num_groups > 1
            and grp <= cap
            and triton.next_power_of_2(num_groups * n) * IBNt <= red_elem
        )
    else:
        two_level = False
    use_tree = flat or two_level
    # fold the tree reduction into the apply kernel for small fp32 stacks
    BRMt = max(16, triton.next_power_of_2(Rm))
    fold = flat and write_Q and esz == 4 and BRMt * IBN <= _TSQR_FOLD_ELEM

    if write_Q:
        Q = out_Q if out_Q is not None else torch.empty(B, m, n, dtype=dt, device=dev)
        V_local = torch.empty(B, m, n, dtype=dt, device=dev)
        TAU_local = torch.empty(B, num_blocks, n, dtype=dt, device=dev)
        Qt = Racc if fold else torch.empty(B, Rm, n, dtype=dt, device=dev)
        Qt2 = (
            torch.empty(B, num_groups * n, n, dtype=dt, device=dev) if two_level else Qt
        )
    else:
        # dummy pointers, never read or written (STORE_V / write_Q are False)
        Q = Qt = Qt2 = V_local = TAU_local = torch.empty(B, 1, 1, dtype=dt, device=dev)

    # ---- Phase 1: local QR of all row blocks in one launch ----
    k_max = min(br, n)
    BM = triton.next_power_of_2(br)
    sVb, sVm, sVn = V_local.stride()
    _tsqr_local_sram_kernel[(B, num_blocks)](
        W,
        R_blocks,
        V_local,
        TAU_local,
        m,
        n,
        br,
        num_blocks,
        k_max,
        sWb,
        sWm,
        sWn,
        R_blocks.stride(0),
        R_blocks.stride(1),
        R_blocks.stride(2),
        sVb,
        sVm,
        sVn,
        TAU_local.stride(0),
        BM=BM,
        IBN=IBN,
        STORE_V=write_Q,
        num_warps=max(4, min(16, (BM * IBN) // 4096)),
    )

    # ---- Phase 2: tree reduction of the stacked local R factors ----
    def tree_warps(tile_rows):
        return _TSQR_TREE_WARPS if tile_rows * IBNt * esz <= 16384 else 8

    if use_tree:
        if fold:
            pass  # the apply kernel factors the stack redundantly per CTA
        elif two_level:
            Rg = torch.empty(B, num_groups, n, n, dtype=dt, device=dev)
            _tsqr_tree_kernel[(B, num_groups)](
                R_blocks,
                Rg,
                Qt,
                n,
                num_blocks,
                grp,
                write_Q,
                Rg.stride(0),
                Rg.stride(2),
                Rg.stride(3),
                BRM=grp * IBNt,
                IBN=IBNt,
                num_warps=tree_warps(grp * IBNt),
            )
            BRMt2 = triton.next_power_of_2(num_groups * n)
            _tsqr_tree_kernel[(B, 1)](
                Rg,
                Racc,
                Qt2,
                n,
                num_groups,
                num_groups,
                write_Q,
                Racc.stride(0),
                Racc.stride(1),
                Racc.stride(2),
                BRM=BRMt2,
                IBN=IBNt,
                num_warps=tree_warps(BRMt2),
            )
        else:
            # stack fits one register-resident CTA per batch: factor + build Q_t
            _tsqr_tree_kernel[(B, 1)](
                R_blocks,
                Racc,
                Qt,
                n,
                num_blocks,
                num_blocks,
                write_Q,
                Racc.stride(0),
                Racc.stride(1),
                Racc.stride(2),
                BRM=BRM,
                IBN=IBNt,
                num_warps=tree_warps(BRM),
            )
    else:
        # Stack too tall even for a two-level tree: factor it with the blocked
        # path and assemble Q_t from its reflectors (the metax launch wrappers
        # and stream-ordered assembly are used automatically).
        Rstack = R_blocks.reshape(B, Rm, n)
        Vt = (
            torch.zeros(B, Rm, n, dtype=dt, device=dev)
            if write_Q
            else torch.empty(B, Rm, n, dtype=dt, device=dev)
        )
        tau_t = torch.empty(B, n, dtype=dt, device=dev)
        Tbuf_t = torch.empty(B, n, n, dtype=dt, device=dev)
        _blocked_qr_mx(Rstack, Vt, tau_t, Tbuf_t, Rm, n, n)
        _triu_copy(Rstack, Racc, n, n, B)
        if write_Q:
            _assemble_q_mx(Vt, tau_t, Tbuf_t, Rm, n, n, n, _PANEL_IB, B, Qt)
        two_level = False

    if not write_Q:
        return None, Racc

    # ---- Phase 3: Q[block rows] = Q_local @ Q_t[block rows], one CTA/block ----
    _launch_tsqr_apply(
        (B, num_blocks),
        V_local,
        TAU_local,
        Qt,
        Qt2,
        Q,
        R_blocks,
        Racc,
        m,
        n,
        br,
        k_max,
        grp,
        num_blocks,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        Racc.stride(0),
        Racc.stride(1),
        Racc.stride(2),
        BM=BM,
        IBN=IBN,
        TWO_LEVEL=two_level,
        FOLD_TREE=fold,
        BRMt=BRMt if fold else 16,
        num_warps=max(4, min(16, (BM * IBN) // 4096)),
    )
    return Q, Racc


def _assemble_q_mx(V, tau, Tbuf, m, n, k, qcols, ib, B, out):
    """Stream-ordered Q assembly (quirk #2).

    identity kernel + one larfb launch per panel (reverse order), so every
    cross-panel dependency is ordered by the stream instead of an intra-kernel
    global write->read.  Uses the metax larft/larfb wrappers, so fp64 also
    gets the dot-free kernels.  The panel width is overridden for fp64 so it
    always matches _blocked_qr_mx's factorisation.
    """
    if V.element_size() == 8:
        ib = _PANEL_IB_FP64
    P = (k + ib - 1) // ib
    kk_last = (P - 1) * ib
    ib_last = min(ib, k - kk_last)
    if kk_last + ib_last >= n and not _panel_has_fused_t(
        V.element_size(), m, kk_last, ib
    ):
        # the last panel never got a T from _blocked_qr_mx (no trailing update)
        # unless the T-fused SRAM kernel already wrote it
        Vp = V[:, kk_last:m, kk_last : kk_last + ib_last]
        taup = tau[:, kk_last : kk_last + ib_last]
        Tp = Tbuf[:, kk_last : kk_last + ib_last, kk_last : kk_last + ib_last]
        _launch_larft_mx(Vp, taup, Tp, m, kk_last, ib_last, B)
    sQb, sQm, sQn = out.stride()
    if P == 1 and (V.element_size() == 4 or (k <= _ASSEMBLE_1P_MAX_K_FP64 and m >= n)):
        # single panel: Q = I - V T V^H in one pass -- W1 = T V[p,:]^H read
        # straight off V's rows (no identity write + no Q re-reads).  fp32
        # uses the generic kernel (fp32 tl.dot is exact here); fp64 uses the
        # dot-free row-parallel metax variant for tall/square shapes -- for
        # wide ones (m < n, tiny square Q) the stream-ordered larfb is as
        # fast and the redundant per-CTA W1 solve just adds launches.
        Vp = V[:, :, :k]
        Tp = Tbuf[:, :k, :k]
        sVb, sVm, sVn = Vp.stride()
        sTb, sTm, sTn = Tp.stride()
        ibn = max(16, triton.next_power_of_2(k))
        grid_p = (qcols + _ASSEMBLE_TN - 1) // _ASSEMBLE_TN
        if V.element_size() == 4:
            _assemble_q_single_panel_kernel[(B, grid_p)](
                Vp,
                Tp,
                out,
                m,
                k,
                qcols,
                sVb,
                sVm,
                sVn,
                sTb,
                sTm,
                sTn,
                sQb,
                sQm,
                sQn,
                RM=_ASSEMBLE_RM,
                TN=_ASSEMBLE_TN,
                IBN=ibn,
                num_warps=_ASSEMBLE_WARPS,
            )
        else:
            grid_r = (m + _ASSEMBLE_RM - 1) // _ASSEMBLE_RM
            _assemble_q_single_panel_kernel_mx[(B, grid_r, grid_p)](
                Vp,
                Tp,
                out,
                m,
                k,
                qcols,
                sVb,
                sVm,
                sVn,
                sTb,
                sTm,
                sTn,
                sQb,
                sQm,
                sQn,
                RM=_ASSEMBLE_RM,
                TN=_ASSEMBLE_TN,
                IBN=ibn,
                num_warps=_ASSEMBLE_WARPS,
            )
        return out
    grid_e = (m * qcols + 1023) // 1024
    _identity_kernel[(B * grid_e,)](out, m, qcols, grid_e, sQb, sQm, sQn, BLOCK=1024)
    for kk in reversed(range(0, k, ib)):
        ib_active = min(ib, k - kk)
        Vp = V[:, kk:m, kk : kk + ib_active]
        Tp = Tbuf[:, kk : kk + ib_active, kk : kk + ib_active]
        _launch_larfb_mx(
            Vp, Tp, out[:, kk:m, :], m - kk, qcols, ib_active, B, upper=True
        )
    return out


# fp64 single-panel one-pass assembly gate (see _assemble_q_mx): row-parallel
# one-pass wins for tall/square panels up to k=64 on this backend.
_ASSEMBLE_1P_MAX_K_FP64 = 64


# ===========================================================================
# Public op.  Mirrors the generic _linalg_qr routing with the metax variants
# plugged in (fused path and helpers are reused unchanged).
# ===========================================================================
def _linalg_qr_mx(A, mode="reduced", *, out=None):
    _validate_mode(mode)

    if A.dim() < 2:
        raise RuntimeError("linalg_qr: input must have at least 2 dimensions")
    if A.dtype not in (torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems linalg_qr currently supports float32 and float64 inputs; "
            f"got dtype={A.dtype}"
        )

    orig_dtype = A.dtype
    batch_shape = A.shape[:-2]
    m, n = A.shape[-2], A.shape[-1]
    k = min(m, n)
    B = 1
    for d in batch_shape:
        B *= d

    if out is not None:
        _validate_out(out, A.dtype, batch_shape, m, n, mode)

    if m == 0 or n == 0:
        # Degenerate input: no factorisation to run.  torch.linalg.qr returns
        # empty factors, except complete mode with zero columns where Q = I.
        if mode == "r":
            q_shape, r_shape = (0,), (*batch_shape, k, n)
        elif mode == "reduced":
            q_shape, r_shape = (*batch_shape, m, k), (*batch_shape, k, n)
        else:
            q_shape, r_shape = (*batch_shape, m, m), (*batch_shape, m, n)
        if out is not None:
            Q, R = out
        else:
            Q = A.new_empty(q_shape)
            R = A.new_empty(r_shape)
        if mode == "complete" and n == 0 and m > 0:
            eye = torch.eye(m, dtype=A.dtype, device=A.device)
            Q.copy_(eye.expand(*batch_shape, m, m))
        return Q, R

    # Read-only view of A.  The fused and TSQR paths only read W (no copy
    # needed); the blocked path factors in place and clones below.
    W = A.reshape(B, m, n)

    qcols = 0 if mode == "r" else (k if mode == "reduced" else m)
    rrows = k if mode in ("reduced", "r") else m
    # Resolve caller-provided output buffers (reshaped to the (B, ...) layout the
    # kernels write).  These are views of the user's tensors for the contiguous
    # case, so the kernels write the user's memory directly -- no alloc/copy.
    out_Q = out_R = None
    if out is not None:
        out_Q, out_R = out
        out_Q = out_Q.reshape(B, m, qcols) if qcols else out_Q.reshape(0)
        out_R = out_R.reshape(B, rrows, n)
    # TSQR only for tall-skinny that fused can't fit (m*n > _FUSED_ELEM or m > _FUSED_M);
    # smaller tall-skinny (e.g. 64×16, 128×32) are faster via the single-launch fused kernel.
    # TSQR also needs m >= _TSQR_MIN_M -- below that the blocked path's zero-sync SRAM
    # panels beat TSQR's per-block serial reflector chains (e.g. 512x64, 256x64).
    is_ts = (
        (m >= _TSQR_ASPECT * n)
        and (m >= _TSQR_MIN_M)
        and (n > 0)
        and (n <= _TSQR_MAX_N)
        and (mode in ("reduced", "r"))
    )
    # _FUSED_ELEM is calibrated for fp32; fp64 doubles register pressure in the
    # fused kernel (A tile + Q tile both live during Q assembly), so larger fp64
    # matrices spill and run ~2x slower than the blocked path (e.g. 64x64 fp64:
    # fused 1120us vs blocked 581us).  Cut the cap to ~1/4 for fp64 -- small /
    # batched matrices stay fused (single launch wins), only the larger singles
    # divert to blocked.
    fused_elem = _FUSED_ELEM if A.element_size() == 4 else _FUSED_ELEM // 4
    fits_fused = (
        m <= _FUSED_M
        and n <= _FUSED_DIM
        and m * n <= fused_elem
        and (mode == "r" or qcols * m <= fused_elem)
    )
    if is_ts:
        fits_fused = fits_fused and (m <= _FUSED_TALL_M)
    if fits_fused:
        return _fused_qr(W, A, orig_dtype, batch_shape, m, n, k, mode, B, out_Q, out_R)

    # large matrices: TSQR for tall-skinny, blocked Householder otherwise.
    if is_ts:
        Qm, Rm = _tsqr_mx(W, m, n, k, mode, B, out_Q, out_R)
        if mode == "r":
            return (
                out_Q if out_Q is not None else A.new_empty(0),
                Rm.reshape(*batch_shape, n, n),
            )
        return (Qm.reshape(*batch_shape, m, n), Rm.reshape(*batch_shape, n, n))

    # blocked Householder path (large matrices): kernels write W in place.
    # V is zero-initialised when Q is assembled (mode != "r"): the paired
    # assembly reads the whole trapezoid of V below each pair start, including
    # entries above each reflector's diagonal that geqrt never writes --
    # mathematically zero, physically stale memory otherwise.  tau/Tbuf are
    # fully written by the geqrt/larft kernels before they are read (empty).
    V = (
        torch.zeros(B, m, k, dtype=W.dtype, device=W.device)
        if mode != "r"
        else torch.empty(B, m, k, dtype=W.dtype, device=W.device)
    )
    tau = torch.empty(B, k, dtype=W.dtype, device=W.device)
    Tbuf = torch.empty(B, k, k, dtype=W.dtype, device=W.device)
    W = W.clone()
    ib = _r_mode_ib(W.element_size(), m, n, k, mode)
    _blocked_qr_mx(W, V, tau, Tbuf, m, n, k, ib)

    if mode == "r":
        R = (
            out_R
            if out_R is not None
            else torch.empty(B, k, n, dtype=W.dtype, device=W.device)
        )
        _triu_copy(W, R, k, n, B)
        return (
            out_Q if out_Q is not None else A.new_empty(0),
            R.reshape(*batch_shape, k, n),
        )

    Q = (
        out_Q
        if out_Q is not None
        else torch.empty(B, m, qcols, dtype=W.dtype, device=W.device)
    )
    _assemble_q_mx(V, tau, Tbuf, m, n, k, qcols, _PANEL_IB, B, Q)

    R = (
        out_R
        if out_R is not None
        else torch.empty(
            B, qcols if mode == "complete" else k, n, dtype=W.dtype, device=W.device
        )
    )
    _triu_copy(W, R, R.shape[-2], n, B)

    if mode == "reduced":
        return (Q.reshape(*batch_shape, m, k), R.reshape(*batch_shape, k, n))
    return (Q.reshape(*batch_shape, m, m), R.reshape(*batch_shape, m, n))


def linalg_qr(A, mode="reduced", *, out=None):
    logger.debug("GEMS_METAX LINALG_QR")
    return _linalg_qr_mx(A, mode, out=out)


def linalg_qr_out(A, mode="reduced", *, Q, R):
    logger.debug("GEMS_METAX LINALG_QR_OUT")
    return _linalg_qr_mx(A, mode, out=(Q, R))
