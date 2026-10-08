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
"""Pure-Triton implementation of ``torch.linalg.qr`` (``aten::linalg_qr``).

Blocked Householder QR with the compact WY (Gram-solve) block representation,
following the algorithm family described in

    Michael Lutz, "QR Decomp at the Speed of Light", https://ml-mike.com/writing/qr_v2

Routing summary (see :func:`linalg_qr` for the exact conditions):

* **Fused path** (small square + short tall-skinny): a single ``_qr_fused_kernel``
  does unblocked QR + R extraction + Q assembly in one launch.
* **TSQR path** (tall-skinny): ``_tsqr_local_sram_kernel`` factors all row
  blocks concurrently (storing the local reflectors); ``_tsqr_tree_kernel``
  (or the blocked path, for very tall stacks) reduces the stacked R factors
  and builds the tree factor Q_t; ``_tsqr_apply_kernel`` forms
  Q = diag(Q_local) @ Q_t by applying the stored reflectors -- never
  A @ R^{-1} -- so rank-deficient inputs stay robust.
* **Blocked path** (large square): ``_geqrt_sram_kernel`` factors IB-wide panels
  that fit in shared memory (one CTA, no global re-reads); taller panels fall
  back to the multi-CTA ``_geqrt_mcta_kernel`` (row-split across NC CTAs).
  ``_larft_kernel`` builds the WY factor T (via the Gram-solve trick and a
  small in-kernel triangular inverse); ``_larfb_kernel`` applies the block
  reflector for the trailing update and Q assembly with plain GEMMs.

Every numerical step lives in a Triton kernel -- the python wrapper only
allocates buffers and launches kernels.

Supports the three modes ``"reduced"`` / ``"complete"`` / ``"r"``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
_PANEL_IB = 32
# Wider panels (ib=64) were tried for large square matrices: panel count halves,
# but the mcta kernel's per-CTA register tile doubles (spill regime), larfb/larft
# tiles spill at num_warps=4, and the net was a ~15% LOSS on 4096^2 (measured).
# Keep ib=32 everywhere; the launchers still size IBN = pow2(ib) adaptively.
_PANEL_RM = 64  # row tile inside the panel kernel
_MCTA_NC_MAX = 64  # max CTAs cooperating on one panel (multi-CTA path)
_MCTA_MIN_NC = 4  # multi-CTA from NC>=4 (M>=256); below it the barrier overhead per reflector dominates
# Row tile for the multi-CTA panel kernel's register-resident fast path (NC =
# ceil(M/_MCTA_RM) CTAs cooperate on one panel).  Swept on H20: 64 is optimal;
# 128/256 (fewer barrier participants) and smaller tiles are all slower -- the
# per-reflector cost is round-trip latency bound, not participant-count bound.
_MCTA_RM = 64
# fp64 doubles the register tile (2 regs/elem): RM=32 halves it and is a
# consistent ~10% win on fp64 mcta panels (measured (512,64)/(512,512)/(1024,1024)).
_MCTA_RM_FP64 = 32
# SRAM-resident panel factorisation is used while the panel tile fits shared
# memory: BM=next_pow2(M) and BM*_PANEL_IB*itemsize must stay within the SRAM
# budget.  _PANEL_IB=32, fp32 -> BM<=1024 (128 KB) is the safe cap.
_GEQRT_SRAM_MAX_M = 512
_LARFB_RM = 64  # row tile for the block-reflector apply kernel
_LARFB_TN = 32  # column tile for the block-reflector apply kernel (tuned: TN=32
#                gives ~1.9x on large trailing updates vs TN=16; TN=64 spills)
_TSQR_ASPECT = 4  # m >= _TSQR_ASPECT * n  =>  tall-skinny candidate for TSQR
# TSQR is a tall-SKINNY algorithm: the register-resident local/tree kernels
# keep (pow2(rows), pow2(n)) tiles live per reflector; for n > 64 they spill
# and lose to the blocked path, which handles wider tall matrices fine.
_TSQR_MAX_N = 64
# TSQR's flat reduction only beats the blocked path (geqrt_sram, zero-sync panels)
# once m is large enough; below this the per-block local-QR sync overhead dominates
# and the blocked path is faster (empirical crossover ~640-700 on H20).
_TSQR_MIN_M = 700
# TSQR row block: the register-resident local kernel keeps a (pow2(br), IBN)
# tile live; smaller blocks give shorter serial reflector chains and more CTAs.
# fp64's 2 regs/element halves the tile budget: br=128 measured ~2x faster than
# 512 on narrow fp64 blocks ((4096,4)/(8192,8)); fp32 stays at 512.
_TSQR_BR = 512  # row-block cap for the local QR (fp32)
_TSQR_BR_FP64 = 128
# Cap on the padded tree-reduction tile BRM*IBN (fp32 elements): one CTA per
# batch factors the stacked R's in registers when it fits; taller stacks go
# through the blocked path instead.
_TSQR_TREE_RED_ELEM = 16384
# Tall fp64 stacks are never reduced flat (single CTA): the serial per-reflector
# tile reductions are so slow in fp64 that the two-level tree wins even when the
# tile fits the register budget.  This is the fp64 flat cutoff, in padded rows.
_TSQR_TREE_FLAT_ROWS = 128
# Below this padded stack-tile size (BRMt*IBN elements, fp32 only) every apply
# CTA can afford to redundantly factor the stacked local R's in registers,
# folding the tree reduction into the apply kernel and saving one kernel
# launch.  Larger tiles lose: the redundant factorisation's register pressure
# and serial reductions outweigh the saved launch overhead.
_TSQR_FOLD_ELEM = 1024
# Max elements per TSQR row block for the register-resident single-CTA local
# QR (no atomics / cross-CTA barriers / global re-reads).
_TSQR_SRAM_ELEM = 16384
# Single-launch fused SRAM kernel: used when the matrix tile (and Q tile) fit in
# shared memory.  Covers small square matrices AND tall-skinny ones (large m,
# small n).  Caps keep the BM/BN/BQ tiles inside SRAM.
_FUSED_DIM = 128  # max columns n (BN tile)
_FUSED_M = 4096  # max rows m (BM tile) -- relevant for tall-skinny
_FUSED_TALL_M = (
    512  # for tall-skinny, fused only up to this m (single CTA serializes beyond)
)
_FUSED_ELEM = 8192  # max elements in the A tile (m*n) and Q tile (qcols*m)
# num_warps per kernel launch (module-level so they can be swept for tuning;
# Triton's launch-time default is 4).
_GEQRT_SRAM_WARPS = 4
_MCTA_WARPS = 4
_LARFT_WARPS = 4
_LARFB_WARPS = 4
_ASSEMBLE_WARPS = 4
# tile shape for the fused Q-assembly kernel (decoupled from _LARFB_RM/_LARFB_TN
# after the larfb sweep showed the two kernels have different optima)
_ASSEMBLE_RM = 64
_ASSEMBLE_TN = 32
# Paired Q assembly (fp32, P >= 2): adjacent panel pairs are composed into one
# (2*ib)-wide compact-WY factor in a single _tcompose_pair_kernel launch
# (in-place in Tbuf), then the fused kernel runs with ib=2*_PANEL_IB and half
# the panels -- halving the number of dependent Q read/write sweeps, which is
# the dominant cost of Q assembly on large matrices.
_ASSEMBLE_PAIR = True
_ASSEMBLE_PAIR_WARPS = 8
# num_stages for the load->dot pipelining in the larfb / assemble tile loops
# (None = Triton default; swept on H20).
_LARFB_STAGES = None
_ASSEMBLE_STAGES = None
_TSQR_TREE_WARPS = 4


# ===========================================================================
# Kernel 1: blocked-Householder panel factorization (unblocked QR of IB cols)
# ===========================================================================
@libentry()
@triton.jit
def _geqrt_kernel(
    W,
    V,
    TAU,
    M,
    kk,
    ib,
    nr,
    sWb,
    sWm,
    sWn,
    sVb,
    sVm,
    sVn,
    sTauB,
    sTauN,
    RM: tl.constexpr,
    IBN: tl.constexpr,
):
    pid = tle.program_id(0)
    Wb = W + pid * sWb
    Vb = V + pid * sVb
    TAUb = TAU + pid * sTauB

    dt = Wb.dtype.element_ty
    zero = tl.full((), 0.0, dtype=dt)
    one = tl.full((), 1.0, dtype=dt)

    rows_local = tl.arange(0, RM)
    col_idx = tl.arange(0, IBN)
    num_tiles = (M + RM - 1) // RM

    # tau and the R diagonal accumulate in registers and are flushed with one
    # vector store each after the loop -- 0-d scalar stores are unreliable on
    # some vendor backends
    tau_arr = tl.zeros([IBN], dtype=dt)
    rdg = tl.zeros([IBN], dtype=dt)
    for j in range(nr):
        # ---- pass A: pivot alpha + tail norm --------------------------------
        alpha = zero
        xnorm_sq = zero
        for t in range(num_tiles):
            local = t * RM + rows_local
            rows_g = kk + local
            rmask = local < M
            col = tl.load(Wb + rows_g * sWm + (kk + j) * sWn, mask=rmask, other=zero)
            alpha += tl.sum(tl.where(rmask & (local == j), col, zero))
            xnorm_sq += tl.sum(tl.where(rmask & (local > j), col * col, zero))

        # ---- dlarfg ---------------------------------------------------------
        norm = tl.sqrt(alpha * alpha + xnorm_sq)
        beta = tl.where(alpha >= zero, -norm, norm)
        reflect = xnorm_sq > zero
        beta_eff = tl.where(reflect, beta, alpha)
        tau = tl.where(reflect, (beta - alpha) / beta, zero)
        denom = alpha - beta
        # guard 0/0: without a reflection denom may be 0; keep the tail
        # quotient finite before the reflect mask discards it
        denom_safe = tl.where(reflect, denom, one)

        tau_arr = tl.where(col_idx == j, tau, tau_arr)
        rdg = tl.where(col_idx == j, beta_eff, rdg)

        # ---- pass B: store Householder vector (explicit unit lower) ---------
        for t in range(num_tiles):
            local = t * RM + rows_local
            rows_g = kk + local
            rmask = local < M
            col = tl.load(Wb + rows_g * sWm + (kk + j) * sWn, mask=rmask, other=zero)
            v_tail = col / denom_safe
            v = tl.where(local > j, v_tail, tl.where(local == j, one, zero))
            v = tl.where(reflect, v, tl.where(local == j, one, zero))
            tl.store(Vb + rows_g * sVm + (kk + j) * sVn, v, mask=rmask)
        tl.debug_barrier()

        # ---- pass C: w = tau * (v^H W[:, trailing]) -------------------------
        w = tl.zeros([IBN], dtype=dt)
        for t in range(num_tiles):
            local = t * RM + rows_local
            rows_g = kk + local
            rmask = local < M
            v = tl.load(Vb + rows_g * sVm + (kk + j) * sVn, mask=rmask, other=zero)
            wblock_off = rows_g[:, None] * sWm + (kk + col_idx)[None, :] * sWn
            cmask = rmask[:, None] & (col_idx[None, :] < ib)
            Wt = tl.load(Wb + wblock_off, mask=cmask, other=zero)
            w += tl.sum(v[:, None] * Wt, axis=0)
        w = tau * w
        w = tl.where((col_idx > j) & (col_idx < ib), w, zero)

        # ---- pass D: apply W[:, trailing] -= v * w --------------------------
        for t in range(num_tiles):
            local = t * RM + rows_local
            rows_g = kk + local
            rmask = local < M
            v = tl.load(Vb + rows_g * sVm + (kk + j) * sVn, mask=rmask, other=zero)
            wblock_off = rows_g[:, None] * sWm + (kk + col_idx)[None, :] * sWn
            cmask = rmask[:, None] & (col_idx[None, :] < ib)
            Wt = tl.load(Wb + wblock_off, mask=cmask, other=zero)
            upd = v[:, None] * w[None, :]
            upd = tl.where((col_idx[None, :] > j) & (col_idx[None, :] < ib), upd, zero)
            Wt = Wt - upd
            smask = rmask[:, None] & (col_idx[None, :] > j) & (col_idx[None, :] < ib)
            tl.store(Wb + wblock_off, Wt, mask=smask)
        tl.debug_barrier()

    # flush tau and the R diagonal with one vector store each
    tl.store(TAUb + (kk + col_idx) * sTauN, tau_arr, mask=col_idx < nr)
    tl.store(Wb + (kk + col_idx) * sWm + (kk + col_idx) * sWn, rdg, mask=col_idx < nr)


# ===========================================================================
# Kernel 2: multi-CTA panel factorisation (row-split across NC CTAs).
# For tall panels the single-CTA geqrt leaves most SMs idle because the
# reflector chain is serial.  Here the panel rows are split across NC CTAs;
# each CTA owns a contiguous row range, so the only cross-CTA coordination is
# the per-column reductions (alpha/norm and the unscaled w partials, both
# accumulated before ONE spinning-counter barrier per reflector).
# ===========================================================================
@triton.jit
def _barrier(ctr, off, NC):
    # release: prior partial-sum atomics are visible before the count lands.
    tl.atomic_add(ctr + off, 1, sem="release")
    # Spin read via an atomic with the DEFAULT memory order (acq_rel).  The
    # two alternatives are both broken on some vendor backend: a volatile
    # load spin never observes the count on hygon, and an atomic with an
    # explicit sem= kwarg in the spin hangs the thead PPU fork.  acq_rel is
    # a superset of the acquire ordering the barrier-protected loads need.
    while tl.atomic_add(ctr + off, 0) < NC:
        pass


@libentry()
@triton.jit
def _geqrt_mcta_kernel(
    W,
    V,
    TAU,
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
    CHUNK: tl.constexpr,
    RM: tl.constexpr,
    IBN: tl.constexpr,
    NC: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    pid_b = tle.program_id(0)
    c = tle.program_id(1)
    Wb = W + pid_b * sWb
    Vb = V + pid_b * sVb
    TAUb = TAU + pid_b * sTauB
    dt = Wb.dtype.element_ty
    zero = tl.full((), 0.0, dtype=dt)
    one = tl.full((), 1.0, dtype=dt)
    rows_local = tl.arange(0, RM)
    col_idx = tl.arange(0, IBN)
    row_lo = c * CHUNK

    if NUM_TILES == 1:
        # Fast path: each CTA owns exactly one row-tile (CHUNK==RM).  Load the
        # panel chunk ONCE into registers and sustain it across the whole
        # reflector loop -- col_j is extracted from it, the trailing update is
        # applied in place, and only V/tau are written per reflector.  This is
        # the multi-CTA analogue of _geqrt_sram_kernel: row parallelism (NC CTAs)
        # with zero per-reflector global reload of the panel.  Registers survive
        # the cross-CTA barriers, so the sustained chunk stays live.
        lr = rows_local
        plr = row_lo + lr
        gr = kk + plr
        rmask = (plr < M) & (plr < row_lo + CHUNK)
        cmask = col_idx < ib
        Wblk = tl.load(
            Wb + gr[:, None] * sWm + (kk + col_idx)[None, :] * sWn,
            mask=rmask[:, None] & cmask[None, :],
            other=zero,
        )
        # tau accumulates in registers, flushed with ONE vector store after the
        # loop (0-d scalar stores: see _geqrt_kernel)
        tau_arr = tl.zeros([IBN], dtype=dt)
        for j in range(nr):
            # column j from the sustained panel chunk
            col_j = tl.sum(tl.where(col_idx[None, :] == j, Wblk, zero), axis=1)
            alpha_c = tl.sum(tl.where(rmask & (plr == j), col_j, zero))
            xnorm_c = tl.sum(tl.where(rmask & (plr > j), col_j * col_j, zero))
            tl.atomic_add(alpha_buf + pid_b * sAB + p * sAM + j, alpha_c)
            tl.atomic_add(xnorm_buf + pid_b * sXB + p * sXM + j, xnorm_c)
            # Unscaled partials for w = tau * v^H A, computed BEFORE the barrier:
            # v's tail is col_j/denom with an implicit 1 at row j, and tau/denom
            # are scalars known only after the barrier, so accumulate
            #   wraw = sum_{i>j} col_j[i] * A[i, trailing]
            #   rowj = A[j, trailing]            (owning CTA only)
            # here and apply the scalar fix-up after the barrier.  This folds the
            # two cross-CTA barriers per reflector into one (~2x on this kernel).
            wmask = (col_idx[None, :] > j) & cmask[None, :]
            tail_v = tl.where(plr > j, col_j, zero)
            wraw_c = tl.sum(tl.where(wmask, tail_v[:, None] * Wblk, zero), axis=0)
            tl.atomic_add(
                w_sum + pid_b * sWB + p * sWM + j * sWN + col_idx,
                wraw_c,
                mask=col_idx < ib,
            )
            head = tl.where(plr == j, one, zero)
            rowj_c = tl.sum(tl.where(wmask, head[:, None] * Wblk, zero), axis=0)
            tl.atomic_add(
                rowj_buf + pid_b * sRB + p * sRM2 + j * sRN2 + col_idx,
                rowj_c,
                mask=col_idx < ib,
            )
            _barrier(ctr, pid_b * sCtrB + p * sCtrM + j, NC)
            alpha = tl.load(alpha_buf + pid_b * sAB + p * sAM + j, volatile=True)
            xnorm_sq = tl.load(xnorm_buf + pid_b * sXB + p * sXM + j, volatile=True)
            wraw = tl.load(
                w_sum + pid_b * sWB + p * sWM + j * sWN + col_idx,
                mask=col_idx < ib,
                other=zero,
                volatile=True,
            )
            rowj = tl.load(
                rowj_buf + pid_b * sRB + p * sRM2 + j * sRN2 + col_idx,
                mask=col_idx < ib,
                other=zero,
                volatile=True,
            )

            norm = tl.sqrt(alpha * alpha + xnorm_sq)
            beta = tl.where(alpha >= zero, -norm, norm)
            reflect = xnorm_sq > zero
            beta_eff = tl.where(reflect, beta, alpha)
            tau = tl.where(reflect, (beta - alpha) / beta, zero)
            denom = alpha - beta
            # guard 0/0: without a reflection denom may be 0 and the tail /
            # fix-up quotients would go non-finite before the reflect mask
            # discards them -- keep them finite on any FP backend
            denom_safe = tl.where(reflect, denom, one)
            tau_arr = tl.where(col_idx == j, tau, tau_arr)
            # V from col_j -> global V buffer
            v_tail = col_j / denom_safe
            v = tl.where(plr > j, v_tail, tl.where(plr == j, one, zero))
            v = tl.where(reflect, v, tl.where(plr == j, one, zero))
            tl.store(Vb + gr * sVm + (kk + j) * sVn, v, mask=rmask)
            # write R diagonal into the sustained chunk (owning CTA only)
            diag = (lr[:, None] == (j - row_lo)) & (col_idx[None, :] == j)
            Wblk = tl.where(diag, beta_eff, Wblk)
            # scalar fix-up of the pre-barrier partials (denom_safe != 0 always)
            w = tl.where(reflect, tau * (rowj + wraw / denom_safe), zero)
            # in-place trailing update of the sustained chunk
            upd = v[:, None] * w[None, :]
            upd = tl.where(wmask, upd, zero)
            Wblk = Wblk - upd
        # flush the sustained panel chunk once (upper triangle now holds R)
        tl.store(
            Wb + gr[:, None] * sWm + (kk + col_idx)[None, :] * sWn,
            Wblk,
            mask=rmask[:, None] & cmask[None, :],
        )
        # flush tau with one vector store
        tl.store(TAUb + (kk + col_idx) * sTauN, tau_arr, mask=col_idx < nr)
    else:
        # General path (CHUNK > RM, very tall panels): tile loop per pass.
        # Same single-barrier scheme as the fast path: accumulate the unscaled
        # w partials (wraw / rowj) together with alpha/xnorm in ONE tile loop,
        # barrier once, then fix up with the tau/denom scalars.
        # tau and the R diagonal accumulate in registers, flushed with one
        # vector store each after the loop (0-d scalar stores: see _geqrt_kernel)
        tau_arr = tl.zeros([IBN], dtype=dt)
        rdg = tl.zeros([IBN], dtype=dt)
        for j in range(nr):
            alpha_c = zero
            xnorm_c = zero
            wraw_c = tl.zeros([IBN], dtype=dt)
            rowj_c = tl.zeros([IBN], dtype=dt)
            wmask = (col_idx > j) & (col_idx < ib)
            for t in range(NUM_TILES):
                lr = t * RM + rows_local
                plr = row_lo + lr
                gr = kk + plr
                rmask = (plr < M) & (plr < row_lo + CHUNK)
                wblk = tl.load(
                    Wb + gr[:, None] * sWm + (kk + col_idx)[None, :] * sWn,
                    mask=rmask[:, None] & (col_idx[None, :] < ib),
                    other=zero,
                )
                col = tl.sum(tl.where(col_idx[None, :] == j, wblk, zero), axis=1)
                alpha_c += tl.sum(tl.where(rmask & (plr == j), col, zero))
                xnorm_c += tl.sum(tl.where(rmask & (plr > j), col * col, zero))
                tail_v = tl.where(plr > j, col, zero)
                wraw_c += tl.sum(
                    tl.where(wmask[None, :], tail_v[:, None] * wblk, zero), axis=0
                )
                head = tl.where(plr == j, one, zero)
                rowj_c += tl.sum(
                    tl.where(wmask[None, :], head[:, None] * wblk, zero), axis=0
                )
            tl.atomic_add(alpha_buf + pid_b * sAB + p * sAM + j, alpha_c)
            tl.atomic_add(xnorm_buf + pid_b * sXB + p * sXM + j, xnorm_c)
            tl.atomic_add(
                w_sum + pid_b * sWB + p * sWM + j * sWN + col_idx,
                wraw_c,
                mask=col_idx < ib,
            )
            tl.atomic_add(
                rowj_buf + pid_b * sRB + p * sRM2 + j * sRN2 + col_idx,
                rowj_c,
                mask=col_idx < ib,
            )
            _barrier(ctr, pid_b * sCtrB + p * sCtrM + j, NC)
            alpha = tl.load(alpha_buf + pid_b * sAB + p * sAM + j, volatile=True)
            xnorm_sq = tl.load(xnorm_buf + pid_b * sXB + p * sXM + j, volatile=True)
            wraw = tl.load(
                w_sum + pid_b * sWB + p * sWM + j * sWN + col_idx,
                mask=col_idx < ib,
                other=zero,
                volatile=True,
            )
            rowj = tl.load(
                rowj_buf + pid_b * sRB + p * sRM2 + j * sRN2 + col_idx,
                mask=col_idx < ib,
                other=zero,
                volatile=True,
            )

            norm = tl.sqrt(alpha * alpha + xnorm_sq)
            beta = tl.where(alpha >= zero, -norm, norm)
            reflect = xnorm_sq > zero
            beta_eff = tl.where(reflect, beta, alpha)
            tau = tl.where(reflect, (beta - alpha) / beta, zero)
            denom = alpha - beta
            # guard 0/0: see the fast path above
            denom_safe = tl.where(reflect, denom, one)
            tau_arr = tl.where(col_idx == j, tau, tau_arr)
            rdg = tl.where(col_idx == j, beta_eff, rdg)
            # scalar fix-up of the pre-barrier partials (denom_safe != 0 always)
            w = tl.where(reflect, tau * (rowj + wraw / denom_safe), zero)

            for t in range(NUM_TILES):
                lr = t * RM + rows_local
                plr = row_lo + lr
                gr = kk + plr
                rmask = (plr < M) & (plr < row_lo + CHUNK)
                wblk = tl.load(
                    Wb + gr[:, None] * sWm + (kk + col_idx)[None, :] * sWn,
                    mask=rmask[:, None] & (col_idx[None, :] < ib),
                    other=zero,
                )
                col = tl.sum(tl.where(col_idx[None, :] == j, wblk, zero), axis=1)
                v_tail = col / denom_safe
                v = tl.where(plr > j, v_tail, tl.where(plr == j, one, zero))
                v = tl.where(reflect, v, tl.where(plr == j, one, zero))
                tl.store(Vb + gr * sVm + (kk + j) * sVn, v, mask=rmask)
                upd = v[:, None] * w[None, :]
                upd = tl.where(wmask[None, :], upd, zero)
                wblk = wblk - upd
                tl.store(
                    Wb + gr[:, None] * sWm + (kk + col_idx)[None, :] * sWn,
                    wblk,
                    mask=rmask[:, None] & wmask[None, :],
                )
            tl.debug_barrier()
        # flush tau and the R diagonal with one vector store each
        tl.store(TAUb + (kk + col_idx) * sTauN, tau_arr, mask=col_idx < nr)
        tl.store(
            Wb + (kk + col_idx) * sWm + (kk + col_idx) * sWn, rdg, mask=col_idx < nr
        )


# ===========================================================================
# Kernel 3: register-resident TSQR local QR.  One CTA per row block; the
# whole block tile (BM x IBN) lives in registers across the reflector chain --
# no atomics, no cross-CTA barriers, no global re-reads.
# Output: R_blocks[block] = triu(local R); with STORE_V the reflectors
# (LAPACK-style v with implicit-1 pivot, stored with the 1) and tau are
# written to V_local / TAU_local for the downstream Q application.
# ===========================================================================
@libentry()
@triton.jit
def _tsqr_local_sram_kernel(
    W,
    R_out,
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
    sRb,
    sRm,
    sRn,
    sVb,
    sVm,
    sVn,
    sTauB,
    BM: tl.constexpr,
    IBN: tl.constexpr,
    STORE_V: tl.constexpr,
):
    pid_b = tle.program_id(0)
    block_id = tle.program_id(1)
    Wb = W + pid_b * sWb
    ROb = R_out + pid_b * sRb + block_id * sRm
    Vb = V_local + pid_b * sVb
    TAUb = TAU_local + pid_b * sTauB + block_id * n

    dt = Wb.dtype.element_ty
    zero = tl.full((), 0.0, dtype=dt)
    one = tl.full((), 1.0, dtype=dt)

    rm = tl.arange(0, BM)  # block-local row index
    cn = tl.arange(0, IBN)  # col index 0..n-1
    blk_start = block_id * br
    M = tl.minimum(br, m - blk_start)
    nr = tl.minimum(k_max, M)
    rmask = rm < M
    cmask = cn < n
    rows_g = blk_start + rm

    # load the whole row block into one register tile
    A = tl.load(
        Wb + rows_g[:, None] * sWm + cn[None, :] * sWn,
        mask=rmask[:, None] & cmask[None, :],
        other=zero,
    )

    for j in range(nr):
        col_j = tl.sum(tl.where(cn[None, :] == j, A, zero), axis=1)
        alpha = tl.sum(tl.where(rm == j, col_j, zero))
        xnorm_sq = tl.sum(tl.where(rm > j, col_j * col_j, zero))
        norm = tl.sqrt(alpha * alpha + xnorm_sq)
        beta = tl.where(alpha >= zero, -norm, norm)
        reflect = xnorm_sq > zero
        beta_eff = tl.where(reflect, beta, alpha)
        tau = tl.where(reflect, (beta - alpha) / beta, zero)
        denom = alpha - beta
        v_tail = tl.where(reflect, col_j / denom, zero)  # guard 0/0 NaN
        # R diagonal + reflector tail into the in-register block
        A = tl.where((rm[:, None] == j) & (cn[None, :] == j), beta_eff, A)
        A = tl.where((rm[:, None] > j) & (cn[None, :] == j), v_tail[:, None], A)
        # Householder vector vj (0/<j, 1/==j, tail/>j)
        vj = tl.where(rm > j, v_tail, tl.where(rm == j, one, zero))
        if STORE_V:
            # reflector + tau for the downstream Q application (apply kernel)
            tl.store(Vb + rows_g * sVm + j * sVn, vj, mask=rmask)
            tl.store(TAUb + j, tau)
        # trailing update within the block (cols j+1..n), in registers
        pmask = cn[None, :] > j
        w = tau * tl.sum(tl.where(pmask, vj[:, None] * A, zero), axis=0)
        A = tl.where(pmask, A - vj[:, None] * w[None, :], A)

    # extract R = triu(A) into R_blocks[block, :n, :n]
    r_tile = tl.where(rm[:, None] <= cn[None, :], A, zero)
    tl.store(
        ROb + rm[:, None] * sRn + cn[None, :],
        r_tile,
        mask=(rm[:, None] < n) & (cn[None, :] < n),
    )


# ===========================================================================
# Kernel 4: TSQR tree reduction -- factor (a group slice of) the stacked local
# R's ((num_blocks*n) x n) in registers, storing the reflectors in place
# (LAPACK style: v tail below the pivot, beta on the diagonal).  Grid is
# (B, num_groups): CTA (b, g) factors rows [g*grp*n, min((g+1)*grp*n,
# num_blocks*n)) of the stack, writes its n x n R factor to R_out[g] and its
# Q factor back to the same rows of Qt.
# Q_t = H_0 H_1 ... H_{n-1} [I; 0] is built by applying the reflectors to
# the identity in REVERSE order (single extra register tile, no R inverse --
# robust to exactly rank-deficient inputs, unlike Q = A R^{-1}).
# A single group (num_groups == 1) covers the whole stack (flat reduction);
# with several groups this is the first/second level of a two-level tree.
# Stacks too tall even for the tree go through the blocked path
# (_blocked_qr + _assemble_q) instead.
# ===========================================================================
@libentry()
@triton.jit
def _tsqr_tree_kernel(
    Rblocks,
    R_out,
    Qt,
    n,
    num_blocks,
    grp,
    write_Q,
    sRb,
    sRm,
    sRn,
    BRM: tl.constexpr,
    IBN: tl.constexpr,
):
    pid_b = tle.program_id(0)
    pid_g = tle.program_id(1)
    dt = Rblocks.dtype.element_ty
    zero = tl.full((), 0.0, dtype=dt)
    one = tl.full((), 1.0, dtype=dt)

    # Rblocks and Qt are internal contiguous buffers: a (B, num_blocks, n, n)
    # stack viewed as (B, num_blocks*n, n); Qt shares the layout.  Only the
    # R_out strides are passed (it may be the caller's non-contiguous out_R).
    rb_batch = num_blocks * n * n
    rr = tl.arange(0, BRM)  # reduction rows (padded grp*n)
    cn = tl.arange(0, IBN)  # padded n
    row0 = pid_g * grp * n
    Rm = tl.minimum(grp, num_blocks - pid_g * grp) * n
    redmask = rr < Rm
    cmask = cn < n

    # ---- factor the stacked local R factors, reflectors kept in place ----
    G = tl.load(
        Rblocks + pid_b * rb_batch + (row0 + rr)[:, None] * n + cn[None, :],
        mask=redmask[:, None] & cmask[None, :],
        other=zero,
    )
    tau_vec = tl.zeros([IBN], dtype=dt)
    for j in range(n):
        col_j = tl.sum(tl.where(cn[None, :] == j, G, zero), axis=1)
        alpha = tl.sum(tl.where(rr == j, col_j, zero))
        xnorm_sq = tl.sum(tl.where(rr > j, col_j * col_j, zero))
        norm = tl.sqrt(alpha * alpha + xnorm_sq)
        beta = tl.where(alpha >= zero, -norm, norm)
        reflect = xnorm_sq > zero
        beta_eff = tl.where(reflect, beta, alpha)
        tau = tl.where(reflect, (beta - alpha) / beta, zero)
        denom = alpha - beta
        v_tail = tl.where(reflect, col_j / denom, zero)  # guard 0/0 NaN
        G = tl.where((rr[:, None] == j) & (cn[None, :] == j), beta_eff, G)
        G = tl.where((rr[:, None] > j) & (cn[None, :] == j), v_tail[:, None], G)
        tau_vec = tl.where(cn == j, tau, tau_vec)
        vj = tl.where(rr > j, v_tail, tl.where(rr == j, one, zero))
        pmask = cn[None, :] > j
        w = tau * tl.sum(tl.where(pmask, vj[:, None] * G, zero), axis=0)
        G = tl.where(pmask, G - vj[:, None] * w[None, :], G)
    # R = triu of the first n rows (the v tails live at rows > j, the diagonal
    # holds beta_eff, so rows < n are exactly R's upper triangle).
    R_tile = tl.where(rr[:, None] <= cn[None, :], G, zero)
    tl.store(
        R_out + pid_b * sRb + (pid_g * n + rr)[:, None] * sRm + cn[None, :] * sRn,
        R_tile,
        mask=(rr[:, None] < n) & cmask[None, :],
    )

    if write_Q:
        # ---- Q_t = H_0 ... H_{n-1} [I; 0] via reverse reflector application --
        X = tl.where(rr[:, None] == cn[None, :], one, zero)
        for jj in range(n):
            j = n - 1 - jj
            v_tail = tl.sum(tl.where(cn[None, :] == j, G, zero), axis=1)
            vj = tl.where(rr > j, v_tail, tl.where(rr == j, one, zero))
            tau_j = tl.sum(tl.where(cn == j, tau_vec, zero))
            w = tau_j * tl.sum(vj[:, None] * X, axis=0)
            X = X - vj[:, None] * w[None, :]
        tl.store(
            Qt + pid_b * rb_batch + (row0 + rr)[:, None] * n + cn[None, :],
            X,
            mask=redmask[:, None] & cmask[None, :],
        )


# ===========================================================================
# Kernel 5: TSQR Q application -- one CTA per row block.  Rebuilds the block's
# local Q factor from its stored reflectors (reverse application to [I; 0],
# robust to zero columns) and multiplies by the tree factor(s):
#     flat:       Q[block rows] = Q_local @ Q_t[block*n : block*n+n, :]
#     two-level:  Q[block rows] = Q_local @ Q_g[block rows] @ Q_t[group rows]
# where Q_g is the first-level (group) factor and Q_t the top-level one.
# FOLD_TREE (small stacks): every CTA redundantly factors the stacked local
# R's in registers and builds Q_t itself (a few tiny reflector steps), which
# eliminates the separate tree-reduction launch -- kernel-launch overhead
# dominates TSQR on small tall-skinny inputs.  CTA (b, 0) also writes R.
# ===========================================================================
@libentry()
@triton.jit
def _tsqr_apply_kernel(
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
):
    pid_b = tle.program_id(0)
    block_id = tle.program_id(1)
    # V_local / TAU_local / Qt / Qt2 / Rblocks are internal contiguous buffers
    # ((B, m, n), (B, num_blocks, n), two (B, *, n) factor stacks and the
    # (B, num_blocks, n, n) R stack); their strides are derived from the
    # shapes.  Only Q / Racc strides are passed (caller's out= buffers).
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
        # Redundantly factor the (small) stacked local R's in registers --
        # same reflector loop as the tree kernel -- and build Q_t locally.
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
        # gather this block's rows of Q_t via a 0/1 selection matmul; rows
        # beyond n belong to other blocks and must not leak into the output
        # through X's padded identity columns
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
        Qti = tl.dot(Qti, Qt2i, allow_tf32=False)
    out = tl.dot(X, Qti, allow_tf32=False)
    tl.store(
        Q + pid_b * sQb + (blk_start + rm)[:, None] * sQm + cn[None, :] * sQn,
        out,
        mask=rmask[:, None] & cmask[None, :],
    )


# ===========================================================================
# Kernel 6: fused QR for matrices that fit in shared memory.
# One CTA per matrix does the *entire* job -- unblocked Householder
# factorisation, R extraction and (optionally) Q assembly -- operating on
# in-SRAM tiles with no global round-trips and no per-panel launches.  This is
# what makes single small/medium matrices competitive (one launch instead of
# ~20).
# ===========================================================================
@libentry()
@triton.jit
def _qr_fused_kernel(
    W,
    Qout,
    Rout,
    m,
    n,
    k,
    qcols,
    rrows,
    put_Q,
    sWb,
    sWm,
    sWn,
    sQb,
    sQm,
    sQn,
    sRb,
    sRm,
    sRn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tle.program_id(0)
    Wb = W + pid * sWb
    dt = Wb.dtype.element_ty
    zero = tl.full((), 0.0, dtype=dt)
    one = tl.full((), 1.0, dtype=dt)

    rm = tl.arange(0, BM)
    cn = tl.arange(0, BN)
    cq = tl.arange(0, BQ)
    ik = tl.arange(0, BK)
    rmask = rm < m
    cmask = cn < n

    # load the whole matrix into a register/SRAM tile
    A = tl.load(
        Wb + rm[:, None] * sWm + cn[None, :] * sWn,
        mask=rmask[:, None] & cmask[None, :],
        other=zero,
    )

    tau_arr = tl.zeros([BK], dtype=dt)

    # ---- unblocked Householder QR, in place on the tile ----
    for j in range(k):
        col_j = tl.sum(tl.where(cn[None, :] == j, A, zero), axis=1)  # A[:, j]
        alpha = tl.sum(tl.where(rm == j, col_j, zero))
        xnorm_sq = tl.sum(tl.where(rm > j, col_j * col_j, zero))
        norm = tl.sqrt(alpha * alpha + xnorm_sq)
        beta = tl.where(alpha >= zero, -norm, norm)
        reflect = xnorm_sq > zero
        beta_eff = tl.where(reflect, beta, alpha)
        tau = tl.where(reflect, (beta - alpha) / beta, zero)
        denom = alpha - beta
        # Guard the 0/0 case: without a reflection (zero tail) v_tail would be
        # NaN, and it is stored into A's strict-lower triangle where the Q
        # assembly below reads it back (0 * NaN = NaN).
        v_tail = tl.where(reflect, col_j / denom, zero)  # reflector tail (rows > j)
        tau_arr = tl.where(ik == j, tau, tau_arr)

        # write R diagonal A[j,j] and reflector tail A[r>j, j]
        A = tl.where((rm[:, None] == j) & (cn[None, :] == j), beta_eff, A)
        A = tl.where((rm[:, None] > j) & (cn[None, :] == j), v_tail[:, None], A)

        # reflector vector vj: 0 (r<j), 1 (r==j), v_tail (r>j)
        vj = tl.where(rm > j, v_tail, tl.where(rm == j, one, zero))
        vj = tl.where(reflect, vj, tl.where(rm == j, one, zero))

        # trailing update A[:, c>j] -= vj * (tau * vj^T A[:, c>j])
        w = tau * tl.sum(vj[:, None] * A, axis=0)  # (BN,)
        w = tl.where((cn > j) & cmask, w, zero)
        A = A - vj[:, None] * w[None, :]

    # ---- R = triu(A), written to the first `rrows` rows ----
    R_tile = tl.where(rm[:, None] <= cn[None, :], A, zero)
    rrmask = rm < rrows
    tl.store(
        Rout + pid * sRb + rm[:, None] * sRm + cn[None, :] * sRn,
        R_tile,
        mask=rrmask[:, None] & cmask[None, :],
    )

    if put_Q:
        # Q (m x qcols) = identity, then apply reflectors in reverse
        Q = tl.where(rm[:, None] == cq[None, :], one, zero)
        Q = tl.where(rmask[:, None] & (cq[None, :] < qcols), Q, zero)
        for jj in range(k):
            j = k - 1 - jj
            tauj = tl.sum(tl.where(ik == j, tau_arr, zero))
            col_j = tl.sum(tl.where(cn[None, :] == j, A, zero), axis=1)
            vj = tl.where(rm > j, col_j, tl.where(rm == j, one, zero))
            w = tauj * tl.sum(vj[:, None] * Q, axis=0)  # (BQ,)
            Q = Q - vj[:, None] * w[None, :]
        tl.store(
            Qout + pid * sQb + rm[:, None] * sQm + cq[None, :] * sQn,
            Q,
            mask=rmask[:, None] & (cq[None, :] < qcols),
        )


# ===========================================================================
# Kernel 7: SRAM-resident panel factorisation (blocked-path geqrt replacement).
# The multi-CTA _geqrt_mcta_kernel re-reads the panel from global memory on each
# of its 4 passes per reflector and, for moderately-tall panels (e.g. 256x32),
# only spawns NC = M/RM ~= 4 CTAs -- severe SM under-utilisation plus redundant
# global traffic.  This kernel keeps the whole panel in one SRAM tile (one CTA
# per batch) and runs the unblocked reflector chain with no global re-reads, the
# same trick that makes _qr_fused_kernel fast.  Measured ~3x faster than mcta on
# a 256x32 panel.  Used only while the panel tile fits shared memory
# (BM*IB*itemsize <= ~SRAM budget); taller panels fall back to multi-CTA.
# Output contract matches _geqrt_kernel: V (unit diag + tail) to the V buffer,
# tau to the tau buffer, R left in W's panel upper triangle.
# ===========================================================================
@libentry()
@triton.jit
def _geqrt_sram_kernel(
    W,
    V,
    TAU,
    M,
    ib,
    kk,
    sWb,
    sWm,
    sWn,
    sVb,
    sVm,
    sVn,
    sTauB,
    sTauN,
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

    # tau accumulates in registers, flushed with ONE vector store after the
    # loop (0-d scalar stores: see _geqrt_kernel)
    tau_arr = tl.zeros([IBN], dtype=dt)
    for j in range(ib):
        col_j = tl.sum(tl.where(cn[None, :] == j, A, zero), axis=1)
        alpha = tl.sum(tl.where(rm == j, col_j, zero))
        xnorm_sq = tl.sum(tl.where(rm > j, col_j * col_j, zero))
        norm = tl.sqrt(alpha * alpha + xnorm_sq)
        beta = tl.where(alpha >= zero, -norm, norm)
        reflect = xnorm_sq > zero
        beta_eff = tl.where(reflect, beta, alpha)
        tau = tl.where(reflect, (beta - alpha) / beta, zero)
        denom = alpha - beta
        v_tail = tl.where(reflect, col_j / denom, zero)  # guard 0/0 NaN
        # R diagonal + reflector tail into the in-SRAM panel
        A = tl.where((rm[:, None] == j) & (cn[None, :] == j), beta_eff, A)
        A = tl.where((rm[:, None] > j) & (cn[None, :] == j), v_tail[:, None], A)
        tau_arr = tl.where(cn == j, tau, tau_arr)
        # write Householder vector vj (0/<j, 1/==j, tail/>j) to the V buffer
        vj = tl.where(rm > j, v_tail, tl.where(rm == j, one, zero))
        vj = tl.where(reflect, vj, tl.where(rm == j, one, zero))
        tl.store(Vb + rows_g * sVm + (kk + j) * sVn, vj, mask=rmask)
        # trailing update within the panel (cols j+1..ib), in SRAM
        pmask = cn[None, :] > j
        w = tau * tl.sum(tl.where(pmask, vj[:, None] * A, zero), axis=0)
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


# ===========================================================================
# Kernel 8: build the WY factor T (DLARFT, Gram-solve form).  One CTA per
# batch element.
#
# The Gram-solve trick (ml-mike.com/writing/qr_v2) builds M = T^{-1} directly
# from the reflector Gram matrix, then inverts the small upper-triangular M
# in-kernel (32 serial steps on a 32x32 tile) to obtain T.  Larfb then applies
# Y = T (T^H) @ W1 with plain GEMMs -- no per-tile serial solve.
# ===========================================================================
@libentry()
@triton.jit
def _larft_kernel(
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
    RM: tl.constexpr,
    IBN: tl.constexpr,
    INVERT: tl.constexpr,
):
    """Build the WY factor from the reflector Gram matrix.

    INVERT=True (fp32): store T = (triu(V^H V, 1) + diag(1/tau))^{-1}, so larfb
    applies Y = T (T^H) @ W1 with plain GEMMs (blog w1431).
    INVERT=False (fp64): store M = T^{-1} directly; larfb falls back to its
    in-kernel triangular solve (fp64 GEMMs + serial inversion are slower).
    """
    pid = tle.program_id(0)
    Vb = V + pid * sVb
    TAUb = TAU + pid * sTauB
    Tb = MOUT + pid * sMb

    dt = Vb.dtype.element_ty
    zero = tl.full((), 0.0, dtype=dt)
    one = tl.full((), 1.0, dtype=dt)
    idx = tl.arange(0, IBN)  # row/col index 0..IBN-1
    num_tiles = (M + RM - 1) // RM

    # ---- Gram G = V^H V  (IBN x IBN), one parallel matmul over the M rows ----
    # (V holds explicit zeros above each reflector's diagonal -- written by the
    # geqrt kernels -- so no trapezoid mask is needed here; adding one showed a
    # measurable slowdown by inhibiting vectorised loads.)
    G = tl.zeros((IBN, IBN), dtype=dt)
    for t in range(num_tiles):
        rows = t * RM + tl.arange(0, RM)
        rmask = rows < M
        v_off = rows[:, None] * sVm + idx[None, :] * sVn
        Vt = tl.load(Vb + v_off, mask=rmask[:, None] & (idx[None, :] < ib), other=0.0)
        G += tl.dot(tl.trans(Vt), Vt, allow_tf32=False)

    # ---- M = T^{-1} = triu(G, 1) + diag(1/tau)  (upper triangular) ----
    tau_vec = tl.load(TAUb + idx * sTauN, mask=idx < ib, other=1.0)
    # tau == 0 (rank-deficient panel) makes the reflector the identity; its
    # T row must come out exactly zero.  Put +inf on M's diagonal explicitly
    # instead of relying on 1/0, so the reciprocal-based solve in larfb (fp64
    # path) yields that zero row on any floating-point backend.
    inv_tau = tl.where(tau_vec != 0.0, 1.0 / tau_vec, float("inf"))
    Mmat = tl.where(idx[:, None] < idx[None, :], G, 0.0)
    Mmat = tl.where(idx[:, None] == idx[None, :], inv_tau[:, None], Mmat)

    Tmat = tl.zeros((IBN, IBN), dtype=dt)
    if INVERT:
        # ---- invert M -> T via the nilpotent (Neumann) product ----
        # M = D(I + N) with D = diag(M), N = D^{-1} U strictly upper, hence
        # N^IBN == 0 and
        #   T = M^{-1} = (I + N)^{-1} D^{-1}
        #   = (I + X)(I + X^2)(I + X^4)... D^{-1},   X = -N.
        # Plain GEMMs only: the serial masked-reduction back-substitution this
        # replaces was flaky on vendor backends with weak cross-warp sync.
        # dinv = 1/M_ii comes out division-free: M_ii = 1/tau (resp. +inf for
        # tau == 0), so dinv = tau (resp. 0) -- the zero-tau T row is exactly
        # zero, as with the old reciprocal scheme.
        dinv = tl.where(tau_vec != 0.0, tau_vec, zero)
        U = tl.where(idx[:, None] < idx[None, :], Mmat, zero)
        X = -U * dinv[:, None]  # N = D^{-1} U: ROW scaling
        eye = tl.where(idx[:, None] == idx[None, :], one, zero)
        S = eye + X
        # 5 squarings cover IBN <= 64 (S then holds the full series through
        # X^63); extra iterations are no-ops once X is nilpotent-zero.
        for _ in range(5):
            X = tl.dot(X, X, allow_tf32=False)
            S = tl.dot(S, eye + X, allow_tf32=False)
        Tmat = S * dinv[None, :]  # T = S D^{-1}: COLUMN scaling
    out = Tmat if INVERT else Mmat

    # ---- store T (INVERT) or M = T^{-1} (upper triangular) ----
    tl.store(
        Tb + idx[:, None] * sMm + idx[None, :] * sMn,
        out,
        mask=(idx[:, None] < ib) & (idx[None, :] < ib),
    )


# ===========================================================================
# Kernel 9: apply block reflector H = I - V T V^H on the left (DLARFB).
#   C <- C - V Y,  Y = T @ W1   (Q assembly,   UPPER=True)
#   C <- C - V Y,  Y = T^H @ W1 (trailing update, UPPER=False)
# with T pre-computed (inverted) by _larft_kernel -- plain GEMMs only, no
# per-tile serial triangular solve (blog w1431: inverse + GEMM).
# ===========================================================================
@triton.jit
def _larfb_kernel(
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
    num_tiles = (M + RM - 1) // RM

    # ---- load T (fp32, upper triangular) or M = T^{-1} (fp64) ----
    Msram = tl.load(
        Tb + col_idx[:, None] * sTm + col_idx[None, :] * sTn,
        mask=(col_idx[:, None] < ib) & (col_idx[None, :] < ib),
        other=0.0,
    )

    # ---- W1 = V^H C[:, p-tile] ----
    W1 = tl.zeros((IBN, TN), dtype=dt)
    for t in range(num_tiles):
        rows = t * RM + tl.arange(0, RM)
        rmask = rows < M
        v_off = rows[:, None] * sVm + col_idx[None, :] * sVn
        c_off = rows[:, None] * sCm + p_idx[None, :] * sCn
        Vt = tl.load(
            Vb + v_off, mask=rmask[:, None] & (col_idx[None, :] < ib), other=0.0
        )
        Ct = tl.load(Cb + c_off, mask=rmask[:, None] & pmask[None, :], other=0.0)
        W1 += tl.dot(tl.trans(Vt), Ct, allow_tf32=False)
    W1 = tl.where(col_idx[:, None] < ib, W1, 0.0)

    # ---- Y = T (T^H) @ W1 : one GEMM, no serial solve (fp32) ----
    # or in-kernel triangular substitution on M (fp64; fp64 GEMMs + serial
    # inversion are slower than the masked-reduction solve on this hardware).
    Y = tl.zeros((IBN, TN), dtype=dt)
    if SOLVE:
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
                    # *reciprocal: zero-tau -> Mii=+inf -> exact zero row
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
                    # *reciprocal: zero-tau -> Mii=+inf -> exact zero row
                    Yrow = (W1row - contrib) * (1.0 / Mii)
                    Y = tl.where(col_idx[:, None] == i, Yrow[None, :], Y)
    else:
        Tsram = Msram
        if not UPPER:
            Tsram = tl.trans(Tsram)
        Y = tl.dot(Tsram, W1, allow_tf32=False)
    Y = tl.where(col_idx[:, None] < ib, Y, 0.0)

    # ---- C[:, p-tile] -= V Y ----
    for t in range(num_tiles):
        rows = t * RM + tl.arange(0, RM)
        rmask = rows < M
        v_off = rows[:, None] * sVm + col_idx[None, :] * sVn
        c_off = rows[:, None] * sCm + p_idx[None, :] * sCn
        Vt = tl.load(
            Vb + v_off, mask=rmask[:, None] & (col_idx[None, :] < ib), other=0.0
        )
        Ct = tl.load(Cb + c_off, mask=rmask[:, None] & pmask[None, :], other=0.0)
        Ct = Ct - tl.dot(Vt, Y, allow_tf32=False)
        tl.store(Cb + c_off, Ct, mask=rmask[:, None] & pmask[None, :])


# ===========================================================================
# Kernel 10: compose pairs of adjacent panel T's into one (2*ib) x (2*ib)
# compact-WY factor, for the paired Q-assembly path (fp32).
#   H_2g H_{2g+1} = I - [V1 V2] T_pair [V1 V2]^H,
#   T_pair = [[T1, -T1 (V1^H V2) T2], [0, T2]]
# Z = V1^H V2 is summed over rows >= kk+ib only (V2 is zero above its own
# panel start; V must be zero-initialised so the fused assembly kernel reads
# exact zeros above each reflector's diagonal within a pair).
# One CTA per pair, ONE launch for all pairs; reads T1/T2 before writing, so
# the pair factors can overwrite Tbuf in place (pairs touch disjoint regions).
# ===========================================================================
@libentry()
@triton.jit
def _tcompose_pair_kernel(
    V,
    TIN,
    m,
    k,
    ib,
    sVb,
    sVm,
    sVn,
    sTib,
    sTim,
    sTin,
    RM: tl.constexpr,
    IBN: tl.constexpr,
):
    pid_b = tle.program_id(0)
    g = tle.program_id(1)
    Vb = V + pid_b * sVb
    Tb = TIN + pid_b * sTib
    kk = g * (2 * ib)
    iba = tl.minimum(2 * ib, k - kk)
    dt = Vb.dtype.element_ty
    zero = tl.zeros((), dtype=dt)
    half = IBN // 2
    rr = tl.arange(0, IBN)
    cc = tl.arange(0, IBN)
    rm = tl.arange(0, RM)

    # All operands live in (IBN, IBN) tiles with each factor embedded in its
    # natural quadrant, so the products land in place with plain dots:
    #   Zfull (top-right)   = V1full^H @ V2full
    #   off   (top-right)   = -T1full @ Zfull @ T2full
    #   T_pair = T1full (top-left) + off + T2full (bottom-right)
    rows_start = kk + ib
    Zf = tl.zeros((IBN, IBN), dtype=dt)
    for t in range((m - rows_start + RM - 1) // RM):
        rows = rows_start + t * RM + rm
        rmask = rows < m
        v1 = tl.load(
            Vb + rows[:, None] * sVm + (kk + cc)[None, :] * sVn,
            mask=rmask[:, None] & (cc[None, :] < half) & ((kk + cc)[None, :] < k),
            other=zero,
        )
        v2 = tl.load(
            Vb + rows[:, None] * sVm + (kk + cc)[None, :] * sVn,
            mask=rmask[:, None] & (cc[None, :] >= half) & ((kk + cc)[None, :] < k),
            other=zero,
        )
        Zf += tl.dot(tl.trans(v1), v2, allow_tf32=False)

    t1m = (rr[:, None] < half) & (cc[None, :] < half)
    t2m = (rr[:, None] >= half) & (cc[None, :] >= half)
    t1 = tl.load(
        Tb + (kk + rr)[:, None] * sTim + (kk + cc)[None, :] * sTin,
        mask=t1m & ((kk + rr)[:, None] < k) & ((kk + cc)[None, :] < k),
        other=zero,
    )
    t2 = tl.load(
        Tb + (kk + rr)[:, None] * sTim + (kk + cc)[None, :] * sTin,
        mask=t2m & ((kk + rr)[:, None] < k) & ((kk + cc)[None, :] < k),
        other=zero,
    )
    off = -tl.dot(t1, tl.dot(Zf, t2, allow_tf32=False), allow_tf32=False)

    out = t1 + t2 + off
    tl.store(
        Tb + (kk + rr)[:, None] * sTim + (kk + cc)[None, :] * sTin,
        out,
        mask=(rr[:, None] < iba) & (cc[None, :] < iba),
    )


# ===========================================================================
# Kernel 11: fused Q assembly -- identity + all panels in ONE launch.
# Q <- (H_0 H_1 ... H_{P-1}) applied to identity: each CTA owns a TN-wide
# column slice of Q and loops the panels in reverse, loading V_p/T_p and
# applying Q <- Q - V_p (T_p (V_p^H Q)) with the same GEMM-only body as
# _larfb_kernel(UPPER=True).  Replaces _identity_kernel + P per-panel larfb
# launches (blog w1422's launch-count lesson: fewer, bigger kernels).
# ===========================================================================
@libentry()
@triton.jit
def _assemble_q_fused_kernel(
    V,
    Tbuf,
    Q,
    m,
    k,
    qcols,
    ib,
    P,
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
    IBN: tl.constexpr,
    TN: tl.constexpr,
):
    pid_b = tle.program_id(0)
    pid_p = tle.program_id(1)
    Vb = V + pid_b * sVb
    Tb = Tbuf + pid_b * sTb
    Qb = Q + pid_b * sQb

    dt = Vb.dtype.element_ty
    zero = tl.full((), 0.0, dtype=dt)
    one = tl.full((), 1.0, dtype=dt)
    col_idx = tl.arange(0, IBN)
    p_idx = pid_p * TN + tl.arange(0, TN)
    pmask = p_idx < qcols
    rm = tl.arange(0, RM)

    # Q column slice = identity (rows 0..m-1 of the p-tile columns)
    for t in range((m + RM - 1) // RM):
        rows = t * RM + rm
        rmask = rows < m
        qt = tl.where(rows[:, None] == p_idx[None, :], one, zero)
        qt = tl.where(rmask[:, None] & pmask[None, :], qt, zero)
        tl.store(
            Qb + rows[:, None] * sQm + p_idx[None, :] * sQn,
            qt,
            mask=rmask[:, None] & pmask[None, :],
        )
    # The panel loop below reads back what this loop just wrote (and each
    # panel iteration reads what the previous one stored): cross-thread
    # global write->read within a CTA is unordered without a barrier.
    tl.debug_barrier()

    # apply panels in reverse: Q <- H_p Q
    for pp in range(P - 1, -1, -1):
        kk = pp * ib
        iba = ib
        if kk + iba > k:
            iba = k - kk
        num_tiles = (m - kk + RM - 1) // RM
        # W1 = V_p^H Q[kk:m, p-tile]
        W1 = tl.zeros((IBN, TN), dtype=dt)
        for t in range(num_tiles):
            rows = kk + t * RM + rm
            rmask = rows < m
            v_off = rows[:, None] * sVm + (kk + col_idx)[None, :] * sVn
            q_off = rows[:, None] * sQm + p_idx[None, :] * sQn
            Vt = tl.load(
                Vb + v_off, mask=rmask[:, None] & (col_idx[None, :] < iba), other=zero
            )
            Qt = tl.load(Qb + q_off, mask=rmask[:, None] & pmask[None, :], other=zero)
            W1 += tl.dot(tl.trans(Vt), Qt, allow_tf32=False)
        W1 = tl.where(col_idx[:, None] < iba, W1, zero)
        # T_p (upper triangular, pre-built by _blocked_qr / larft)
        Tt = tl.load(
            Tb + (kk + col_idx)[:, None] * sTm + (kk + col_idx)[None, :] * sTn,
            mask=(col_idx[:, None] < iba) & (col_idx[None, :] < iba),
            other=zero,
        )
        # fp32 only: Tbuf holds T -> Y = T @ W1 (one GEMM)
        Y = tl.dot(Tt, W1, allow_tf32=False)
        Y = tl.where(col_idx[:, None] < iba, Y, zero)
        # Q[kk:m, p-tile] -= V_p Y
        for t in range(num_tiles):
            rows = kk + t * RM + rm
            rmask = rows < m
            v_off = rows[:, None] * sVm + (kk + col_idx)[None, :] * sVn
            q_off = rows[:, None] * sQm + p_idx[None, :] * sQn
            Vt = tl.load(
                Vb + v_off, mask=rmask[:, None] & (col_idx[None, :] < iba), other=zero
            )
            Qt = tl.load(Qb + q_off, mask=rmask[:, None] & pmask[None, :], other=zero)
            Qt = Qt - tl.dot(Vt, Y, allow_tf32=False)
            tl.store(Qb + q_off, Qt, mask=rmask[:, None] & pmask[None, :])
        # next panel's W1 accumulation reads the tiles just stored
        tl.debug_barrier()


# ===========================================================================
# Kernel 12: single-panel Q assembly in one pass (fp32).
# When the whole factorization is one panel (k <= IB, P == 1), the block
# reflector is H = I - V T V^H, so Q = H I = I - V T V^H.  W1 = T V[p,:]^H
# is read straight off V's rows (no need to read Q back), and each output
# tile is written exactly once:  Q_tile = I_tile - V_t @ W1.  This removes
# the per-tile serial W1 build over the m rows and the Q re-reads that the
# generic _assemble_q_fused_kernel does (measured ~5x on 8192x8 complete).
# ===========================================================================
@libentry()
@triton.jit
def _assemble_q_single_panel_kernel(
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
    pid_p = tle.program_id(1)
    Vb = V + pid_b * sVb
    Tb = T + pid_b * sTb
    Qb = Q + pid_b * sQb

    dt = Vb.dtype.element_ty
    zero = tl.full((), 0.0, dtype=dt)
    one = tl.full((), 1.0, dtype=dt)
    col_idx = tl.arange(0, IBN)  # padded reflector columns
    p_idx = pid_p * TN + tl.arange(0, TN)  # Q columns of this CTA
    pmask = p_idx < qcols
    rm = tl.arange(0, RM)

    # W1 = T @ V[p_idx, :]^H  (IBN x TN); identity rows of Q are exactly p_idx.
    # Masks must use k (number of reflectors), not n: IBN = max(16, pow2(k))
    # can exceed k, and V/T only have k valid columns -- masking by n reads
    # out-of-bounds garbage into Q for wide matrices (k < 16, n >> k).
    Vrows = tl.load(
        Vb + p_idx[:, None] * sVm + col_idx[None, :] * sVn,
        mask=(p_idx[:, None] < m) & (col_idx[None, :] < k),
        other=zero,
    )
    Tt = tl.load(
        Tb + col_idx[:, None] * sTm + col_idx[None, :] * sTn,
        mask=(col_idx[:, None] < k) & (col_idx[None, :] < k),
        other=zero,
    )
    W1 = tl.dot(Tt, tl.trans(Vrows), allow_tf32=False)
    W1 = tl.where(col_idx[:, None] < k, W1, zero)

    # Q_tile = I_tile - V_t @ W1, written once
    for t in range((m + RM - 1) // RM):
        rows = t * RM + rm
        rmask = rows < m
        Vt = tl.load(
            Vb + rows[:, None] * sVm + col_idx[None, :] * sVn,
            mask=rmask[:, None] & (col_idx[None, :] < k),
            other=zero,
        )
        Qt = tl.where(rows[:, None] == p_idx[None, :], one, zero)
        Qt = Qt - tl.dot(Vt, W1, allow_tf32=False)
        tl.store(
            Qb + rows[:, None] * sQm + p_idx[None, :] * sQn,
            Qt,
            mask=rmask[:, None] & pmask[None, :],
        )


# ===========================================================================
# Kernel 13: copy the upper triangle of W into R (zero below).  R[i,j]=W[i,j] if i<=j.
# ===========================================================================
@libentry()
@triton.jit
def _triu_copy_kernel(
    W, ROUT, rm, n, sWb, sWm, sWn, sRb, sRm, sRn, BLOCK: tl.constexpr
):
    pid_b = tle.program_id(0)  # batch
    pid_e = tle.program_id(1)  # element tile
    numel = rm * n
    offs = pid_e * BLOCK + tl.arange(0, BLOCK)
    mmask = offs < numel
    i = offs // n
    j = offs % n
    val = tl.load(W + pid_b * sWb + i * sWm + j * sWn, mask=mmask, other=0.0)
    tl.store(
        ROUT + pid_b * sRb + i * sRm + j * sRn, tl.where(i <= j, val, 0.0), mask=mmask
    )


# ===========================================================================
# Kernel 14: write an identity matrix into Q (m x qcols)
# ===========================================================================
@libentry()
@triton.jit
def _identity_kernel(Q, m, qcols, grid_e, sQb, sQm, sQn, BLOCK: tl.constexpr):
    # 1D grid (B * grid_e): a 2D (B, grid_e) grid would exceed CUDA's 65535
    # gridY limit for large complete-mode Q (e.g. 8192x8192 -> grid_e = 65536).
    pid = tle.program_id(0)
    pid_b = pid // grid_e
    pid_e = pid % grid_e
    numel = m * qcols
    offs = pid_e * BLOCK + tl.arange(0, BLOCK)
    mmask = offs < numel
    i = offs // qcols
    j = offs % qcols
    tl.store(
        Q + pid_b * sQb + i * sQm + j * sQn, tl.where(i == j, 1.0, 0.0), mask=mmask
    )


# ===========================================================================
# Python orchestration (memory/layout + kernel launches only)
# ===========================================================================
def _launch_geqrt_sram(W, V, tau, m, k, kk, ib, B):
    """SRAM-resident panel factorisation (single CTA per batch)."""
    ib_active = min(ib, k - kk)
    M = m - kk
    sWb, sWm, sWn = W.stride()
    sVb, sVm, sVn = V.stride()
    sTauB, sTauN = tau.stride()
    _geqrt_sram_kernel[(B,)](
        W,
        V,
        tau,
        M,
        ib_active,
        kk,
        sWb,
        sWm,
        sWn,
        sVb,
        sVm,
        sVn,
        sTauB,
        sTauN,
        BM=triton.next_power_of_2(M),
        IBN=max(16, triton.next_power_of_2(ib_active)),
        num_warps=_GEQRT_SRAM_WARPS,
    )


def _launch_geqrt(W, V, tau, m, k, kk, ib, B):
    ib_active = min(ib, k - kk)
    M = m - kk
    nr = min(ib_active, k - kk)
    sWb, sWm, sWn = W.stride()
    sVb, sVm, sVn = V.stride()
    sTauB, sTauN = tau.stride()
    _geqrt_kernel[(B,)](
        W,
        V,
        tau,
        M,
        kk,
        ib_active,
        nr,
        sWb,
        sWm,
        sWn,
        sVb,
        sVm,
        sVn,
        sTauB,
        sTauN,
        RM=_PANEL_RM,
        IBN=max(16, triton.next_power_of_2(ib_active)),
    )


def _launch_geqrt_mcta(
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
    M = m - kk
    nr = min(ib, k - kk)
    # p is the NOMINAL panel index (kk // blocking width), passed by the caller:
    # the scratch buffers are laid out per nominal panel, and for a partial last
    # panel (ib_active < blocking) kk // ib_active would index far out of bounds.
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
    _geqrt_mcta_kernel[(B, NC)](
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


def _launch_larft(V, tau, Tout, m, kk, ib, B):
    M = m - kk
    sVb, sVm, sVn = V.stride()
    sTauB, sTauN = tau.stride()
    sTb, sTm, sTn = Tout.stride()
    _larft_kernel[(B,)](
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
        INVERT=V.element_size() == 4,
        num_warps=_LARFT_WARPS,
    )


def _launch_larfb(V, Tp, C, m, p, ib, B, upper):
    """Apply block reflector; T loaded from Tp (pre-computed by _launch_larft)."""
    sVb, sVm, sVn = V.stride()
    sTb, sTm, sTn = Tp.stride()
    sCb, sCm, sCn = C.stride()
    # TN=32 is a ~1.9x win for the fp32 GEMM path on large trailing updates;
    # fp64 keeps TN=16 (the solve path and fp64 tiles spill at TN=32).
    tn = _LARFB_TN if V.element_size() == 4 else 16
    grid_p = (p + tn - 1) // tn
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
        SOLVE=V.element_size() != 4,
        num_warps=_LARFB_WARPS,
        **({"num_stages": _LARFB_STAGES} if _LARFB_STAGES else {}),
    )


def _fused_qr(W, A, orig_dtype, batch_shape, m, n, k, mode, B, out_Q=None, out_R=None):
    """Single-launch QR for matrices that fit in shared memory (m, n <= _FUSED_DIM).

    One kernel per matrix does the factorisation, R extraction and Q assembly.
    Writes directly into caller-provided out_Q/out_R when given (true out variant).
    """
    qcols = 0 if mode == "r" else (k if mode == "reduced" else m)
    rrows = k if mode in ("reduced", "r") else m
    dt = W.dtype
    dev = W.device
    # r-mode: the kernel still needs a typed 3-D Q pointer (put_Q=False, never
    # written); use a dummy, and return the caller's (0-element) out_Q as-is.
    if mode == "r":
        Q = torch.empty(B, m, 1, dtype=dt, device=dev)
    else:
        Q = (
            out_Q
            if out_Q is not None
            else torch.empty(B, m, qcols, dtype=dt, device=dev)
        )
    R = out_R if out_R is not None else torch.empty(B, rrows, n, dtype=dt, device=dev)
    BM = triton.next_power_of_2(m)
    BN = triton.next_power_of_2(n)
    BQ = triton.next_power_of_2(max(qcols, 1))
    BK = triton.next_power_of_2(max(k, 1))
    sWb, sWm, sWn = W.stride()
    sQb, sQm, sQn = Q.stride()
    sRb, sRm, sRn = R.stride()
    _qr_fused_kernel[(B,)](
        W,
        Q,
        R,
        m,
        n,
        k,
        qcols,
        rrows,
        mode != "r",
        sWb,
        sWm,
        sWn,
        sQb,
        sQm,
        sQn,
        sRb,
        sRm,
        sRn,
        BM=BM,
        BN=BN,
        BQ=BQ,
        BK=BK,
    )
    if mode == "r":
        return (
            out_Q if out_Q is not None else A.new_empty(0),
            R.to(orig_dtype).reshape(*batch_shape, k, n),
        )
    Q_out = Q.to(orig_dtype).reshape(*batch_shape, m, qcols)
    R_out = R.to(orig_dtype).reshape(*batch_shape, rrows, n)
    return (Q_out, R_out)


def _blocked_qr(W, V, tau, Tbuf, m, n, k, ib=_PANEL_IB):
    """In-place blocked Householder QR; leaves R in the upper triangle of W."""
    B = W.shape[0]
    P = (k + ib - 1) // ib
    dt = W.dtype
    dev = W.device
    # _GEQRT_SRAM_MAX_M is calibrated for fp32.  fp64 doubles register pressure
    # (2 regs/elem) and the kernel keeps both the A tile and V_panel live, so
    # the SRAM kernel either spills (BM>=256 -> ~10x slower) or, for small
    # panels, loses to the low-register single-CTA geqrt that reloads from
    # global.  Net: geqrt_sram never wins for fp64 -> disable it.
    sram_max_m = _GEQRT_SRAM_MAX_M if W.element_size() == 4 else 0
    # the SRAM kernel keeps a (BM, IBN) register tile: wider panels (ib=64)
    # double its size, so halve the row cap to stay out of the spill regime.
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
    for kk in range(0, k, ib):
        ib_active = min(ib, k - kk)
        M = m - kk
        bm = triton.next_power_of_2(M)
        if bm <= sram_max_m:
            # panel fits SRAM: single-CTA resident factorisation (no global re-reads)
            _launch_geqrt_sram(W, V, tau, m, k, kk, ib_active, B)
        else:
            # ceil(M/rm) CTAs -> CHUNK == rm -> the register-resident
            # fast path of _geqrt_mcta_kernel (each CTA loads its row chunk
            # once, no per-reflector global re-reads).
            rm = _MCTA_RM if W.element_size() == 4 else _MCTA_RM_FP64
            nc = max(1, min(_MCTA_NC_MAX, (M + rm - 1) // rm))
            if nc >= _MCTA_MIN_NC:
                _launch_geqrt_mcta(
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
        Vp = V[:, kk:m, kk : kk + ib_active]
        taup = tau[:, kk : kk + ib_active]
        Tp = Tbuf[:, kk : kk + ib_active, kk : kk + ib_active]
        if kk + ib_active < n:
            _launch_larft(Vp, taup, Tp, m, kk, ib_active, B)
            C = W[:, kk:m, kk + ib_active : n]
            _launch_larfb(
                Vp, Tp, C, m - kk, n - (kk + ib_active), ib_active, B, upper=False
            )


def _assemble_q(V, tau, Tbuf, m, n, k, qcols, ib, B, out):
    """Q <- (H_0 H_1 ... H_{P-1}) applied to identity; writes into `out` (B, m, qcols).

    T is already in Tbuf from _blocked_qr for every panel except a last panel
    without a trailing update (kk + ib >= n) -- build that one T first.

    fp32 uses the one-pass single-panel kernel when P == 1 (Q = I - V T V^H,
    no Q re-reads), otherwise the fused all-panels kernel; fp64 keeps the
    per-panel larfb path (the fused kernel's runtime-`iba` triangular solve
    cannot be statically specialised and is slower for fp64).
    """
    P = (k + ib - 1) // ib
    kk_last = (P - 1) * ib
    ib_last = min(ib, k - kk_last)
    if kk_last + ib_last >= n:
        Vp = V[:, kk_last:m, kk_last : kk_last + ib_last]
        taup = tau[:, kk_last : kk_last + ib_last]
        Tp = Tbuf[:, kk_last : kk_last + ib_last, kk_last : kk_last + ib_last]
        _launch_larft(Vp, taup, Tp, m, kk_last, ib_last, B)

    if V.element_size() == 4 and P == 1:
        # single panel: Q = I - V T V^H written once (one launch)
        Vp = V[:, :, :k]
        Tp = Tbuf[:, :k, :k]
        sVb, sVm, sVn = Vp.stride()
        sTb, sTm, sTn = Tp.stride()
        sQb, sQm, sQn = out.stride()
        ibn = max(16, triton.next_power_of_2(k))
        grid_p = (qcols + _ASSEMBLE_TN - 1) // _ASSEMBLE_TN
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
    elif V.element_size() == 4:
        # NOTE: grouping panels into wider (ib=64/128) composite blocks built
        # via the larft kernel (Gram + serial inversion) was tried: the long
        # inversion and the (IBN, IBN) register tiles made it a net ~50% LOSS
        # on 4096^2 (measured).  The paired path below composes pairs from the
        # EXISTING per-panel T's instead (no inversion), which is cheap.
        sVb, sVm, sVn = V.stride()
        sTb, sTm, sTn = Tbuf.stride()
        sQb, sQm, sQn = out.stride()
        ib2 = 2 * ib
        npairs = (k + ib2 - 1) // ib2
        paired = _ASSEMBLE_PAIR and P >= 2
        if paired:
            # compose T pairs in place (Tbuf), one launch; halves the number
            # of dependent Q sweeps in the fused kernel below.
            _tcompose_pair_kernel[(B, npairs)](
                V,
                Tbuf,
                m,
                k,
                ib,
                sVb,
                sVm,
                sVn,
                sTb,
                sTm,
                sTn,
                RM=_PANEL_RM,
                IBN=max(32, triton.next_power_of_2(ib2)),
                num_warps=4,
            )
        use_ib, use_P = (ib2, npairs) if paired else (ib, P)
        warps = _ASSEMBLE_PAIR_WARPS if paired else _ASSEMBLE_WARPS
        grid_p = (qcols + _ASSEMBLE_TN - 1) // _ASSEMBLE_TN
        _assemble_q_fused_kernel[(B, grid_p)](
            V,
            Tbuf,
            out,
            m,
            k,
            qcols,
            use_ib,
            use_P,
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
            IBN=max(16, triton.next_power_of_2(use_ib)),
            TN=_ASSEMBLE_TN,
            num_warps=warps,
            **({"num_stages": _ASSEMBLE_STAGES} if _ASSEMBLE_STAGES else {}),
        )
    else:
        # fp64: identity + per-panel larfb (static ib, fast solve path)
        sQb, sQm, sQn = out.stride()
        grid_e = (m * qcols + 1023) // 1024
        _identity_kernel[(B * grid_e,)](
            out, m, qcols, grid_e, sQb, sQm, sQn, BLOCK=1024
        )
        for p in reversed(range(0, k, ib)):
            kk = p
            ib_active = min(ib, k - kk)
            Vp = V[:, kk:m, kk : kk + ib_active]
            Tp = Tbuf[:, kk : kk + ib_active, kk : kk + ib_active]
            _launch_larfb(
                Vp, Tp, out[:, kk:m, :], m - kk, qcols, ib_active, B, upper=True
            )
    return out


def _triu_copy(W, R, rm, n, B):
    sWb, sWm, sWn = W.stride()
    sRb, sRm, sRn = R.stride()
    grid_e = (rm * n + 1023) // 1024
    _triu_copy_kernel[(B, grid_e)](
        W, R, rm, n, sWb, sWm, sWn, sRb, sRm, sRn, BLOCK=1024
    )


# ---------------------------------------------------------------------------
# TSQR fast path (tall-skinny): local QR of row blocks, a tree reduction of the
# stacked R factors, and a reflector-based Q application.  Q is formed by
# applying the stored Householder reflectors (never A @ R^{-1}), so exactly
# rank-deficient inputs yield a valid orthonormal Q and an R with zero rows.
# ---------------------------------------------------------------------------
def _tsqr(W, m, n, k, mode, B, out_Q=None, out_R=None):
    """Returns (Q or None, R).  R is (B, n, n); Q (reduced) is (B, m, n)."""
    dt = W.dtype
    dev = W.device
    IBN = max(16, triton.next_power_of_2(n))
    sWb, sWm, sWn = W.stride()
    # fp64 register tiles cost 2 regs/element, so the single-CTA register
    # budget is 1/4 of the fp32 element cap.
    reg_scale = 4 if W.element_size() == 8 else 1
    sram_elem = _TSQR_SRAM_ELEM // reg_scale
    red_elem = _TSQR_TREE_RED_ELEM // reg_scale
    write_Q = mode != "r"

    # Row-block size: the local kernel keeps a (pow2(br), IBN) register tile.
    # Every block must hold >= n+1 rows (a shorter block could not produce n
    # orthonormal local Q columns), so rebalance until the last block does.
    fin_br = _TSQR_BR if W.element_size() == 4 else _TSQR_BR_FP64
    br = max(n + 1, min(m, fin_br, sram_elem // n))
    num_blocks = (m + br - 1) // br
    while num_blocks > 1 and m - (num_blocks - 1) * br < n + 1:
        num_blocks -= 1
        br = max(n + 1, (m + num_blocks - 1) // num_blocks)
    Rm = num_blocks * n
    esz = W.element_size()

    R_blocks = torch.empty(B, num_blocks, n, n, dtype=dt, device=dev)
    Racc = out_R if out_R is not None else torch.empty(B, n, n, dtype=dt, device=dev)

    # Tree-reduction tiling: the tree kernel has no tl.dot, so its column tile
    # needs no 16-wide minimum -- padding n=4/8 up to 16 would waste 2-4x of
    # the register budget on fp64.  Route: a flat single-CTA reduction when
    # the stack fits the register budget -- except tall fp64 stacks, whose
    # serial per-reflector tile reductions are so slow in one CTA that a
    # two-level tree (parallel group reductions + one top reduction, grp
    # balanced ~sqrt(num_blocks)) wins even when the tile would fit
    # (empirical on H20: fp64 512x8 flat 279us vs two-level ~40us; fp32 flat
    # stays faster up to the budget).  Stacks too tall even for the tree go
    # through the blocked path.  grp (blocks per group) is a power of two so
    # the padded group tile is exactly grp*IBNt x IBNt.
    IBNt = triton.next_power_of_2(n)
    BRM = triton.next_power_of_2(Rm)
    flat = BRM * IBNt <= red_elem and (esz == 4 or BRM <= _TSQR_TREE_FLAT_ROWS)
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
    # Fold the tree reduction into the apply kernel for small fp32 stacks:
    # every CTA redundantly factors the stacked R's in registers (a few tiny
    # reflector steps), saving a whole kernel launch + the Q_t round-trip.
    # fp64 is excluded: its register-tile reductions are ~10x slower, so the
    # redundant per-CTA factorisation costs more than the saved launch.
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
    # W is read-only here -- the caller's input is never scribbled on.
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
        # path (the mcta panels handle any height) and assemble Q_t from its
        # reflectors.  R_blocks is dead after this, so the blocked kernels may
        # factor it in place (Rstack is a view).
        Rstack = R_blocks.reshape(B, Rm, n)
        Vt = (
            torch.zeros(B, Rm, n, dtype=dt, device=dev)
            if write_Q
            else torch.empty(B, Rm, n, dtype=dt, device=dev)
        )
        tau_t = torch.empty(B, n, dtype=dt, device=dev)
        Tbuf_t = torch.empty(B, n, n, dtype=dt, device=dev)
        _blocked_qr(Rstack, Vt, tau_t, Tbuf_t, Rm, n, n)
        _triu_copy(Rstack, Racc, n, n, B)
        if write_Q:
            _assemble_q(Vt, tau_t, Tbuf_t, Rm, n, n, n, _PANEL_IB, B, Qt)
        two_level = False

    if not write_Q:
        return None, Racc

    # ---- Phase 3: Q[block rows] = Q_local @ Q_t[block rows], one CTA/block ----
    _tsqr_apply_kernel[(B, num_blocks)](
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


# ===========================================================================
# Public op
# ===========================================================================
def _validate_mode(mode):
    if mode not in ("reduced", "complete", "r"):
        raise ValueError(
            f"linalg_qr: mode must be one of 'reduced', 'complete', 'r', got {mode!r}"
        )


def _validate_out(out, dtype, batch_shape, m, n, mode):
    """Enforce torch's out= contract: matching dtype and exact shapes.

    torch raises on an out dtype mismatch; on a shape mismatch it resizes
    (deprecated), so a hard error here is the future-proof behaviour -- and
    it catches wrong-but-reshapeable shapes that would silently pass.
    """
    k = min(m, n)
    if mode == "r":
        q_shape, r_shape = (0,), (*batch_shape, k, n)
    elif mode == "reduced":
        q_shape, r_shape = (*batch_shape, m, k), (*batch_shape, k, n)
    else:
        q_shape, r_shape = (*batch_shape, m, m), (*batch_shape, m, n)
    for name, t, shape in (("Q", out[0], q_shape), ("R", out[1], r_shape)):
        if t.dtype != dtype:
            raise RuntimeError(
                f"linalg_qr: expected out tensor {name} to have dtype {dtype}, "
                f"but got {t.dtype}"
            )
        if tuple(t.shape) != tuple(shape):
            raise RuntimeError(
                f"linalg_qr: out tensor {name} has shape {tuple(t.shape)}, "
                f"expected {tuple(shape)}"
            )


def linalg_qr(A, mode="reduced", *, out=None):
    logger.debug("GEMS LINALG_QR")
    return _linalg_qr(A, mode, out=out)


def _linalg_qr(A, mode="reduced", *, out=None):
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
        Qm, Rm = _tsqr(W, m, n, k, mode, B, out_Q, out_R)
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
    _blocked_qr(W, V, tau, Tbuf, m, n, k)

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
    _assemble_q(V, tau, Tbuf, m, n, k, qcols, _PANEL_IB, B, Q)

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


def linalg_qr_out(A, mode="reduced", *, Q, R):
    logger.debug("GEMS LINALG_QR_OUT")
    return _linalg_qr(A, mode, out=(Q, R))
