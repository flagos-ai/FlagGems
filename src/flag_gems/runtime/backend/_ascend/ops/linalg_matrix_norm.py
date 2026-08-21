import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_matrix_norm import (
    _RANK2_BLOCK_R_MAX,
    _fro_kernel,
    _rank2_svals_kernel,
    _svd_shape,
)
from flag_gems.runtime.backend._ascend.utils import CORE_NUM
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# ===========================================================================
# Triton kernels for one-sided Jacobi SVD on Ascend NPU (fp32 only)
# ===========================================================================


@libentry()
@triton.jit
def _onesided_jacobi_svd_kernel(
    A,
    S,
    BATCH: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    MAX_SWEEPS: tl.constexpr,
):
    """Single-program onesided Jacobi for k ≤ 32: all column pairs in one
    kernel launch.  Deterministic on Ascend (single program per batch)."""
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK_R)
    rmask = idx < M
    eps = 1.0e-20
    dtype = tl.float32
    base = A + pid * M * N
    for sweep in range(MAX_SWEEPS):
        for p in range(K - 1):
            for q in range(p + 1, K):
                col_p = tl.load(base + idx * N + p, mask=rmask, other=0.0).to(dtype)
                col_q = tl.load(base + idx * N + q, mask=rmask, other=0.0).to(dtype)
                aa = tl.sum(col_p * col_p) + eps
                bb = tl.sum(col_q * col_q) + eps
                ab = tl.sum(col_p * col_q)
                tau = (bb - aa) / (ab + ab + eps)
                sign_t = tl.where(tau >= 0.0, 1.0, -1.0)
                t = sign_t / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
                c = 1.0 / tl.sqrt(1.0 + t * t)
                s = t * c
                new_p = c * col_p - s * col_q
                new_q = s * col_p + c * col_q
                tl.store(base + idx * N + p, new_p, mask=rmask)
                tl.store(base + idx * N + q, new_q, mask=rmask)
                # MTE3 (store) / MTE2 (load) of one program are unordered on
                # Ascend; fence before the next pair's loads re-read these
                # columns.  (Load-bearing: without it rotations read stale
                # columns and the singular values are wrong.)
                tl.debug_barrier()
    for j in range(K):
        col = tl.load(base + idx * N + j, mask=rmask, other=0.0).to(dtype)
        tl.store(S + pid * K + j, tl.sqrt(tl.sum(col * col) + eps))


@libentry()
@triton.jit
def _jacobi_sweep_kernel(A_WORK, KS, ROWS, CHUNK: tl.constexpr):
    """One full Brent-Luk one-sided Jacobi sweep, single program per matrix.

    KS = K + K % 2 (odd K gets a zero dummy column so ring pairs stay even).
    The j == 0 ring pair is its own block (scalar integer selects miscompile);
    columns load/rotate in CHUNK slices to stay inside UB.  Ring pairs of one
    step are disjoint, so one debug_barrier per step fences MTE3 stores.
    """
    pid = tl.program_id(0)
    dtype = tl.float32
    eps = 1.0e-20
    rounds = KS - 1
    half = KS // 2
    aw = A_WORK + pid * KS * ROWS
    for step in range(rounds):
        # j == 0: the odd column (the dummy column when K is odd, the real
        # last column otherwise).  p = step is already in [0, rounds).
        p = step
        q = rounds
        aa = 0.0
        bb = 0.0
        ab = 0.0
        for chunk_start in range(0, ROWS, CHUNK):
            c_rows = chunk_start + tl.arange(0, CHUNK)
            c_mask = c_rows < ROWS
            cp = tl.load(aw + p * ROWS + c_rows, mask=c_mask, other=0.0).to(dtype)
            cq = tl.load(aw + q * ROWS + c_rows, mask=c_mask, other=0.0).to(dtype)
            aa += tl.sum(cp * cp)
            bb += tl.sum(cq * cq)
            ab += tl.sum(cp * cq)
        tau = (bb - aa) / (ab + ab + eps)
        sign_t = tl.where(tau >= 0.0, 1.0, -1.0)
        t = sign_t / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
        c = tl.rsqrt(1.0 + t * t)
        s = t * c
        for chunk_start in range(0, ROWS, CHUNK):
            c_rows = chunk_start + tl.arange(0, CHUNK)
            c_mask = c_rows < ROWS
            cp = tl.load(aw + p * ROWS + c_rows, mask=c_mask, other=0.0).to(dtype)
            cq = tl.load(aw + q * ROWS + c_rows, mask=c_mask, other=0.0).to(dtype)
            tl.store(aw + p * ROWS + c_rows, c * cp - s * cq, mask=c_mask)
            tl.store(aw + q * ROWS + c_rows, s * cp + c * cq, mask=c_mask)
        for j in range(1, half):
            p = (step + j) % rounds
            q = (step - j + rounds) % rounds
            aa = 0.0
            bb = 0.0
            ab = 0.0
            for chunk_start in range(0, ROWS, CHUNK):
                c_rows = chunk_start + tl.arange(0, CHUNK)
                c_mask = c_rows < ROWS
                cp = tl.load(aw + p * ROWS + c_rows, mask=c_mask, other=0.0).to(dtype)
                cq = tl.load(aw + q * ROWS + c_rows, mask=c_mask, other=0.0).to(dtype)
                aa += tl.sum(cp * cp)
                bb += tl.sum(cq * cq)
                ab += tl.sum(cp * cq)
            tau = (bb - aa) / (ab + ab + eps)
            sign_t = tl.where(tau >= 0.0, 1.0, -1.0)
            t = sign_t / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            c = tl.rsqrt(1.0 + t * t)
            s = t * c
            for chunk_start in range(0, ROWS, CHUNK):
                c_rows = chunk_start + tl.arange(0, CHUNK)
                c_mask = c_rows < ROWS
                cp = tl.load(aw + p * ROWS + c_rows, mask=c_mask, other=0.0).to(dtype)
                cq = tl.load(aw + q * ROWS + c_rows, mask=c_mask, other=0.0).to(dtype)
                tl.store(aw + p * ROWS + c_rows, c * cp - s * cq, mask=c_mask)
                tl.store(aw + q * ROWS + c_rows, s * cp + c * cq, mask=c_mask)
        # ring pairs of one step are disjoint: one fence per step suffices
        tl.debug_barrier()


@libentry()
@triton.jit
def _bidiag_svd_kernel(
    A,
    E2H,
    E2L,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    BATCH: tl.constexpr,
    NPROG: tl.constexpr,
    TRANSPOSED_LOAD: tl.constexpr,
):
    # GK bidiagonalization of the (ROWS, K) tile into a GK seed (written here —
    # a separate gk_init launch would dominate).  The tile loads UNMASKED with
    # clamped addressing (a masked 2D load miscompiles) and pads in-register.
    # Linear domain, so sigma_min keeps full relative precision (the Gram-square
    # route loses it).  Only reshape-based outer products (the only
    # numerically-correct broadcast form); the right reflection takes row j via
    # a (BLOCK, 1) row mask + axis-0 reduction (a (1, BLOCK) column mask
    # silently no-ops).  enable_fp_fusion=False: fusion degrades the _split_f32
    # Veltkamp split to plain fp32.
    for b in range(tl.program_id(0), BATCH, NPROG):
        batch = b
        rows = tl.arange(0, BLOCK)
        cols = tl.arange(0, BLOCK)
        rrow = tl.minimum(rows, ROWS - 1)
        ccol = tl.minimum(cols, K - 1)
        if TRANSPOSED_LOAD == 1:
            g = tl.load(A + batch * (ROWS * K) + ccol[None, :] * ROWS + rrow[:, None])
        else:
            g = tl.load(A + batch * (ROWS * K) + rrow[:, None] * K + ccol[None, :])
        g = tl.where((rows[:, None] < ROWS) & (cols[None, :] < K), g, 0.0)
        gT = tl.trans(g)
        # Plain range, not tl.range: the pipelined-loop lowering silently
        # produces zero output under the cann900 toolchain (triton 3.5 /
        # triton_ascend 3.2.1 + auto-blockify); keep the loop static.
        for j in range(K):
            # j runs to K (not K-1): the final left reflection zeroes column
            # K-1 below the diagonal.  Without it the tile is [B; x] with a
            # nonzero tail when rows > k hold real data (wide/tall inputs),
            # and svd(B) != svd(A).
            # ---- left reflection: zero g[j+1:, j] ----
            colmask = (cols[None, :] > j - 1) & (cols[None, :] < j + 1)
            colj = tl.sum(tl.where(colmask, g, 0.0), axis=1)
            x0 = tl.sum(colj * ((rows > j - 1) & (rows < j + 1)).to(tl.float32), axis=0)
            x = colj * (rows >= j).to(tl.float32)
            sigma = tl.sqrt(tl.sum(x * x, axis=0))
            alpha = tl.where(x0 >= 0.0, -sigma, sigma)
            v2 = tl.where((rows > j - 1) & (rows < j + 1), x0 - alpha, x)
            vnorm2 = 2.0 * sigma * (sigma + tl.abs(x0))
            tau = tl.where(vnorm2 > 0.0, 2.0 / vnorm2, 0.0)
            # w = tau * (g[j:, :]^T v2) via gT (axis-1 reduce, stable)
            w = tau * tl.sum(gT * v2[None, :], axis=1)
            g = g - tl.reshape(v2, (BLOCK, 1)) * tl.reshape(w, (1, BLOCK))
            gT = tl.trans(g)
            # ---- right reflection: zero g[j, j+2:] ----
            # Applied UNCONDITIONALLY (no `if j + 2 < N` guard): under the
            # cann900 toolchain a loop-carried tile yielded through a
            # runtime scf.if on the induction variable miscompiles to
            # all-zero output.  The unguarded form is numerically
            # equivalent: for j = K-1 the row slice g[j, j+1:] is the zero
            # padding, so the reflection is a no-op; for j = K-2 it only
            # sign-flips column K-1 (an orthogonal equivalence), and the
            # GK seed below carries squared values, which are unaffected.
            rowmask = (rows[:, None] > j - 1) & (rows[:, None] < j + 1)
            rowj = tl.sum(tl.where(rowmask, g, 0.0), axis=0)
            u0 = tl.sum(rowj * ((cols > j) & (cols < j + 2)).to(tl.float32), axis=0)
            u = rowj * (cols > j).to(tl.float32)
            sigma2 = tl.sqrt(tl.sum(u * u, axis=0))
            alpha2 = tl.where(u0 >= 0.0, -sigma2, sigma2)
            u2 = tl.where((cols > j) & (cols < j + 2), u0 - alpha2, u)
            vnorm3 = 2.0 * sigma2 * (sigma2 + tl.abs(u0))
            tau2 = tl.where(vnorm3 > 0.0, 2.0 / vnorm3, 0.0)
            # z = tau2 * (g[:, j+1:] u2): axis-1 reduce
            z = tau2 * tl.sum(g * u2[None, :], axis=1)
            g = g - tl.reshape(z, (BLOCK, 1)) * tl.reshape(u2, (1, BLOCK))
            gT = tl.trans(g)
        # ---- fused GK seed: d = diag(g), s = superdiag(g) ----
        # Extracted via where + axis-1 reductions (DSA register extraction
        # crashes on this backend); s[K-1] = g[K-1, K] = 0 from the padding
        # zeroing above.  E2H/E2L hold the same interleaved double-single
        # squared layout the blocked path's _gk_init_kernel produces.
        diagmask = rows[:, None] == cols[None, :]
        supmask = cols[None, :] == rows[:, None] + 1
        dv = tl.sum(tl.where(diagmask, g, 0.0), axis=1)
        sv = tl.sum(tl.where(supmask, g, 0.0), axis=1)
        dh, dl = _split_f32(dv)
        sh, sl = _split_f32(sv)
        d2h, d2l = _df64_mul_ds(dh, dl, dh, dl)
        s2h, s2l = _df64_mul_ds(sh, sl, sh, sl)
        e2base = batch * (2 * K)
        tl.store(E2H + e2base + 2 * rows, d2h, mask=rows < K)
        tl.store(E2L + e2base + 2 * rows, d2l, mask=rows < K)
        tl.store(E2H + e2base + 2 * rows + 1, s2h, mask=rows < K - 1)
        tl.store(E2L + e2base + 2 * rows + 1, s2l, mask=rows < K - 1)


@libentry()
@triton.jit
def _bidiag_svd_rect_kernel(
    A,
    E2H,
    E2L,
    K: tl.constexpr,
    RB: tl.constexpr,
    KB: tl.constexpr,
    ROWS: tl.constexpr,
    BATCH: tl.constexpr,
    NPROG: tl.constexpr,
):
    # Rectangular-tile variant of _bidiag_svd_kernel for rows > 64: tile is
    # (RB, KB) instead of the square (BLOCK, BLOCK), keeping 2*RB*KB under UB
    # for tall/wide shapes.  Same numerics/seed layout as the square kernel.
    # KB is floored at 16 (masked vector ops miscompile with < 8 active lanes).
    for b in range(tl.program_id(0), BATCH, NPROG):
        batch = b
        rr = tl.arange(0, RB)
        cc = tl.arange(0, KB)
        rrow = tl.minimum(rr, ROWS - 1)
        ccol = tl.minimum(cc, K - 1)
        g = tl.load(A + batch * (ROWS * K) + rrow[:, None] * K + ccol[None, :])
        g = tl.where((rr[:, None] < ROWS) & (cc[None, :] < K), g, 0.0)
        gT = tl.trans(g)
        # Same unrolled, unguarded two-sided Householder sweep as the
        # square kernel (see its comment for why the right reflection is
        # applied unconditionally).
        for j in range(K):
            # ---- left reflection: zero g[j+1:, j] ----
            colmask = (cc[None, :] > j - 1) & (cc[None, :] < j + 1)
            colj = tl.sum(tl.where(colmask, g, 0.0), axis=1)
            x0 = tl.sum(colj * ((rr > j - 1) & (rr < j + 1)).to(tl.float32), axis=0)
            x = colj * (rr >= j).to(tl.float32)
            sigma = tl.sqrt(tl.sum(x * x, axis=0))
            alpha = tl.where(x0 >= 0.0, -sigma, sigma)
            v2 = tl.where((rr > j - 1) & (rr < j + 1), x0 - alpha, x)
            vnorm2 = 2.0 * sigma * (sigma + tl.abs(x0))
            tau = tl.where(vnorm2 > 0.0, 2.0 / vnorm2, 0.0)
            w = tau * tl.sum(gT * v2[None, :], axis=1)
            g = g - tl.reshape(v2, (RB, 1)) * tl.reshape(w, (1, KB))
            gT = tl.trans(g)
            # ---- right reflection: zero g[j, j+2:] ----
            rowmask = (rr[:, None] > j - 1) & (rr[:, None] < j + 1)
            rowj = tl.sum(tl.where(rowmask, g, 0.0), axis=0)
            u0 = tl.sum(rowj * ((cc > j) & (cc < j + 2)).to(tl.float32), axis=0)
            u = rowj * (cc > j).to(tl.float32)
            sigma2 = tl.sqrt(tl.sum(u * u, axis=0))
            alpha2 = tl.where(u0 >= 0.0, -sigma2, sigma2)
            u2 = tl.where((cc > j) & (cc < j + 2), u0 - alpha2, u)
            vnorm3 = 2.0 * sigma2 * (sigma2 + tl.abs(u0))
            tau2 = tl.where(vnorm3 > 0.0, 2.0 / vnorm3, 0.0)
            z = tau2 * tl.sum(g * u2[None, :], axis=1)
            g = g - tl.reshape(z, (RB, 1)) * tl.reshape(u2, (1, KB))
            gT = tl.trans(g)
        # ---- fused GK seed: d = diag(g), s = superdiag(g) ----
        diagmask = rr[:, None] == cc[None, :]
        supmask = cc[None, :] == rr[:, None] + 1
        dv = tl.sum(tl.where(diagmask, g, 0.0), axis=1)
        sv = tl.sum(tl.where(supmask, g, 0.0), axis=1)
        dh, dl = _split_f32(dv)
        sh, sl = _split_f32(sv)
        d2h, d2l = _df64_mul_ds(dh, dl, dh, dl)
        s2h, s2l = _df64_mul_ds(sh, sl, sh, sl)
        e2base = batch * (2 * K)
        tl.store(E2H + e2base + 2 * rr, d2h, mask=rr < K)
        tl.store(E2L + e2base + 2 * rr, d2l, mask=rr < K)
        tl.store(E2H + e2base + 2 * rr + 1, s2h, mask=rr < K - 1)
        tl.store(E2L + e2base + 2 * rr + 1, s2l, mask=rr < K - 1)


@libentry()
@triton.jit
def _bidiag_left_step_kernel(
    W,
    K,
    ROWS,
    J,
    BATCH: tl.constexpr,
    NPROG: tl.constexpr,
    CHUNK: tl.constexpr,
    MAX_ROWS: tl.constexpr,
):
    # One GK left Householder step (column j), 1D CHUNK-wide vectors only
    # (non-square 2D tiles miscompile, square tiles overflow UB).  Column j
    # is written only at the end, so there is no intra-kernel store->load
    # round-trip (MTE2/MTE3 reorder unordered; tl.debug_barrier is a no-op;
    # the kernel boundary is the only fence).  Mask lesson: anchor the vector
    # at j with a single upper-bound mask — a two-bound mask on a fixed-offset
    # vector miscompiles below 8 active lanes.  Loop-bound lesson: the chunk
    # loop must have a compile-time trip count (MAX_ROWS constexpr); runtime
    # bounds miscompile when the trip count exceeds 1.
    j = J
    for b in range(tl.program_id(0), BATCH, NPROG):
        w = W + b * K * ROWS
        x0 = tl.load(w + j * ROWS + j)
        sigmasq = 0.0
        for cs in range(0, MAX_ROWS, CHUNK):
            rr = j + cs + tl.arange(0, CHUNK)
            m = rr < ROWS
            x = tl.load(w + j * ROWS + rr, mask=m, other=0.0)
            sigmasq += tl.sum(x * x)
        sigma = tl.sqrt(sigmasq)
        alpha = tl.where(x0 >= 0.0, -sigma, sigma)
        vnorm2 = 2.0 * sigma * (sigma + tl.abs(x0))
        tau = tl.where(vnorm2 > 0.0, 2.0 / vnorm2, 0.0)
        # apply H x_c = x_c - uv * (tau * (uv . x_c)) to columns j+1..K-1,
        # deriving uv from the untouched column j in registers (uv[j] =
        # x0 - alpha, uv[r] = x[r] for r > j).  Each apply writes only its own
        # column, so the c-loop iterations are mutually disjoint and nothing
        # this kernel writes is ever re-read by it.  With rr anchored at j the
        # strict-interval select degenerates to rr < j + 1.
        for c in range(j + 1, K):
            wvc = 0.0
            for cs in range(0, MAX_ROWS, CHUNK):
                rr = j + cs + tl.arange(0, CHUNK)
                m = rr < ROWS
                t = tl.load(w + c * ROWS + rr, mask=m, other=0.0)
                x = tl.load(w + j * ROWS + rr, mask=m, other=0.0)
                one = rr < j + 1
                uv = tl.where(one, x0 - alpha, x)
                wvc += tau * tl.sum(t * uv)
            for cs in range(0, MAX_ROWS, CHUNK):
                rr = j + cs + tl.arange(0, CHUNK)
                m = rr < ROWS
                t = tl.load(w + c * ROWS + rr, mask=m, other=0.0)
                x = tl.load(w + j * ROWS + rr, mask=m, other=0.0)
                one = rr < j + 1
                uv = tl.where(one, x0 - alpha, x)
                tl.store(w + c * ROWS + rr, t - uv * wvc, mask=m)
        # column j itself: H x_j = alpha * e_j (rows < j kept by the mask)
        for cs in range(0, MAX_ROWS, CHUNK):
            rr = j + cs + tl.arange(0, CHUNK)
            m = rr < ROWS
            one = rr < j + 1
            tl.store(w + j * ROWS + rr, tl.where(one, alpha, 0.0), mask=m)


@libentry()
@triton.jit
def _bidiag_right_step_kernel(
    W,
    K,
    ROWS,
    J,
    DO_RIGHT: tl.constexpr,
    BATCH: tl.constexpr,
    NPROG: tl.constexpr,
    CHUNK: tl.constexpr,
    MAX_ROWS: tl.constexpr,
):
    # One GK right Householder step (row j): zero W[j, j+2:] with the
    # alpha-adjusted form (the plain u = x only sign-flips, zeroes nothing).
    # No intra-kernel store->load round-trip; DO_RIGHT is a constexpr guard.
    # Aliasing lesson: the apply must EXCLUDE row j (it holds the reflection
    # vector u; storing it would zero mid-kernel gathers — corrupted
    # rows=1024/2048).  Row j is finalized directly at the end (exact alpha2).
    j = J
    for b in range(tl.program_id(0), BATCH, NPROG):
        w = W + b * K * ROWS
        if DO_RIGHT:
            u0 = tl.load(w + (j + 1) * ROWS + j)
            sigma2sq = u0 * u0
            for c in range(j + 2, K):
                u = tl.load(w + c * ROWS + j)
                sigma2sq += u * u
            sigma2 = tl.sqrt(sigma2sq)
            alpha2 = tl.where(u0 >= 0.0, -sigma2, sigma2)
            vnorm3 = 2.0 * sigma2 * (sigma2 + tl.abs(u0))
            tau2 = tl.where(vnorm3 > 0.0, 2.0 / vnorm3, 0.0)
            uadj = u0 - alpha2
            for cs in range(0, MAX_ROWS, CHUNK):
                # vector anchored at row j+1 (row j excluded — see the
                # aliasing lesson above), single upper-bound mask (two-bound
                # masks on a fixed-offset vector miscompile for < 8 active
                # lanes), compile-time trip count MAX_ROWS (runtime-bounded
                # chunk loops miscompile when the trip count exceeds 1 —
                # see the left kernel's header comment)
                rr = j + 1 + cs + tl.arange(0, CHUNK)
                m = rr < ROWS
                zz = tl.zeros([CHUNK], dtype=tl.float32)
                t = tl.load(w + (j + 1) * ROWS + rr, mask=m, other=0.0)
                zz += t * uadj
                for c in range(j + 2, K):
                    u = tl.load(w + c * ROWS + j)
                    t = tl.load(w + c * ROWS + rr, mask=m, other=0.0)
                    zz += t * u
                zz = tau2 * zz
                t = tl.load(w + (j + 1) * ROWS + rr, mask=m, other=0.0)
                tl.store(w + (j + 1) * ROWS + rr, t - zz * uadj, mask=m)
                for c in range(j + 2, K):
                    u = tl.load(w + c * ROWS + j)
                    t = tl.load(w + c * ROWS + rr, mask=m, other=0.0)
                    tl.store(w + c * ROWS + rr, t - zz * u, mask=m)
            # row j itself: H x_j = alpha2 * e_{j+1} (superdiagonal), zeros
            # beyond — written after all the gathers of row j are done.
            tl.store(w + (j + 1) * ROWS + j, alpha2)
            cvec = j + 2 + tl.arange(0, CHUNK)
            tl.store(
                w + j + cvec * ROWS,
                tl.zeros([CHUNK], dtype=tl.float32),
                mask=cvec < K,
            )


# ===========================================================================
# k > 64 blocked path (Phase 3): grid-parallel bidiagonalization + Sturm
# bisection on the Golub-Kahan tridiagonal.  No CANN, no CPU transfer.
#
# Grid-parallel kernels (one program per column/row chunk) keep every
# kernel at <= 2 tiling parameters, so they compile for any K (the serial
# form's runtime-bounded column loop dies in the autotuner for K > 64).
# The bidiagonalization yields a LOWER bidiagonal B (diagonal + SUBdiagonal);
# the GK tridiagonal of order 2K with off-diagonals [d0, s0, ...] then has
# eigenvalues exactly +/- sigma_i (the superdiagonal silently gives the
# wrong matrix: ~46% sigma error).  The Sturm qd recurrence runs in
# double-single fp32 pairs (BiSheng rejects fp64) with enable_fp_fusion=False
# so the exact-pair identities survive.
# ===========================================================================


@triton.jit
def _df64_add(h1, l1, h2, l2):
    # Error-free addition of two double-single fp32 numbers (Knuth TwoSum).
    s = h1 + h2
    z = s - h1
    e = (h1 - (s - z)) + (h2 - z)
    lo = l1 + l2 + e
    h = s + lo
    e2 = lo - (h - s)
    return h, e2


@triton.jit
def _df64_mul_ds(a_h, a_l, b_h, b_l):
    # Double-single product: TwoProd on the hi parts plus the cross terms.
    p = a_h * b_h
    e = tl.fma(a_h, b_h, -p) + a_h * b_l + a_l * b_h
    h = p + e
    ll = e - (h - p)
    return h, ll


@triton.jit
def _df64_div_ds(a_h, a_l, b_h, b_l):
    # Double-single division: fp32 quotient plus one df64 correction step
    # (avoids slow native fp64 division; fp32 division is a native op).
    q1 = a_h / b_h
    p = q1 * b_h
    pe = tl.fma(q1, b_h, -p)
    r_h, r_l = _df64_add(a_h, a_l, -p, -(pe + q1 * b_l))
    q2 = r_h / b_h
    h = q1 + q2
    ll = q2 - (h - q1)
    return h, ll


@triton.jit
def _split_f32(a):
    # Veltkamp split of one fp32 into an exact hi/lo pair (no fp64 needed).
    # Requires enable_fp_fusion=False -- a contracted t - (t - a) degrades
    # the pair to plain fp32.
    t = 4097.0 * a
    hi = t - (t - a)
    lo = a - hi
    return hi, lo


@triton.jit
def _gk_sturm_count_less(E2H, E2L, base, N: tl.constexpr, xh, xl):
    # #{lambda <= x} for the zero-diagonal GK tridiagonal of order N whose
    # squared off-diagonals are stored as double-single pairs.  D = 0, so
    # p_i = -x - g^2_{i-1}/p_{i-1}; LAPACK DLANEG convention: a zero pivot
    # becomes a tiny negative value (keeps the count consistent for
    # clustered spectra).
    qh, ql = -xh, -xl
    zero_q = (qh == 0.0) & (ql == 0.0)
    qh = tl.where(zero_q, -1.1754944e-38, qh)
    ql = tl.where(zero_q, 0.0, ql)
    neg = tl.where(qh < 0.0, 1, 0)
    # Plain for (N is constexpr), not while: while-loop lowering is
    # unreliable under the cann900 toolchain (see the bidiag note).
    for i in range(1, N):
        e2h = tl.load(E2H + base + i - 1)
        e2l = tl.load(E2L + base + i - 1)
        rh, rl = _df64_div_ds(e2h, e2l, qh, ql)
        qh, ql = _df64_add(-xh, -xl, -rh, -rl)
        zero_q = (qh == 0.0) & (ql == 0.0)
        qh = tl.where(zero_q, -1.1754944e-38, qh)
        ql = tl.where(zero_q, 0.0, ql)
        neg += tl.where(qh < 0.0, 1, 0)
    return neg


@libentry()
@triton.jit
def _bidiag_left_apply_kernel(
    W,
    K,
    ROWS,
    J,
    BATCH: tl.constexpr,
    NPROG: tl.constexpr,
    CHUNK: tl.constexpr,
    MAX_ROWS: tl.constexpr,
):
    # Grid-parallel left reflection: 1D grid <= the AIV block count striding
    # (batch, column slot) items.  The grid must stay 1D (a 2D grid exceeds
    # the block count for batch >= 2 and its auto-map wrapper does not fence
    # MTE3 stores across kernel boundaries -> stale reads).  sigma/alpha/tau
    # use the identical load order/reduction tree so all programs agree
    # bit-for-bit.  Same mask/loop-bound discipline as the Phase 2 kernels.
    WTOT: tl.constexpr = BATCH * NPROG
    for w in range(tl.program_id(0), WTOT, NPROG):
        b = w // NPROG
        cc = w % NPROG
        wbase = W + b * K * ROWS
        j = J
        x0 = tl.load(wbase + j * ROWS + j)
        sigmasq = 0.0
        for cs in range(0, MAX_ROWS, CHUNK):
            rr = j + cs + tl.arange(0, CHUNK)
            m = rr < ROWS
            x = tl.load(wbase + j * ROWS + rr, mask=m, other=0.0)
            sigmasq += tl.sum(x * x)
        sigma = tl.sqrt(sigmasq)
        alpha = tl.where(x0 >= 0.0, -sigma, sigma)
        vnorm2 = 2.0 * sigma * (sigma + tl.abs(x0))
        tau = tl.where(vnorm2 > 0.0, 2.0 / vnorm2, 0.0)
        for c in range(j + 1 + cc, K, NPROG):
            wvc = 0.0
            for cs in range(0, MAX_ROWS, CHUNK):
                rr = j + cs + tl.arange(0, CHUNK)
                m = rr < ROWS
                t = tl.load(wbase + c * ROWS + rr, mask=m, other=0.0)
                x = tl.load(wbase + j * ROWS + rr, mask=m, other=0.0)
                one = rr < j + 1
                uv = tl.where(one, x0 - alpha, x)
                wvc += tau * tl.sum(t * uv)
            for cs in range(0, MAX_ROWS, CHUNK):
                rr = j + cs + tl.arange(0, CHUNK)
                m = rr < ROWS
                t = tl.load(wbase + c * ROWS + rr, mask=m, other=0.0)
                x = tl.load(wbase + j * ROWS + rr, mask=m, other=0.0)
                one = rr < j + 1
                uv = tl.where(one, x0 - alpha, x)
                tl.store(wbase + c * ROWS + rr, t - uv * wvc, mask=m)


@libentry()
@triton.jit
def _bidiag_left_finalize_kernel(
    W,
    K,
    ROWS,
    J,
    BATCH: tl.constexpr,
    NPROG: tl.constexpr,
    CHUNK: tl.constexpr,
    MAX_ROWS: tl.constexpr,
):
    # Row-j workspace finalize: H x_j = alpha * e_j.  Runs after the apply
    # launch; row j is never written by an apply program (no race).  1D grid
    # <= the AIV block count striding the batch elements.
    for b in range(tl.program_id(0), BATCH, NPROG):
        w = W + b * K * ROWS
        j = J
        x0 = tl.load(w + j * ROWS + j)
        sigmasq = 0.0
        for cs in range(0, MAX_ROWS, CHUNK):
            rr = j + cs + tl.arange(0, CHUNK)
            m = rr < ROWS
            x = tl.load(w + j * ROWS + rr, mask=m, other=0.0)
            sigmasq += tl.sum(x * x)
        sigma = tl.sqrt(sigmasq)
        alpha = tl.where(x0 >= 0.0, -sigma, sigma)
        for cs in range(0, MAX_ROWS, CHUNK):
            rr = j + cs + tl.arange(0, CHUNK)
            m = rr < ROWS
            one = rr < j + 1
            tl.store(w + j * ROWS + rr, tl.where(one, alpha, 0.0), mask=m)


@libentry()
@triton.jit
def _bidiag_right_apply_kernel(
    W,
    K,
    ROWS,
    J,
    BATCH: tl.constexpr,
    NPROG: tl.constexpr,
    NCHUNKS: tl.constexpr,
    CHUNK: tl.constexpr,
    MAX_ROWS: tl.constexpr,
):
    # Grid-parallel right reflection: 1D grid <= the AIV block count
    # striding the (batch element, row chunk) work items.  Item (b, cs)
    # applies the step-j reflection to workspace columns rr of its chunk
    # (rr >= j+1; row j holds the reflection vector and must not be written
    # while gathers of it are pending).  u (column j below j+1) is read once
    # per program via scalar gathers.
    WTOT: tl.constexpr = BATCH * NCHUNKS
    for w in range(tl.program_id(0), WTOT, NPROG):
        b = w // NCHUNKS
        wbase = W + b * K * ROWS
        j = J
        u0 = tl.load(wbase + (j + 1) * ROWS + j)
        sigma2sq = u0 * u0
        for c in range(j + 2, K):
            u = tl.load(wbase + c * ROWS + j)
            sigma2sq += u * u
        sigma2 = tl.sqrt(sigma2sq)
        alpha2 = tl.where(u0 >= 0.0, -sigma2, sigma2)
        vnorm3 = 2.0 * sigma2 * (sigma2 + tl.abs(u0))
        tau2 = tl.where(vnorm3 > 0.0, 2.0 / vnorm3, 0.0)
        uadj = u0 - alpha2
        cs = (w % NCHUNKS) * CHUNK
        rr = j + 1 + cs + tl.arange(0, CHUNK)
        m = rr < ROWS
        zz = tl.zeros([CHUNK], dtype=tl.float32)
        t = tl.load(wbase + (j + 1) * ROWS + rr, mask=m, other=0.0)
        zz += t * uadj
        for c in range(j + 2, K):
            u = tl.load(wbase + c * ROWS + j)
            t = tl.load(wbase + c * ROWS + rr, mask=m, other=0.0)
            zz += t * u
        zz = tau2 * zz
        t = tl.load(wbase + (j + 1) * ROWS + rr, mask=m, other=0.0)
        tl.store(wbase + (j + 1) * ROWS + rr, t - zz * uadj, mask=m)
        for c in range(j + 2, K):
            u = tl.load(wbase + c * ROWS + j)
            t = tl.load(wbase + c * ROWS + rr, mask=m, other=0.0)
            tl.store(wbase + c * ROWS + rr, t - zz * u, mask=m)


@libentry()
@triton.jit
def _bidiag_right_finalize_kernel(
    W, K, ROWS, J, BATCH: tl.constexpr, NPROG: tl.constexpr, CHUNK: tl.constexpr
):
    # Row-j finalize of the right reflection: W[j+1, j] = alpha2 and
    # column j below the subdiagonal zeroed -- written directly (exact
    # alpha2, no cancellation through t - zz*u) after all gathers of row j
    # and column j are done.  1D grid <= the AIV block count striding the
    # batch elements.
    for b in range(tl.program_id(0), BATCH, NPROG):
        w = W + b * K * ROWS
        j = J
        u0 = tl.load(w + (j + 1) * ROWS + j)
        sigma2sq = u0 * u0
        for c in range(j + 2, K):
            u = tl.load(w + c * ROWS + j)
            sigma2sq += u * u
        sigma2 = tl.sqrt(sigma2sq)
        alpha2 = tl.where(u0 >= 0.0, -sigma2, sigma2)
        tl.store(w + (j + 1) * ROWS + j, alpha2)
        cvec = j + 2 + tl.arange(0, CHUNK)
        tl.store(
            w + j + cvec * ROWS, tl.zeros([CHUNK], dtype=tl.float32), mask=cvec < K
        )


@libentry()
@triton.jit
def _gk_init_kernel(
    W,
    E2H,
    E2L,
    ROWS,
    K: tl.constexpr,
    BATCH: tl.constexpr,
    NPROG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # GK tridiagonal of order N = 2K for the lower-bidiagonal corner B:
    # zero diagonal, off-diagonal [d0, s0, ...] (d = diagonal, s = SUBdiagonal
    # of B), eigenvalues exactly +/- sigma_i, stored as interleaved
    # double-single pairs.  BLOCK lanes are reused across chunks (BLOCK = N
    # overflows UB under TRITON_ALL_BLOCKS_PARALLEL's multi-buffer options).
    N: tl.constexpr = 2 * K
    for b in range(tl.program_id(0), BATCH, NPROG):
        base = b * N
        wb = W + b * K * ROWS
        idx = tl.arange(0, BLOCK)
        for cs in range(0, N, 2 * BLOCK):
            i = cs // 2 + idx
            dv = tl.load(wb + i * ROWS + i, mask=i < K, other=0.0)
            sv = tl.load(wb + (i + 1) * ROWS + i, mask=i < K - 1, other=0.0)
            dh, dl = _split_f32(dv)
            sh, sl = _split_f32(sv)
            d2h, d2l = _df64_mul_ds(dh, dl, dh, dl)
            s2h, s2l = _df64_mul_ds(sh, sl, sh, sl)
            tl.store(E2H + base + cs + 2 * idx, d2h, mask=i < K)
            tl.store(E2L + base + cs + 2 * idx, d2l, mask=i < K)
            tl.store(E2H + base + cs + 2 * idx + 1, s2h, mask=i < K - 1)
            tl.store(E2L + base + cs + 2 * idx + 1, s2l, mask=i < K - 1)


@libentry()
@triton.jit
def _sturm_sigmas_kernel(
    E2H,
    E2L,
    S,
    K: tl.constexpr,
    NPROG: tl.constexpr,
    BISECT_ITERS: tl.constexpr,
    N: tl.constexpr,
    BATCH: tl.constexpr,
):
    # 1D grid <= the AIV block count striding (batch, column) items (a 2D
    # grid races with the dependent reduction reading S).  Item (b, cc)
    # bisects eigenvalues K+j (ascending: S runs sigma_min..sigma_max, zeros
    # front for rank-deficient).  lo = 0 is a valid lower bound for every
    # target; hi = 2*max|g| (Gershgorin).  Scalar-only bisection loop.
    WTOT: tl.constexpr = BATCH * NPROG
    for w in range(tl.program_id(0), WTOT, NPROG):
        b = w // NPROG
        cc = w % NPROG
        base = b * N
        # emax = max|g| over the whole sequence, as a SCALAR max chain:
        # consecutive scalar loads pipeline, while a BLOCK-lane masked
        # vector load + tl.max reduction serializes ~8x when many
        # programs run concurrently under the multi-buffer compile
        # options (measured: the k=32 sturm dropped 4.8 -> 0.8 ms).
        emax = 0.0
        for i in range(1, N):
            emax = tl.maximum(emax, tl.load(E2H + base + i - 1))
        emax = tl.sqrt(emax)
        JMAX: tl.constexpr = (K + NPROG - 1) // NPROG
        # Plain range, not tl.range (see the bidiag note).
        for jj in range(JMAX):
            # j = jj*NPROG + cc clamped to K-1.  A dynamic loop START
            # (range(cc, K, NPROG)) serializes the qd chain ~4x slower;
            # the clamp keeps the store in-bounds without a runtime branch
            # around it (miscompiles).  Programs whose j exceeds K-1
            # recompute the SAME j = K-1 bisection deterministically and
            # store the identical value — redundant but benign.
            j = tl.minimum(jj * NPROG + cc, K - 1)
            # hi AND lo are re-initialized per j: the bisection rewrites
            # both, so a carry-over hi would confine later j's to
            # [0, ~sigma_prev].
            lo = 0.0
            hi = 2.0 * emax * (1.0 + 1e-9) + 1e-292
            target = K + j
            # Plain for (BISECT_ITERS is constexpr), not while (see the
            # bidiag note).
            for it in range(BISECT_ITERS):
                mid = 0.5 * (lo + hi)
                xh, xl = _split_f32(mid)
                cnt = _gk_sturm_count_less(E2H, E2L, base, N, xh, xl)
                if cnt >= target + 1:
                    hi = mid
                else:
                    lo = mid
            tl.store(S + b * K + j, 0.5 * (lo + hi))


@libentry()
@triton.jit
def _dim1_reduce_kernel(
    S,
    O,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    BATCH: tl.constexpr,
    NPROG: tl.constexpr,
):
    # Max/min/sum over the last dim of S (batch, K) in one fast launch (torch
    # ops/dispatchers cost ~0.17 ms each — dominant at tiny shapes).  1D grid
    # <= the AIV block count striding batch elements (S's producer is
    # grid-flattened).  K <= 512.
    idx = tl.arange(0, BLOCK)
    mask = idx < K
    for b in range(tl.program_id(0), BATCH, NPROG):
        if MODE == 0:  # max (sigmas >= 0, other=-1 is neutral)
            v = tl.load(S + b * K + idx, mask=mask, other=-1.0)
            r = tl.max(v)
        elif MODE == 1:  # min
            v = tl.load(S + b * K + idx, mask=mask, other=3.4028235e38)
            r = tl.min(v)
        else:  # sum
            v = tl.load(S + b * K + idx, mask=mask, other=0.0)
            r = tl.sum(v)
        tl.store(O + b, r)


def _dim1_reduce(S, mode):
    """Pure-Triton max/min/sum over S.shape[-1] (k >= 2)."""
    batch = S.numel() // S.shape[-1]
    k = S.shape[-1]
    nprog = _block_parallel_programs()
    s = S.contiguous().reshape(batch, k)
    o = torch.empty(batch, dtype=s.dtype, device=s.device)
    _dim1_reduce_kernel[(min(nprog, batch),)](
        s,
        o,
        K=k,
        BLOCK=triton.next_power_of_2(k),
        MODE={"max": 0, "min": 1, "sum": 2}[mode],
        BATCH=batch,
        NPROG=nprog,
        num_warps=1,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return o.reshape(S.shape[:-1])


@libentry()
@triton.jit
def _rank2_svals_norm_kernel(
    A,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    TALL: tl.constexpr,
    BLOCK_R: tl.constexpr,
    MODE: tl.constexpr,
    BATCH: tl.constexpr,
    NPROG: tl.constexpr,
):
    # Closed-form k=2 singular values with the norm reduction fused in (both
    # sigmas are in-register scalars).  A separate reduce launch would dominate
    # the launch-bound (2, 2048) / (2048, 2) shapes.
    eps = 1.0e-20
    offs = tl.arange(0, BLOCK_R)
    for b in range(tl.program_id(0), BATCH, NPROG):
        if TALL:
            mask = offs < M
            base = A + b * M * N
            x = tl.load(base + offs * N, mask=mask, other=0.0)
            y = tl.load(base + offs * N + 1, mask=mask, other=0.0)
        else:
            mask = offs < N
            base = A + b * M * N
            x = tl.load(base + offs, mask=mask, other=0.0)
            y = tl.load(base + N + offs, mask=mask, other=0.0)
        aa = tl.sum(x * x)
        bbv = tl.sum(y * y)
        ab = tl.sum(x * y)
        diff = aa - bbv
        root = tl.sqrt(diff * diff + 4.0 * ab * ab)
        l0 = tl.maximum(0.0, 0.5 * (aa + bbv + root))
        det = tl.maximum(0.0, aa * bbv - ab * ab)
        l1 = tl.where(l0 > eps, det / l0, 0.0)
        s0 = tl.sqrt(l0)
        s1 = tl.sqrt(l1)
        if MODE == 0:  # max
            r = tl.maximum(s0, s1)
        elif MODE == 1:  # min
            r = tl.minimum(s0, s1)
        else:  # sum
            r = s0 + s1
        tl.store(O + b, r)


def _rank2_norm_fast(A, mode):
    """k=2 norm (max/min/sum of the two singular values) — one fast
    launch, fused reduction."""
    if A.dtype in (torch.float16, torch.bfloat16):
        A = A.float()
    *batch_dims, M, N = A.shape
    batch = 1
    for d in batch_dims:
        batch *= d
    a = A.contiguous().reshape(batch, M, N)
    o = torch.empty(batch, dtype=torch.float32, device=A.device)
    block_r = triton.next_power_of_2(max(M, N))
    nprog = _block_parallel_programs()
    _rank2_svals_norm_kernel[(min(nprog, batch),)](
        a,
        o,
        M=M,
        N=N,
        TALL=M >= N,
        BLOCK_R=block_r,
        MODE={"max": 0, "min": 1, "sum": 2}[mode],
        BATCH=batch,
        NPROG=nprog,
        num_warps=1 if block_r <= 64 else 4,
    )
    return o.reshape(batch_dims)


@libentry()
@triton.jit
def _rank3_svals_norm_kernel(
    A,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    TALL: tl.constexpr,
    BLOCK_R: tl.constexpr,
    MODE: tl.constexpr,
    BATCH: tl.constexpr,
    NPROG: tl.constexpr,
):
    # Closed-form k=3 norm (max/min/sum of the three singular values) in one
    # launch.  G = A^T A (tall) / A A^T (wide) is a 3x3 symmetric matrix; its
    # eigenvalues come from Kopp's analytic symmetric-3x3 eigendecomposition
    # (no Sturm, no seed round-trip -> no second launch), then sqrt.  Replaces
    # the 2-launch bidiag+sturm _svd_norm_tiny path at launch-bound k=3 shapes.
    eps = 1.0e-20
    offs = tl.arange(0, BLOCK_R)
    for b in range(tl.program_id(0), BATCH, NPROG):
        if TALL:
            mask = offs < M
            base = A + b * M * N
            x = tl.load(base + offs * N, mask=mask, other=0.0)
            y = tl.load(base + offs * N + 1, mask=mask, other=0.0)
            z = tl.load(base + offs * N + 2, mask=mask, other=0.0)
        else:
            mask = offs < N
            base = A + b * M * N
            x = tl.load(base + offs, mask=mask, other=0.0)
            y = tl.load(base + N + offs, mask=mask, other=0.0)
            z = tl.load(base + 2 * N + offs, mask=mask, other=0.0)
        g00 = tl.sum(x * x)
        g01 = tl.sum(x * y)
        g02 = tl.sum(x * z)
        g11 = tl.sum(y * y)
        g12 = tl.sum(y * z)
        g22 = tl.sum(z * z)
        # Kopp (2008) analytic symmetric-3x3 eigenvalues, sorted eig0>=eig1>=eig2.
        q = (g00 + g11 + g22) / 3.0
        p1 = g01 * g01 + g02 * g02 + g12 * g12
        p2 = (
            (g00 - q) * (g00 - q)
            + (g11 - q) * (g11 - q)
            + (g22 - q) * (g22 - q)
            + 2.0 * p1
        )
        p = tl.sqrt(p2 / 6.0)
        sp = tl.maximum(p, eps)
        b00 = (g00 - q) / sp
        b01 = g01 / sp
        b02 = g02 / sp
        b11 = (g11 - q) / sp
        b12 = g12 / sp
        b22 = (g22 - q) / sp
        r = (
            b00 * (b11 * b22 - b12 * b12)
            - b01 * (b01 * b22 - b12 * b02)
            + b02 * (b01 * b12 - b11 * b02)
        ) / 2.0
        r = tl.minimum(tl.maximum(r, -1.0), 1.0)
        # theta = acos(r) via atan2 (acos is not a native tl op here).
        theta = tl.atan2(tl.sqrt(tl.maximum(0.0, 1.0 - r * r)), r)
        phi = theta / 3.0
        eig0 = q + 2.0 * sp * tl.cos(phi)
        eig2 = q + 2.0 * sp * tl.cos(phi + 2.0943951023931953)  # 2*pi/3
        eig1 = 3.0 * q - eig0 - eig2
        s0 = tl.sqrt(tl.maximum(0.0, eig0))
        s1 = tl.sqrt(tl.maximum(0.0, eig1))
        s2 = tl.sqrt(tl.maximum(0.0, eig2))
        if MODE == 0:  # max
            r = tl.maximum(tl.maximum(s0, s1), s2)
        elif MODE == 1:  # min
            r = tl.minimum(tl.minimum(s0, s1), s2)
        else:  # sum
            r = s0 + s1 + s2
        tl.store(O + b, r)


_RANK3_BLOCK_R_MAX = 2048


def _rank3_norm_fast(A, mode):
    """k=3 norm (max/min/sum of the three singular values) — one fused
    launch with the analytic 3x3 eigendecomposition."""
    if A.dtype in (torch.float16, torch.bfloat16):
        A = A.float()
    *batch_dims, M, N = A.shape
    batch = 1
    for d in batch_dims:
        batch *= d
    a = A.contiguous().reshape(batch, M, N)
    o = torch.empty(batch, dtype=torch.float32, device=A.device)
    block_r = triton.next_power_of_2(max(M, N))
    nprog = _block_parallel_programs()
    _rank3_svals_norm_kernel[(min(nprog, batch),)](
        a,
        o,
        M=M,
        N=N,
        TALL=M >= N,
        BLOCK_R=block_r,
        MODE={"max": 0, "min": 1, "sum": 2}[mode],
        BATCH=batch,
        NPROG=nprog,
        num_warps=1 if block_r <= 64 else 4,
    )
    return o.reshape(batch_dims)


_JACOBI8_SWEEPS = 6


@libentry()
@triton.jit
def _jacobi8_norm_kernel(
    A,
    O,
    ROWS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    LDS: tl.constexpr,
    TALL: tl.constexpr,
    MODE: tl.constexpr,
    SWEEPS: tl.constexpr,
    BATCH: tl.constexpr,
    NPROG: tl.constexpr,
    TOTAL: tl.constexpr,
):
    """Register-resident one-sided Jacobi for k=8 (single launch, no work
    buffer, no input mutation).  Columns stay in registers across all
    sweeps, so no store/reload fence is needed (unlike the in-place
    _onesided_jacobi_svd_kernel).  For tall (rows x 8) columns are the 8
    columns of A; for wide (8 x rows) they are the 8 rows of A (i.e. columns
    of A^T).  Returns max/min/sum of the 8 singular values directly."""
    eps = 1.0e-20
    idx = tl.arange(0, BLOCK_R)
    rmask = idx < ROWS
    for b in range(tl.program_id(0), BATCH, NPROG):
        base = A + b * TOTAL
        if TALL:
            c0 = tl.load(base + idx * LDS + 0, mask=rmask, other=0.0)
            c1 = tl.load(base + idx * LDS + 1, mask=rmask, other=0.0)
            c2 = tl.load(base + idx * LDS + 2, mask=rmask, other=0.0)
            c3 = tl.load(base + idx * LDS + 3, mask=rmask, other=0.0)
            c4 = tl.load(base + idx * LDS + 4, mask=rmask, other=0.0)
            c5 = tl.load(base + idx * LDS + 5, mask=rmask, other=0.0)
            c6 = tl.load(base + idx * LDS + 6, mask=rmask, other=0.0)
            c7 = tl.load(base + idx * LDS + 7, mask=rmask, other=0.0)
        else:
            c0 = tl.load(base + 0 * LDS + idx, mask=rmask, other=0.0)
            c1 = tl.load(base + 1 * LDS + idx, mask=rmask, other=0.0)
            c2 = tl.load(base + 2 * LDS + idx, mask=rmask, other=0.0)
            c3 = tl.load(base + 3 * LDS + idx, mask=rmask, other=0.0)
            c4 = tl.load(base + 4 * LDS + idx, mask=rmask, other=0.0)
            c5 = tl.load(base + 5 * LDS + idx, mask=rmask, other=0.0)
            c6 = tl.load(base + 6 * LDS + idx, mask=rmask, other=0.0)
            c7 = tl.load(base + 7 * LDS + idx, mask=rmask, other=0.0)
        for sweep in range(SWEEPS):
            aa = tl.sum(c0 * c0)
            bb = tl.sum(c1 * c1)
            ab = tl.sum(c0 * c1)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c0 = cc * c0 - ss * c1
            new_c1 = ss * c0 + cc * c1
            c0 = new_c0
            c1 = new_c1
            aa = tl.sum(c0 * c0)
            bb = tl.sum(c2 * c2)
            ab = tl.sum(c0 * c2)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c0 = cc * c0 - ss * c2
            new_c2 = ss * c0 + cc * c2
            c0 = new_c0
            c2 = new_c2
            aa = tl.sum(c0 * c0)
            bb = tl.sum(c3 * c3)
            ab = tl.sum(c0 * c3)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c0 = cc * c0 - ss * c3
            new_c3 = ss * c0 + cc * c3
            c0 = new_c0
            c3 = new_c3
            aa = tl.sum(c0 * c0)
            bb = tl.sum(c4 * c4)
            ab = tl.sum(c0 * c4)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c0 = cc * c0 - ss * c4
            new_c4 = ss * c0 + cc * c4
            c0 = new_c0
            c4 = new_c4
            aa = tl.sum(c0 * c0)
            bb = tl.sum(c5 * c5)
            ab = tl.sum(c0 * c5)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c0 = cc * c0 - ss * c5
            new_c5 = ss * c0 + cc * c5
            c0 = new_c0
            c5 = new_c5
            aa = tl.sum(c0 * c0)
            bb = tl.sum(c6 * c6)
            ab = tl.sum(c0 * c6)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c0 = cc * c0 - ss * c6
            new_c6 = ss * c0 + cc * c6
            c0 = new_c0
            c6 = new_c6
            aa = tl.sum(c0 * c0)
            bb = tl.sum(c7 * c7)
            ab = tl.sum(c0 * c7)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c0 = cc * c0 - ss * c7
            new_c7 = ss * c0 + cc * c7
            c0 = new_c0
            c7 = new_c7
            aa = tl.sum(c1 * c1)
            bb = tl.sum(c2 * c2)
            ab = tl.sum(c1 * c2)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c1 = cc * c1 - ss * c2
            new_c2 = ss * c1 + cc * c2
            c1 = new_c1
            c2 = new_c2
            aa = tl.sum(c1 * c1)
            bb = tl.sum(c3 * c3)
            ab = tl.sum(c1 * c3)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c1 = cc * c1 - ss * c3
            new_c3 = ss * c1 + cc * c3
            c1 = new_c1
            c3 = new_c3
            aa = tl.sum(c1 * c1)
            bb = tl.sum(c4 * c4)
            ab = tl.sum(c1 * c4)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c1 = cc * c1 - ss * c4
            new_c4 = ss * c1 + cc * c4
            c1 = new_c1
            c4 = new_c4
            aa = tl.sum(c1 * c1)
            bb = tl.sum(c5 * c5)
            ab = tl.sum(c1 * c5)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c1 = cc * c1 - ss * c5
            new_c5 = ss * c1 + cc * c5
            c1 = new_c1
            c5 = new_c5
            aa = tl.sum(c1 * c1)
            bb = tl.sum(c6 * c6)
            ab = tl.sum(c1 * c6)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c1 = cc * c1 - ss * c6
            new_c6 = ss * c1 + cc * c6
            c1 = new_c1
            c6 = new_c6
            aa = tl.sum(c1 * c1)
            bb = tl.sum(c7 * c7)
            ab = tl.sum(c1 * c7)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c1 = cc * c1 - ss * c7
            new_c7 = ss * c1 + cc * c7
            c1 = new_c1
            c7 = new_c7
            aa = tl.sum(c2 * c2)
            bb = tl.sum(c3 * c3)
            ab = tl.sum(c2 * c3)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c2 = cc * c2 - ss * c3
            new_c3 = ss * c2 + cc * c3
            c2 = new_c2
            c3 = new_c3
            aa = tl.sum(c2 * c2)
            bb = tl.sum(c4 * c4)
            ab = tl.sum(c2 * c4)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c2 = cc * c2 - ss * c4
            new_c4 = ss * c2 + cc * c4
            c2 = new_c2
            c4 = new_c4
            aa = tl.sum(c2 * c2)
            bb = tl.sum(c5 * c5)
            ab = tl.sum(c2 * c5)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c2 = cc * c2 - ss * c5
            new_c5 = ss * c2 + cc * c5
            c2 = new_c2
            c5 = new_c5
            aa = tl.sum(c2 * c2)
            bb = tl.sum(c6 * c6)
            ab = tl.sum(c2 * c6)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c2 = cc * c2 - ss * c6
            new_c6 = ss * c2 + cc * c6
            c2 = new_c2
            c6 = new_c6
            aa = tl.sum(c2 * c2)
            bb = tl.sum(c7 * c7)
            ab = tl.sum(c2 * c7)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c2 = cc * c2 - ss * c7
            new_c7 = ss * c2 + cc * c7
            c2 = new_c2
            c7 = new_c7
            aa = tl.sum(c3 * c3)
            bb = tl.sum(c4 * c4)
            ab = tl.sum(c3 * c4)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c3 = cc * c3 - ss * c4
            new_c4 = ss * c3 + cc * c4
            c3 = new_c3
            c4 = new_c4
            aa = tl.sum(c3 * c3)
            bb = tl.sum(c5 * c5)
            ab = tl.sum(c3 * c5)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c3 = cc * c3 - ss * c5
            new_c5 = ss * c3 + cc * c5
            c3 = new_c3
            c5 = new_c5
            aa = tl.sum(c3 * c3)
            bb = tl.sum(c6 * c6)
            ab = tl.sum(c3 * c6)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c3 = cc * c3 - ss * c6
            new_c6 = ss * c3 + cc * c6
            c3 = new_c3
            c6 = new_c6
            aa = tl.sum(c3 * c3)
            bb = tl.sum(c7 * c7)
            ab = tl.sum(c3 * c7)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c3 = cc * c3 - ss * c7
            new_c7 = ss * c3 + cc * c7
            c3 = new_c3
            c7 = new_c7
            aa = tl.sum(c4 * c4)
            bb = tl.sum(c5 * c5)
            ab = tl.sum(c4 * c5)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c4 = cc * c4 - ss * c5
            new_c5 = ss * c4 + cc * c5
            c4 = new_c4
            c5 = new_c5
            aa = tl.sum(c4 * c4)
            bb = tl.sum(c6 * c6)
            ab = tl.sum(c4 * c6)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c4 = cc * c4 - ss * c6
            new_c6 = ss * c4 + cc * c6
            c4 = new_c4
            c6 = new_c6
            aa = tl.sum(c4 * c4)
            bb = tl.sum(c7 * c7)
            ab = tl.sum(c4 * c7)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c4 = cc * c4 - ss * c7
            new_c7 = ss * c4 + cc * c7
            c4 = new_c4
            c7 = new_c7
            aa = tl.sum(c5 * c5)
            bb = tl.sum(c6 * c6)
            ab = tl.sum(c5 * c6)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c5 = cc * c5 - ss * c6
            new_c6 = ss * c5 + cc * c6
            c5 = new_c5
            c6 = new_c6
            aa = tl.sum(c5 * c5)
            bb = tl.sum(c7 * c7)
            ab = tl.sum(c5 * c7)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c5 = cc * c5 - ss * c7
            new_c7 = ss * c5 + cc * c7
            c5 = new_c5
            c7 = new_c7
            aa = tl.sum(c6 * c6)
            bb = tl.sum(c7 * c7)
            ab = tl.sum(c6 * c7)
            tau = (bb - aa) / (ab + ab + eps)
            st = tl.where(tau >= 0.0, 1.0, -1.0)
            t = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
            cc = 1.0 / tl.sqrt(1.0 + t * t)
            ss = t * cc
            new_c6 = cc * c6 - ss * c7
            new_c7 = ss * c6 + cc * c7
            c6 = new_c6
            c7 = new_c7
        s0 = tl.sqrt(tl.sum(c0 * c0) + eps)
        s1 = tl.sqrt(tl.sum(c1 * c1) + eps)
        s2 = tl.sqrt(tl.sum(c2 * c2) + eps)
        s3 = tl.sqrt(tl.sum(c3 * c3) + eps)
        s4 = tl.sqrt(tl.sum(c4 * c4) + eps)
        s5 = tl.sqrt(tl.sum(c5 * c5) + eps)
        s6 = tl.sqrt(tl.sum(c6 * c6) + eps)
        s7 = tl.sqrt(tl.sum(c7 * c7) + eps)
        if MODE == 0:  # max
            r = tl.maximum(
                tl.maximum(tl.maximum(s0, s1), tl.maximum(s2, s3)),
                tl.maximum(tl.maximum(s4, s5), tl.maximum(s6, s7)),
            )
        elif MODE == 1:  # min
            r = tl.minimum(
                tl.minimum(tl.minimum(s0, s1), tl.minimum(s2, s3)),
                tl.minimum(tl.minimum(s4, s5), tl.minimum(s6, s7)),
            )
        else:  # sum
            r = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
        tl.store(O + b, r)


def _jacobi8_norm_fast(A, mode):
    """k=8 norm (max/min/sum of the eight singular values) — one fused
    one-sided-Jacobi launch, replacing the 3-launch Gram path for
    launch-bound shapes like (256, 8)."""
    if A.dtype in (torch.float16, torch.bfloat16):
        A = A.float()
    *batch_dims, M, N = A.shape
    batch = 1
    for d in batch_dims:
        batch *= d
    a = A.contiguous().reshape(batch, M, N)
    k = min(M, N)
    rows = max(M, N)
    tall = M >= N
    o = torch.empty(batch, dtype=torch.float32, device=A.device)
    block_r = triton.next_power_of_2(rows)
    lds = k if tall else rows
    nprog = _block_parallel_programs()
    _jacobi8_norm_kernel[(min(nprog, batch),)](
        a,
        o,
        ROWS=rows,
        BLOCK_R=block_r,
        LDS=lds,
        TALL=tall,
        MODE={"max": 0, "min": 1, "sum": 2}[mode],
        SWEEPS=_JACOBI8_SWEEPS,
        BATCH=batch,
        NPROG=nprog,
        TOTAL=M * N,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return o.reshape(batch_dims)


# ===========================================================================
# SVD dispatch (pure Triton — no torch.linalg, no CANN, no CPU transfer)
# ===========================================================================


def _rank2_svals_fast(input):
    """Closed-form k=2 singular values in one kernel launch (the k=2
    benchmark shapes are launch-bound)."""
    batch, m, n = _svd_shape(input)
    a = input.contiguous().reshape(batch, m, n)
    s = torch.empty((batch, 2), dtype=input.dtype, device=input.device)
    largest = max(m, n)
    block_r = triton.next_power_of_2(largest)
    if largest <= 16 and batch >= 16:
        block_b = 2 if largest <= 2 else (2 if m >= n else 8) if largest == 16 else 16
        _rank2_svals_kernel[(triton.cdiv(batch, block_b),)](
            a,
            s,
            BATCH=batch,
            M=m,
            N=n,
            TALL=m >= n,
            BLOCK_B=block_b,
            BLOCK_R=block_r,
            num_warps=1,
        )
    else:
        _rank2_svals_kernel[(batch,)](
            a,
            s,
            BATCH=batch,
            M=m,
            N=n,
            TALL=m >= n,
            BLOCK_B=1,
            BLOCK_R=block_r,
            num_warps=1 if block_r <= 64 else 4,
        )
    return s.reshape(*input.shape[:-2], 2)


_TINY_MAX_K = 16
_TINY_MAX_ROWS = 64


@libentry()
@triton.jit
def _sturm_norm_kernel(
    E2H,
    E2L,
    O,
    K: tl.constexpr,
    BISECT_ITERS: tl.constexpr,
    N: tl.constexpr,
    MODE: tl.constexpr,
):
    # One program per batch element: the K Sturm chains run SERIALLY, with
    # the norm reduction fused into the accumulator (a separate reduce launch
    # would dominate).  Same verified pattern as _sturm_sigmas_kernel, but the
    # per-chain store becomes an accumulator update.  NOTE: a register-resident
    # variant (equality-mask seed extraction) crashes with unaligned UB
    # accesses — reduction-of-a-reduction extractions miscompile.
    b = tl.program_id(0)
    base = b * N
    # emax = max|g| over the whole seed sequence, scalar max chain (see
    # the sturm kernel note on scalar vs vector loads).
    emax = 0.0
    for i in range(1, N):
        emax = tl.maximum(emax, tl.load(E2H + base + i - 1))
    emax = tl.sqrt(emax)
    if MODE == 0:  # max (sigmas >= 0, -1 is neutral)
        acc = -1.0
    elif MODE == 1:  # min: reset on the first chain (the FLT_MAX-neutral
        # init miscompiles to a device crash in this where-select chain)
        acc = 0.0
    else:  # sum
        acc = 0.0
    for j in range(K):
        # hi AND lo re-initialized per j (see the sturm kernel note).
        lo = 0.0
        hi = 2.0 * emax * (1.0 + 1e-9) + 1e-292
        target = K + j
        for it in range(BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            xh, xl = _split_f32(mid)
            cnt = _gk_sturm_count_less(E2H, E2L, base, N, xh, xl)
            if cnt >= target + 1:
                hi = mid
            else:
                lo = mid
        sigma = 0.5 * (lo + hi)
        if MODE == 0:
            # where-select, not tl.maximum: the hfusion pass pattern-
            # matches a scalar max chain into an isnan-checked op and
            # fails on the f64 it fabricates (BiShengHIR pipeline error).
            acc = tl.where(sigma > acc, sigma, acc)
        elif MODE == 1:
            acc = tl.where(j == 0, sigma, tl.where(sigma < acc, sigma, acc))
        else:
            acc = acc + sigma
    tl.store(O + b, acc)


def _svd_norm_tiny(A, mode):
    """Fused norm for tiny matrices (3 <= k <= 16, rows <= 64).

    k <= 7: serial fused sturm-norm (2 launches).  k >= 8: parallel w4 Sturm
    + _dim1_reduce (3 launches — one program running 16 serial chains costs
    ~4x the w4 grid).
    """
    in_dtype = A.dtype
    if in_dtype in (torch.float16, torch.bfloat16):
        A = A.float()
    A = A.contiguous()
    *batch_dims, M, N = A.shape
    batch = math.prod(batch_dims)
    k, rows = min(M, N), max(M, N)
    tall = M >= N
    dev = A.device
    nprog = _block_parallel_programs()
    block = max(triton.next_power_of_2(max(rows, k)), 32)
    # work holds the columns to be rotated: A for tall inputs; for wide
    # ones the kernel loads A^T directly (TRANSPOSED_LOAD) — the
    # transpose copy dominates at these launch-bound shapes.
    work = A.reshape(batch, rows, k) if tall else A.reshape(batch, M, N)
    n = 2 * k
    ekey = (batch, n, dev)
    e2h = _SVD_WORKSPACE_CACHE.get(ekey)
    if e2h is None:
        e2h = torch.empty((batch, n), dtype=torch.float32, device=dev)
        e2l = torch.empty((batch, n), dtype=torch.float32, device=dev)
        if len(_SVD_WORKSPACE_CACHE) >= _SVD_WORKSPACE_CACHE_MAX:
            _SVD_WORKSPACE_CACHE.clear()
        _SVD_WORKSPACE_CACHE[ekey] = e2h
        _SVD_WORKSPACE_CACHE[(batch, n, dev, "lo")] = e2l
    e2l = _SVD_WORKSPACE_CACHE[(batch, n, dev, "lo")]
    o = torch.empty(batch, dtype=torch.float32, device=dev)
    _bidiag_svd_kernel[(min(nprog, batch),)](
        work,
        e2h,
        e2l,
        K=k,
        BLOCK=block,
        ROWS=rows,
        BATCH=batch,
        NPROG=nprog,
        TRANSPOSED_LOAD=0 if tall else 1,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    if k >= 8:
        S = torch.empty((batch, k), dtype=torch.float32, device=dev)
        _small_sturm_run(e2h, e2l, S, k, batch, nprog, n, _SVD_SMALL_BISECT_ITERS)
        return _dim1_reduce(S, mode).reshape(batch_dims).to(in_dtype)
    _sturm_norm_kernel[(batch,)](
        e2h,
        e2l,
        o,
        K=k,
        BISECT_ITERS=_SVD_SMALL_BISECT_ITERS,
        N=n,
        MODE={"max": 0, "min": 1, "sum": 2}[mode],
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return o.reshape(batch_dims).to(in_dtype)


# ===========================================================================
# Gram path (rows > 64, k <= 64): 3 launches instead of the ~20-launch
# row-chunked bidiag chain
# ===========================================================================


def _use_gram(k, rows):
    """Whether the Gram path (3 launches: dot -> tridiag -> Sturm+sqrt) wins.

    Measured on card 6: iters = rows / GL <= 8 always wins; up to 64 iters
    only pays for k >= 20.  rows <= 64 keeps the linear-domain path (full
    sigma_min precision — Gram squares the spectrum).  k in (64, 128] uses
    the quad-tile Gram + rj4 symmetric tridiag (GL=64 tiles); k > 128
    overflows UB and stays on the row-chunked path.
    """
    if rows <= 64:
        return False
    if k > 128:
        return False
    if k > 64:
        return True
    gl = max(triton.next_power_of_2(k), 32)
    iters = (1 << (rows - 1).bit_length()) // gl
    return iters <= 8 or (iters <= 64 and k >= 20)


@libentry()
@triton.jit
def _gram_dot_kernel(
    A,
    G,
    K: tl.constexpr,
    GL: tl.constexpr,
    ROWS: tl.constexpr,
    MAX_ROWS: tl.constexpr,
    TALL: tl.constexpr,
    BATCH: tl.constexpr,
    NPROG: tl.constexpr,
):
    # G = A^T A (tall) or A A^T (wide).  CHUNK == GL: the dot reduces only
    # min(CHUNK, GL) lanes per iteration, so the tile must be square (GL, GL).
    # Padding is zeroed on load (clamped unmasked addressing + in-register where).
    for b in range(tl.program_id(0), BATCH, NPROG):
        rr = tl.arange(0, GL)
        cc = tl.arange(0, GL)
        g = tl.zeros([GL, GL], dtype=tl.float32)
        abase = A + b * ROWS * K
        if TALL == 1:
            for cs in range(0, MAX_ROWS, GL):
                msk = (cs + rr < ROWS)[:, None] & (cc < K)[None, :]
                rrow = tl.minimum(cs + rr, ROWS - 1)
                a = tl.load(abase + rrow[:, None] * K + tl.minimum(cc, K - 1)[None, :])
                a = tl.where(msk, a, 0.0)
                g = tl.dot(tl.trans(a), a, g, allow_tf32=False)
        else:
            for cs in range(0, MAX_ROWS, GL):
                msk = (cc < K)[:, None] & (cs + rr < ROWS)[None, :]
                rrow = tl.minimum(cs + rr, ROWS - 1)
                a = tl.load(abase + cc[:, None] * ROWS + rrow[None, :])
                a = tl.where(msk, a, 0.0)
                g = tl.dot(a, tl.trans(a), g, allow_tf32=False)
        tl.store(
            G + b * K * K + rr[:, None] * K + cc[None, :],
            g,
            mask=(rr < K)[:, None] & (cc < K)[None, :],
        )


@libentry()
@triton.jit
def _gram_quad_kernel(
    A,
    G,
    K: tl.constexpr,
    GL: tl.constexpr,
    ROWS: tl.constexpr,
    MAX_ROWS: tl.constexpr,
    TALL: tl.constexpr,
    BATCH: tl.constexpr,
    NPROG: tl.constexpr,
):
    # G = A^T A / A A^T for K in (64, 128], padded to (128, 128) with the
    # real k x k Gram top-left.  Three (GL, GL) quadrant dots (g11, g12, g22),
    # chunked over ROWS; bottom-left = g12^T.  Clamped unmasked loads +
    # in-register where (masked 2D loads miscompile).
    for b in range(tl.program_id(0), BATCH, NPROG):
        rr = tl.arange(0, GL)
        cc = tl.arange(0, GL)
        g11 = tl.zeros([GL, GL], dtype=tl.float32)
        g12 = tl.zeros([GL, GL], dtype=tl.float32)
        g22 = tl.zeros([GL, GL], dtype=tl.float32)
        abase = A + b * ROWS * K
        if TALL == 1:
            for cs in range(0, MAX_ROWS, GL):
                msk = (cs + rr < ROWS)[:, None]
                rrow = tl.minimum(cs + rr, ROWS - 1)
                a1 = tl.load(abase + rrow[:, None] * K + cc[None, :])
                a1 = tl.where(msk, a1, 0.0)
                col2 = tl.minimum(GL + cc, K - 1)
                a2 = tl.load(abase + rrow[:, None] * K + col2[None, :])
                a2 = tl.where(msk & ((GL + cc) < K)[None, :], a2, 0.0)
                g11 = tl.dot(tl.trans(a1), a1, g11, allow_tf32=False)
                g12 = tl.dot(tl.trans(a1), a2, g12, allow_tf32=False)
                g22 = tl.dot(tl.trans(a2), a2, g22, allow_tf32=False)
        else:
            for cs in range(0, MAX_ROWS, GL):
                msk = (cs + rr < ROWS)[None, :]
                crow = tl.minimum(cs + rr, ROWS - 1)
                a1 = tl.load(abase + cc[:, None] * ROWS + crow[None, :])
                a1 = tl.where(msk, a1, 0.0)
                row2 = tl.minimum(GL + cc, K - 1)
                a2 = tl.load(abase + row2[:, None] * ROWS + crow[None, :])
                a2 = tl.where(msk & ((GL + cc) < K)[:, None], a2, 0.0)
                g11 = tl.dot(a1, tl.trans(a1), g11, allow_tf32=False)
                g12 = tl.dot(a1, tl.trans(a2), g12, allow_tf32=False)
                g22 = tl.dot(a2, tl.trans(a2), g22, allow_tf32=False)
        gbase = G + b * 128 * 128
        tl.store(gbase + rr[:, None] * 128 + cc[None, :], g11)
        tl.store(gbase + rr[:, None] * 128 + (GL + cc)[None, :], g12)
        tl.store(gbase + (GL + rr)[:, None] * 128 + cc[None, :], tl.trans(g12))
        tl.store(gbase + (GL + rr)[:, None] * 128 + (GL + cc)[None, :], g22)


@libentry()
@triton.jit
def _gram_quad_split_kernel(
    A,
    P,
    K: tl.constexpr,
    GL: tl.constexpr,
    ROWS: tl.constexpr,
    MAX_ROWS: tl.constexpr,
    TALL: tl.constexpr,
    BATCH: tl.constexpr,
    NCHUNKS: tl.constexpr,
    NPROG: tl.constexpr,
):
    for w in range(tl.program_id(0), BATCH * NCHUNKS, NPROG):
        b = w // NCHUNKS
        cs = w % NCHUNKS
        rr = tl.arange(0, GL)
        cc = tl.arange(0, GL)
        g11 = tl.zeros([GL, GL], dtype=tl.float32)
        g12 = tl.zeros([GL, GL], dtype=tl.float32)
        g22 = tl.zeros([GL, GL], dtype=tl.float32)
        abase = A + b * ROWS * K
        if TALL == 1:
            msk = (cs * GL + rr < ROWS)[:, None]
            rrow = tl.minimum(cs * GL + rr, ROWS - 1)
            a1 = tl.load(abase + rrow[:, None] * K + cc[None, :])
            a1 = tl.where(msk, a1, 0.0)
            col2 = tl.minimum(GL + cc, K - 1)
            a2 = tl.load(abase + rrow[:, None] * K + col2[None, :])
            a2 = tl.where(msk & ((GL + cc) < K)[None, :], a2, 0.0)
            g11 = tl.dot(tl.trans(a1), a1, g11, allow_tf32=False)
            g12 = tl.dot(tl.trans(a1), a2, g12, allow_tf32=False)
            g22 = tl.dot(tl.trans(a2), a2, g22, allow_tf32=False)
        else:
            msk = (cs * GL + rr < ROWS)[None, :]
            crow = tl.minimum(cs * GL + rr, ROWS - 1)
            a1 = tl.load(abase + cc[:, None] * ROWS + crow[None, :])
            a1 = tl.where(msk, a1, 0.0)
            row2 = tl.minimum(GL + cc, K - 1)
            a2 = tl.load(abase + row2[:, None] * ROWS + crow[None, :])
            a2 = tl.where(msk & ((GL + cc) < K)[:, None], a2, 0.0)
            g11 = tl.dot(a1, tl.trans(a1), g11, allow_tf32=False)
            g12 = tl.dot(a1, tl.trans(a2), g12, allow_tf32=False)
            g22 = tl.dot(a2, tl.trans(a2), g22, allow_tf32=False)
        pbase = P + (b * NCHUNKS + cs) * 3 * GL * GL
        tl.store(pbase + rr[:, None] * GL + cc[None, :], g11)
        tl.store(pbase + GL * GL + rr[:, None] * GL + cc[None, :], g12)
        tl.store(pbase + 2 * GL * GL + rr[:, None] * GL + cc[None, :], g22)


@libentry()
@triton.jit
def _gram_quad_reduce_kernel(
    P,
    G,
    GL: tl.constexpr,
    BATCH: tl.constexpr,
    NCHUNKS: tl.constexpr,
    NPROG: tl.constexpr,
):
    # G = A^T A / A A^T assembled symmetric (g21 = trans(g12)) into the
    # padded 128x128 layout used by the tridiag kernel.
    for b in range(tl.program_id(0), BATCH, NPROG):
        rr = tl.arange(0, GL)
        cc = tl.arange(0, GL)
        acc = tl.zeros([GL, GL], dtype=tl.float32)
        for cs in range(NCHUNKS):
            pbase = P + (b * NCHUNKS + cs) * 3 * GL * GL
            acc += tl.load(pbase + rr[:, None] * GL + cc[None, :])
        g11 = acc
        acc = tl.zeros([GL, GL], dtype=tl.float32)
        for cs in range(NCHUNKS):
            pbase = P + (b * NCHUNKS + cs) * 3 * GL * GL
            acc += tl.load(pbase + GL * GL + rr[:, None] * GL + cc[None, :])
        g12 = acc
        acc = tl.zeros([GL, GL], dtype=tl.float32)
        for cs in range(NCHUNKS):
            pbase = P + (b * NCHUNKS + cs) * 3 * GL * GL
            acc += tl.load(pbase + 2 * GL * GL + rr[:, None] * GL + cc[None, :])
        g22 = acc
        gbase = G + b * 128 * 128
        tl.store(gbase + rr[:, None] * 128 + cc[None, :], g11)
        tl.store(gbase + rr[:, None] * 128 + (GL + cc)[None, :], g12)
        tl.store(gbase + (GL + rr)[:, None] * 128 + cc[None, :], tl.trans(g12))
        tl.store(gbase + (GL + rr)[:, None] * 128 + (GL + cc)[None, :], g22)


@libentry()
@triton.jit
def _tridiag_rj4(
    G,
    E2H,
    E2L,
    TILES,
    K: tl.constexpr,
    K2: tl.constexpr,
    GL: tl.constexpr,
    BATCH: tl.constexpr,
    NPROG: tl.constexpr,
    EXT: tl.constexpr,
):
    for b in range(tl.program_id(0), BATCH, NPROG):
        rr = tl.arange(0, GL)
        cc = tl.arange(0, GL)
        gbase = G + b * K2 * K2
        e2b = b * (2 * K)
        t00 = tl.load(gbase + rr[:, None] * K2 + cc[None, :])
        t10 = tl.load(gbase + (GL + rr)[:, None] * K2 + cc[None, :])
        t11 = tl.load(gbase + (GL + rr)[:, None] * K2 + (GL + cc)[None, :])
        for j in range(K2):
            jl = j - GL
            # 2-lane window masks; out-of-range selects nothing -> 0.
            cmaskj = (cc[None, :] > j - 1) & (cc[None, :] < j + 1)
            cmaskl = (cc[None, :] > jl - 1) & (cc[None, :] < jl + 1)
            rmaskl = (rr[:, None] > jl - 1) & (rr[:, None] < jl + 1)
            colj00 = tl.sum(tl.where(cmaskj, t00, 0.0), axis=1)
            colj10 = tl.sum(tl.where(cmaskj, t10, 0.0), axis=1)
            rowj10 = tl.sum(tl.where(rmaskl, t10, 0.0), axis=0)
            colj11 = tl.sum(tl.where(cmaskl, t11, 0.0), axis=1)
            seg0 = colj00 + rowj10
            seg1 = colj10 + colj11
            # subdiagonal element x0 = A[j+1, j] (lane j+1 / jl+1)
            mj2 = (rr > j) & (rr < j + 2)
            ml2 = (rr > jl) & (rr < jl + 2)
            x0 = tl.sum(seg0 * mj2.to(tl.float32), axis=0) + tl.sum(
                seg1 * ml2.to(tl.float32), axis=0
            )
            x0v = seg0 * (rr > j).to(tl.float32)
            x1v = seg1 * (rr > jl).to(tl.float32)
            sigma = tl.sqrt(tl.sum(x0v * x0v, axis=0) + tl.sum(x1v * x1v, axis=0))
            alpha = tl.where(x0 >= 0.0, -sigma, sigma)
            # per-step seed extraction (only seed source: 2D masked reductions
            # after the loop miscompile on this toolchain)
            mjd = (rr > j - 1) & (rr < j + 1)
            mld = (rr > jl - 1) & (rr < jl + 1)
            d_j = tl.sum(seg0 * mjd.to(tl.float32), axis=0) + tl.sum(
                seg1 * mld.to(tl.float32), axis=0
            )
            if j < K:
                # interleaved DS seeds: d_j (exact fp32 pair) and s_j^2
                # (j = K-1 gives s^2 = 0: alpha = 0, padded subdiag empty).
                # Consumed by the symmetric-tridiag count p = (d-x) - s^2/p.
                dh, dl = _split_f32(d_j)
                sh, sl = _split_f32(alpha)
                s2h, s2l = _df64_mul_ds(sh, sl, sh, sl)
                tl.store(E2H + e2b + 2 * j, dh)
                tl.store(E2L + e2b + 2 * j, dl)
                tl.store(E2H + e2b + 2 * j + 1, s2h)
                tl.store(E2L + e2b + 2 * j + 1, s2l)
            v0 = tl.where(mj2, x0 - alpha, x0v)
            v1 = tl.where(ml2, x0 - alpha, x1v)
            vnorm2 = 2.0 * sigma * (sigma + tl.abs(x0))
            tau = tl.where(vnorm2 > 0.0, 2.0 / vnorm2, 0.0)
            w0 = tl.sum(t00 * v0[None, :], axis=1) + tl.sum(t10 * v1[:, None], axis=0)
            w1 = tl.sum(t10 * v0[None, :], axis=1) + tl.sum(t11 * v1[None, :], axis=1)
            gamma = tl.sum(v0 * w0, axis=0) + tl.sum(v1 * w1, axis=0)
            c1 = tau
            c2 = tau * tau * gamma
            q0 = c1 * w0 - (0.5 * c2) * v0
            q1 = c1 * w1 - (0.5 * c2) * v1
            # v0/q0/v1/q1 support rows > j (resp. > jl), so updates are
            # restricted to the trailing submatrix and frozen rows stay put.
            t00 = (
                t00
                - tl.reshape(v0, (GL, 1)) * tl.reshape(q0, (1, GL))
                - tl.reshape(q0, (GL, 1)) * tl.reshape(v0, (1, GL))
            )
            t10 = (
                t10
                - tl.reshape(v1, (GL, 1)) * tl.reshape(q0, (1, GL))
                - tl.reshape(q1, (GL, 1)) * tl.reshape(v0, (1, GL))
            )
            t11 = (
                t11
                - tl.reshape(v1, (GL, 1)) * tl.reshape(q1, (1, GL))
                - tl.reshape(q1, (GL, 1)) * tl.reshape(v1, (1, GL))
            )
        # ---- optional tail: TILES stores for verification ----
        if EXT:
            tbase = b * 3 * K2 * K2
            tl.store(TILES + tbase + rr[:, None] * K2 + cc[None, :], t00)
            tl.store(
                TILES + tbase + K2 * K2 + (GL + rr)[:, None] * K2 + cc[None, :], t10
            )
            tl.store(
                TILES
                + tbase
                + 2 * K2 * K2
                + (GL + rr)[:, None] * K2
                + (GL + cc)[None, :],
                t11,
            )


@triton.jit
def _sym_chain_step(dh, dl, sh, sl, qh, ql, xh, xl):
    # one step of the symmetric-tridiag Sturm recurrence (DS):
    #   p = (d - x) - s^2 / p_prev
    rh, rl = _df64_div_ds(sh, sl, qh, ql)
    dhx, dlx = _df64_add(dh, dl, -xh, -xl)
    qh2, ql2 = _df64_add(dhx, dlx, -rh, -rl)
    zero_q = (qh2 == 0.0) & (ql2 == 0.0)
    qh2 = tl.where(zero_q, -1.1754944e-38, qh2)
    ql2 = tl.where(zero_q, 0.0, ql2)
    return qh2, ql2, tl.where(qh2 < 0.0, 1, 0)


@triton.jit
def _sym_count_less(E2H, E2L, base, K: tl.constexpr, xh, xl):
    # # {lambda <= x} for tridiag(d, s) with the interleaved DS seeds.
    dh0 = tl.load(E2H + base + 0)
    dl0 = tl.load(E2L + base + 0)
    qh, ql = _df64_add(dh0, dl0, -xh, -xl)
    zero_q = (qh == 0.0) & (ql == 0.0)
    qh = tl.where(zero_q, -1.1754944e-38, qh)
    ql = tl.where(zero_q, 0.0, ql)
    neg = tl.where(qh < 0.0, 1, 0)
    for i in range(1, K):
        dh = tl.load(E2H + base + 2 * i)
        dl = tl.load(E2L + base + 2 * i)
        sh = tl.load(E2H + base + 2 * i - 1)
        sl = tl.load(E2L + base + 2 * i - 1)
        qh, ql, d = _sym_chain_step(dh, dl, sh, sl, qh, ql, xh, xl)
        neg += d
    return neg


@triton.jit
def _sym_w4_count_way4(
    E2H, E2L, base, K: tl.constexpr, x0h, x0l, x1h, x1l, x2h, x2l, x3h, x3l
):
    # four interleaved symmetric-tridiag chains sharing the d/s loads.
    dh0 = tl.load(E2H + base + 0)
    dl0 = tl.load(E2L + base + 0)
    q0h, q0l = _df64_add(dh0, dl0, -x0h, -x0l)
    q1h, q1l = _df64_add(dh0, dl0, -x1h, -x1l)
    q2h, q2l = _df64_add(dh0, dl0, -x2h, -x2l)
    q3h, q3l = _df64_add(dh0, dl0, -x3h, -x3l)
    z0 = (q0h == 0.0) & (q0l == 0.0)
    q0h = tl.where(z0, -1.1754944e-38, q0h)
    q0l = tl.where(z0, 0.0, q0l)
    z1 = (q1h == 0.0) & (q1l == 0.0)
    q1h = tl.where(z1, -1.1754944e-38, q1h)
    q1l = tl.where(z1, 0.0, q1l)
    z2 = (q2h == 0.0) & (q2l == 0.0)
    q2h = tl.where(z2, -1.1754944e-38, q2h)
    q2l = tl.where(z2, 0.0, q2l)
    z3 = (q3h == 0.0) & (q3l == 0.0)
    q3h = tl.where(z3, -1.1754944e-38, q3h)
    q3l = tl.where(z3, 0.0, q3l)
    n0 = tl.where(q0h < 0.0, 1, 0)
    n1 = tl.where(q1h < 0.0, 1, 0)
    n2 = tl.where(q2h < 0.0, 1, 0)
    n3 = tl.where(q3h < 0.0, 1, 0)
    for i in range(1, K):
        dh = tl.load(E2H + base + 2 * i)
        dl = tl.load(E2L + base + 2 * i)
        sh = tl.load(E2H + base + 2 * i - 1)
        sl = tl.load(E2L + base + 2 * i - 1)
        q0h, q0l, d0 = _sym_chain_step(dh, dl, sh, sl, q0h, q0l, x0h, x0l)
        q1h, q1l, d1 = _sym_chain_step(dh, dl, sh, sl, q1h, q1l, x1h, x1l)
        q2h, q2l, d2 = _sym_chain_step(dh, dl, sh, sl, q2h, q2l, x2h, x2l)
        q3h, q3l, d3 = _sym_chain_step(dh, dl, sh, sl, q3h, q3l, x3h, x3l)
        n0 += d0
        n1 += d1
        n2 += d2
        n3 += d3
    return n0, n1, n2, n3


@libentry()
@triton.jit
def _sturm_norm_sqrt_w4_sym_kernel(
    E2H,
    E2L,
    O,
    K: tl.constexpr,
    NPROG: tl.constexpr,
    BISECT_ITERS: tl.constexpr,
    BATCH: tl.constexpr,
    NGROUPS: tl.constexpr,
    MODE: tl.constexpr,
):
    # w4 Sturm bisection + sqrt over symmetric-tridiag seeds, fused norm
    # with atomics (skeleton of _sturm_norm_sqrt_w4_kernel; symmetric
    # count, targets t = j, 3 * emax interval bound).
    WTOT: tl.constexpr = BATCH * NGROUPS
    for w in range(tl.program_id(0), WTOT, NPROG):
        b = w // NGROUPS
        g = w % NGROUPS
        base = b * 2 * K
        emax = 0.0
        for i in range(1, 2 * K):
            e = tl.load(E2H + base + i - 1)
            if (i - 1) % 2 == 0:
                emax = tl.maximum(emax, tl.abs(e))
            else:
                emax = tl.maximum(emax, tl.sqrt(e))
        hi0 = 3.0 * emax * (1.0 + 1e-9) + 1e-292
        j0 = tl.minimum(4 * g, K - 1)
        j1 = tl.minimum(4 * g + 1, K - 1)
        j2 = tl.minimum(4 * g + 2, K - 1)
        j3 = tl.minimum(4 * g + 3, K - 1)
        lo0 = 0.0
        hi1 = hi0
        lo1 = 0.0
        hi2 = hi0
        lo2 = 0.0
        hi3 = hi0
        lo3 = 0.0
        t0 = j0
        t1 = j1
        t2 = j2
        t3 = j3
        for it in range(BISECT_ITERS):
            mid0 = 0.5 * (lo0 + hi1)
            mid1 = 0.5 * (lo1 + hi2)
            mid2 = 0.5 * (lo2 + hi3)
            mid3 = 0.5 * (lo3 + hi0)
            x0h, x0l = _split_f32(mid0)
            x1h, x1l = _split_f32(mid1)
            x2h, x2l = _split_f32(mid2)
            x3h, x3l = _split_f32(mid3)
            c0, c1, c2, c3 = _sym_w4_count_way4(
                E2H, E2L, base, K, x0h, x0l, x1h, x1l, x2h, x2l, x3h, x3l
            )
            if c0 >= t0 + 1:
                hi1 = mid0
            else:
                lo0 = mid0
            if c1 >= t1 + 1:
                hi2 = mid1
            else:
                lo1 = mid1
            if c2 >= t2 + 1:
                hi3 = mid2
            else:
                lo2 = mid2
            if c3 >= t3 + 1:
                hi0 = mid3
            else:
                lo3 = mid3
        s0 = tl.sqrt(0.5 * (lo0 + hi1))
        s1 = tl.sqrt(0.5 * (lo1 + hi2))
        s2 = tl.sqrt(0.5 * (lo2 + hi3))
        s3 = tl.sqrt(0.5 * (lo3 + hi0))
        if MODE == 0:
            acc = tl.maximum(tl.maximum(s0, s1), tl.maximum(s2, s3))
            tl.atomic_max(O + b, acc)
        elif MODE == 1:
            acc = tl.minimum(tl.minimum(s0, s1), tl.minimum(s2, s3))
            tl.atomic_min(O + b, acc)
        else:
            acc = (
                tl.where(4 * g < K, s0, 0.0)
                + tl.where(4 * g + 1 < K, s1, 0.0)
                + tl.where(4 * g + 2 < K, s2, 0.0)
                + tl.where(4 * g + 3 < K, s3, 0.0)
            )
            tl.atomic_add(O + b, acc)


@libentry()
@triton.jit
def _sturm_norm_sqrt_w1_extreme_kernel(
    E2H,
    E2L,
    O,
    K: tl.constexpr,
    NPROG: tl.constexpr,
    BISECT_ITERS: tl.constexpr,
    BATCH: tl.constexpr,
    MODE: tl.constexpr,
):
    # One Sturm chain per batch for the extreme eigenvalue only: MODE=0
    # (max) bisects t = K-1 -> lambda_max, MODE=1 (min) t = 0 -> lambda_min.
    # One program per batch (no groups), so direct stores replace the
    # atomics.
    for b in range(tl.program_id(0), BATCH, NPROG):
        base = b * 2 * K
        emax = 0.0
        for i in range(1, 2 * K):
            e = tl.load(E2H + base + i - 1)
            if (i - 1) % 2 == 0:
                emax = tl.maximum(emax, tl.abs(e))
            else:
                emax = tl.maximum(emax, tl.sqrt(e))
        hi = 3.0 * emax * (1.0 + 1e-9) + 1e-292
        lo = 0.0
        if MODE == 0:
            tgt = K - 1
        else:
            tgt = 0
        for it in range(BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            xh, xl = _split_f32(mid)
            c = _sym_count_less(E2H, E2L, base, K, xh, xl)
            if c >= tgt + 1:
                hi = mid
            else:
                lo = mid
        tl.store(O + b, tl.sqrt(0.5 * (lo + hi)))


@libentry()
@triton.jit
def _sturm_norm_sqrt_kernel(
    E2H,
    E2L,
    O,
    K: tl.constexpr,
    BISECT_ITERS: tl.constexpr,
    N: tl.constexpr,
    MODE: tl.constexpr,
):
    # Serial fused sturm-norm with sqrt (Gram path: bisection yields
    # sigma(G) = sigma(A)^2).  One program per batch element, k == 3
    # (the w4 bisection miscompiles on this toolchain for k == 3).
    b = tl.program_id(0)
    base = b * N
    emax = 0.0
    for i in range(1, N):
        emax = tl.maximum(emax, tl.load(E2H + base + i - 1))
    emax = tl.sqrt(emax)
    if MODE == 0:
        acc = -1.0
    elif MODE == 1:
        acc = 0.0
    else:
        acc = 0.0
    for j in range(K):
        lo = 0.0
        hi = 2.0 * emax * (1.0 + 1e-9) + 1e-292
        target = K + j
        for it in range(BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            xh, xl = _split_f32(mid)
            cnt = _gk_sturm_count_less(E2H, E2L, base, N, xh, xl)
            if cnt >= target + 1:
                hi = mid
            else:
                lo = mid
        sigma = tl.sqrt(0.5 * (lo + hi))
        if MODE == 0:
            acc = tl.where(sigma > acc, sigma, acc)
        elif MODE == 1:
            acc = tl.where(j == 0, sigma, tl.where(sigma < acc, sigma, acc))
        else:
            acc = acc + sigma
    tl.store(O + b, acc)


@libentry()
@triton.jit
def _sturm_norm_sqrt_w4_kernel(
    E2H,
    E2L,
    O,
    K: tl.constexpr,
    NPROG: tl.constexpr,
    BISECT_ITERS: tl.constexpr,
    N: tl.constexpr,
    BATCH: tl.constexpr,
    NGROUPS: tl.constexpr,
    MODE: tl.constexpr,
):
    # w4 sturm bisection + sqrt (Gram path) + per-program partial reduction
    # + atomic into O[b].  O must be pre-filled with the mode's neutral
    # value (garbage defeats the atomics).  Masked-out lanes (4g+i >= K)
    # duplicate the last real sigma: harmless for max/min, `where`-excluded
    # from sum.
    WTOT: tl.constexpr = BATCH * NGROUPS
    for w in range(tl.program_id(0), WTOT, NPROG):
        b = w // NGROUPS
        g = w % NGROUPS
        base = b * N
        emax = 0.0
        for i in range(1, N):
            emax = tl.maximum(emax, tl.load(E2H + base + i - 1))
        emax = tl.sqrt(emax)
        hi0 = 2.0 * emax * (1.0 + 1e-9) + 1e-292
        j0 = tl.minimum(4 * g, K - 1)
        j1 = tl.minimum(4 * g + 1, K - 1)
        j2 = tl.minimum(4 * g + 2, K - 1)
        j3 = tl.minimum(4 * g + 3, K - 1)
        lo0 = 0.0
        hi1 = hi0
        lo1 = 0.0
        hi2 = hi0
        lo2 = 0.0
        hi3 = hi0
        lo3 = 0.0
        t0 = K + j0
        t1 = K + j1
        t2 = K + j2
        t3 = K + j3
        for it in range(BISECT_ITERS):
            mid0 = 0.5 * (lo0 + hi1)
            mid1 = 0.5 * (lo1 + hi2)
            mid2 = 0.5 * (lo2 + hi3)
            mid3 = 0.5 * (lo3 + hi0)
            x0h, x0l = _split_f32(mid0)
            x1h, x1l = _split_f32(mid1)
            x2h, x2l = _split_f32(mid2)
            x3h, x3l = _split_f32(mid3)
            c0, c1, c2, c3 = _w4_count_way4(
                E2H, E2L, base, N, x0h, x0l, x1h, x1l, x2h, x2l, x3h, x3l
            )
            if c0 >= t0 + 1:
                hi1 = mid0
            else:
                lo0 = mid0
            if c1 >= t1 + 1:
                hi2 = mid1
            else:
                lo1 = mid1
            if c2 >= t2 + 1:
                hi3 = mid2
            else:
                lo2 = mid2
            if c3 >= t3 + 1:
                hi0 = mid3
            else:
                lo3 = mid3
        s0 = tl.sqrt(0.5 * (lo0 + hi1))
        s1 = tl.sqrt(0.5 * (lo1 + hi2))
        s2 = tl.sqrt(0.5 * (lo2 + hi3))
        s3 = tl.sqrt(0.5 * (lo3 + hi0))
        if MODE == 0:
            acc = tl.maximum(tl.maximum(s0, s1), tl.maximum(s2, s3))
            tl.atomic_max(O + b, acc)
        elif MODE == 1:
            acc = tl.minimum(tl.minimum(s0, s1), tl.minimum(s2, s3))
            tl.atomic_min(O + b, acc)
        else:
            acc = (
                tl.where(4 * g < K, s0, 0.0)
                + tl.where(4 * g + 1 < K, s1, 0.0)
                + tl.where(4 * g + 2 < K, s2, 0.0)
                + tl.where(4 * g + 3 < K, s3, 0.0)
            )
            tl.atomic_add(O + b, acc)


def _gram_norm_fast_k128(A, mode):
    """Gram path for 64 < k <= 128: quad-tile G = A^T A / A A^T (three
    (64,64) chunked dots, split+reduce when rows > 128) -> rj4 symmetric
    tridiagonalization -> symmetric Sturm bisection + sqrt (w4 for sum, w1
    for max/min).  The tridiag emits interleaved double-single seeds d_j /
    s_j^2 for the symmetric-tridiag count (p = (d - x) - s^2/p).  3-4 launches.
    """
    in_dtype = A.dtype
    if in_dtype in (torch.float16, torch.bfloat16):
        A = A.float()
    A = A.contiguous()
    *batch_dims, M, N = A.shape
    batch = math.prod(batch_dims)
    k, rows = min(M, N), max(M, N)
    tall = M >= N
    dev = A.device
    nprog = _block_parallel_programs()
    K2, GL = 128, 64
    work = A.reshape(batch, M, N)
    G = torch.empty(batch, K2, K2, dtype=torch.float32, device=dev)
    n = 2 * k
    ekey = (batch, n, dev)
    e2h = _SVD_WORKSPACE_CACHE.get(ekey)
    if e2h is None:
        e2h = torch.empty((batch, n), dtype=torch.float32, device=dev)
        e2l = torch.empty((batch, n), dtype=torch.float32, device=dev)
        if len(_SVD_WORKSPACE_CACHE) >= _SVD_WORKSPACE_CACHE_MAX:
            _SVD_WORKSPACE_CACHE.clear()
        _SVD_WORKSPACE_CACHE[ekey] = e2h
        _SVD_WORKSPACE_CACHE[(batch, n, dev, "lo")] = e2l
    e2l = _SVD_WORKSPACE_CACHE[(batch, n, dev, "lo")]
    neutral = -1.0 if mode == "max" else (3.4028235e38 if mode == "min" else 0.0)
    o = torch.full((batch,), neutral, dtype=torch.float32, device=dev)
    modi = {"max": 0, "min": 1, "sum": 2}[mode]
    nchunks = (1 << (rows - 1).bit_length()) // GL
    if nchunks > 2:
        P = torch.zeros(
            (batch * nchunks * 3 * GL * GL,), dtype=torch.float32, device=dev
        )
        _gram_quad_split_kernel[(min(nprog, batch * nchunks),)](
            work,
            P,
            K=k,
            GL=GL,
            ROWS=rows,
            MAX_ROWS=1 << (rows - 1).bit_length(),
            TALL=1 if tall else 0,
            BATCH=batch,
            NCHUNKS=nchunks,
            NPROG=nprog,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
        _gram_quad_reduce_kernel[(min(nprog, batch),)](
            P,
            G,
            GL=GL,
            BATCH=batch,
            NCHUNKS=nchunks,
            NPROG=nprog,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    else:
        _gram_quad_kernel[(min(nprog, batch),)](
            work,
            G,
            K=k,
            GL=GL,
            ROWS=rows,
            MAX_ROWS=1 << (rows - 1).bit_length(),
            TALL=1 if tall else 0,
            BATCH=batch,
            NPROG=nprog,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    _tridiag_rj4[(min(nprog, batch),)](
        G,
        e2h,
        e2l,
        G,
        K=k,
        K2=K2,
        GL=GL,
        BATCH=batch,
        NPROG=nprog,
        EXT=0,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    if modi < 2:
        _sturm_norm_sqrt_w1_extreme_kernel[(min(nprog, batch),)](
            e2h,
            e2l,
            o,
            K=k,
            NPROG=nprog,
            BISECT_ITERS=30,
            BATCH=batch,
            MODE=modi,
            num_warps=1,
            num_stages=1,
            enable_fp_fusion=False,
        )
    else:
        ngroups = (k + 3) // 4
        _sturm_norm_sqrt_w4_sym_kernel[(min(nprog, batch * ngroups),)](
            e2h,
            e2l,
            o,
            K=k,
            NPROG=nprog,
            BISECT_ITERS=30,
            BATCH=batch,
            NGROUPS=ngroups,
            MODE=modi,
            num_warps=1,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return o.reshape(*batch_dims) if batch_dims else o.reshape(())


def _gram_norm_fast(A, mode):
    """Gram path for rows > 64, k <= 64: G = A^T A (or A A^T) -> GK
    bidiagonalization of G -> Sturm bisection + sqrt (sigma(G) = sigma(A)^2).
    The norm is fused into the bisection kernel (atomics), so the chain is 3
    launches instead of the ~20-launch row-chunked bidiag chain.
    """
    in_dtype = A.dtype
    if in_dtype in (torch.float16, torch.bfloat16):
        A = A.float()
    A = A.contiguous()
    *batch_dims, M, N = A.shape
    batch = math.prod(batch_dims)
    k, rows = min(M, N), max(M, N)
    if k > 64:
        return _gram_norm_fast_k128(A, mode)
    if k == 8:
        return _jacobi8_norm_fast(A, mode)
    tall = M >= N
    dev = A.device
    nprog = _block_parallel_programs()
    GL = max(triton.next_power_of_2(k), 32)
    MAX_ROWS = 1 << (rows - 1).bit_length()
    work = A.reshape(batch, rows, k) if tall else A.reshape(batch, M, N)
    G = torch.empty(batch, k, k, dtype=torch.float32, device=dev)
    n = 2 * k
    ekey = (batch, n, dev)
    e2h = _SVD_WORKSPACE_CACHE.get(ekey)
    if e2h is None:
        e2h = torch.empty((batch, n), dtype=torch.float32, device=dev)
        e2l = torch.empty((batch, n), dtype=torch.float32, device=dev)
        if len(_SVD_WORKSPACE_CACHE) >= _SVD_WORKSPACE_CACHE_MAX:
            _SVD_WORKSPACE_CACHE.clear()
        _SVD_WORKSPACE_CACHE[ekey] = e2h
        _SVD_WORKSPACE_CACHE[(batch, n, dev, "lo")] = e2l
    e2l = _SVD_WORKSPACE_CACHE[(batch, n, dev, "lo")]
    neutral = -1.0 if mode == "max" else (3.4028235e38 if mode == "min" else 0.0)
    o = torch.full((batch,), neutral, dtype=torch.float32, device=dev)
    _gram_dot_kernel[(min(nprog, batch),)](
        work,
        G,
        K=k,
        GL=GL,
        ROWS=rows,
        MAX_ROWS=MAX_ROWS,
        TALL=1 if tall else 0,
        BATCH=batch,
        NPROG=nprog,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    _bidiag_svd_kernel[(min(nprog, batch),)](
        G,
        e2h,
        e2l,
        K=k,
        BLOCK=GL,
        ROWS=k,
        BATCH=batch,
        NPROG=nprog,
        TRANSPOSED_LOAD=0,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    modi = {"max": 0, "min": 1, "sum": 2}[mode]
    if k >= 4:
        ngroups = (k + 3) // 4
        _sturm_norm_sqrt_w4_kernel[(min(nprog, batch * ngroups),)](
            e2h,
            e2l,
            o,
            K=k,
            NPROG=nprog,
            BISECT_ITERS=_SVD_SMALL_BISECT_ITERS,
            N=n,
            BATCH=batch,
            NGROUPS=ngroups,
            MODE=modi,
            num_warps=1,
            num_stages=1,
            enable_fp_fusion=False,
        )
    else:
        _sturm_norm_sqrt_kernel[(batch,)](
            e2h,
            e2l,
            o,
            K=k,
            BISECT_ITERS=_SVD_SMALL_BISECT_ITERS,
            N=n,
            MODE=modi,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return o.reshape(*batch_dims) if batch_dims else o.reshape(())


def _svdvals_for_norm(A):
    """Pure-Triton SVD dispatch for ord=2/-2/nuc on Ascend NPU."""
    in_dtype = A.dtype
    if in_dtype in (torch.float16, torch.bfloat16):
        A = A.float()
    A = A.contiguous()
    *batch_dims, M, N = A.shape
    batch = 1
    for d in batch_dims:
        batch *= d
    k = min(M, N)
    rows = max(M, N)

    if k == 1:
        flat = A.reshape(batch, M * N)
        s = torch.empty(batch, 1, dtype=torch.float32, device=A.device)
        blk_n = triton.next_power_of_2(min(M * N, 512))
        _fro_kernel[(batch,)](
            flat,
            s,
            0,
            M * N,
            1,
            blk_n,
            1,
            TILE_2D=False,
            USE_FP64=False,
            num_warps=8,
        )
        return s.reshape(*batch_dims, 1).to(in_dtype)

    # k=2: closed form (one launch — launch-bound at these shapes)
    if k == 2 and rows <= _RANK2_BLOCK_R_MAX:
        return _rank2_svals_fast(A).to(in_dtype)

    # k ≥ 3, k ≤ 64: pure-Triton on AI_CORE (bidiag + Jacobi, or Jacobi on
    # A for tall/wide) — Phase 2.  k > 64: pure-Triton grid-parallel
    # bidiag + Sturm bisection on the Golub-Kahan tridiagonal — Phase 3.
    # No torch.linalg, no CANN QR, no CPU transfer anywhere on the SVD
    # path (numpy is gone from this module entirely).
    if 3 <= k <= _SVD_SMALL_MAX_K and rows <= _SVD_SMALL_MAX_ROWS:
        return _svdvals_small(A).reshape(*batch_dims, k).to(in_dtype)
    if 3 <= k <= _SVD_BLOCKED_MAX_K and rows <= _SVD_BLOCKED_MAX_ROWS:
        return _svdvals_blocked(A).reshape(*batch_dims, k).to(in_dtype)

    raise NotImplementedError(
        f"FlagGems Ascend svdvals: unsupported shape ({M},{N}). "
        f"Supported: k=1 (L2), k=2 (closed form), 3≤k≤512 with rows≤2048."
    )


# ===========================================================================
# k <= 64 pure-Triton singular values (Phase 2)
# ===========================================================================
_SVD_SMALL_MAX_K = 64
_SVD_SMALL_MAX_ROWS = 2048

# Reused small-path intermediate buffers (E2 seeds), keyed by (batch, size,
# device): torch.empty dispatch (~0.02 ms) rivals the kernels at launch-bound
# shapes.  S (the exposed output) is always fresh, so calls never alias.
_SVD_WORKSPACE_CACHE = {}
_SVD_WORKSPACE_CACHE_MAX = 8


def _small_sturm_run(e2h, e2l, S, k, batch, nprog, n, iters):
    """Run the small-path Sturm bisection kernel.

    k >= 4: w4 (four interleaved qd chains).  k == 3: scalar chain — the w4
    binary crashes the device for N = 6 (toolchain bug).
    """
    if k >= 4:
        ngroups = (k + 3) // 4
        _sturm_sigmas_w4_kernel[(min(nprog, batch * ngroups),)](
            e2h,
            e2l,
            S,
            K=k,
            NPROG=nprog,
            BISECT_ITERS=iters,
            N=n,
            BATCH=batch,
            NGROUPS=ngroups,
            num_warps=1,
            num_stages=1,
            enable_fp_fusion=False,
        )
        return
    snprog = min(nprog, k)
    _sturm_sigmas_kernel[(min(nprog, batch * snprog),)](
        e2h,
        e2l,
        S,
        K=k,
        NPROG=snprog,
        BISECT_ITERS=iters,
        N=n,
        BATCH=batch,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )


def _svdvals_small(A):
    """Pure-Triton singular values for 3 <= k <= 64, rows <= 2048 (fp32).

    rows <= 64: GK bidiagonalization into a (BLOCK, BLOCK) register tile +
    Sturm bisection (E2 seed fused into the bidiag launch).  rows > 64:
    row-chunked GK bidiagonalization of the (batch, k, rows) workspace.
    """
    *batch_dims, M, N = A.shape
    batch = math.prod(batch_dims)
    k, rows = min(M, N), max(M, N)
    tall = M >= N
    dev = A.device
    nprog = _block_parallel_programs()

    if rows <= 64:
        block = max(triton.next_power_of_2(max(rows, k)), 32)
        # work holds the columns to rotate: A for tall; wide reads A^T directly
        # (TRANSPOSED_LOAD — the transpose copy dominates at these shapes).
        work = A.reshape(batch, rows, k) if tall else A.reshape(batch, M, N)
        S = torch.empty((batch, k), dtype=torch.float32, device=dev)
        # E2 seed (interleaved double-single squares) is fused into the bidiag
        # launch (a separate gk_init launch would dominate).
        n = 2 * k
        ekey = (batch, n, dev)
        e2h = _SVD_WORKSPACE_CACHE.get(ekey)
        if e2h is None:
            e2h = torch.empty((batch, n), dtype=torch.float32, device=dev)
            e2l = torch.empty((batch, n), dtype=torch.float32, device=dev)
            if len(_SVD_WORKSPACE_CACHE) >= _SVD_WORKSPACE_CACHE_MAX:
                _SVD_WORKSPACE_CACHE.clear()
            _SVD_WORKSPACE_CACHE[ekey] = e2h
            _SVD_WORKSPACE_CACHE[(batch, n, dev, "lo")] = e2l
        e2l = _SVD_WORKSPACE_CACHE[(batch, n, dev, "lo")]
        _bidiag_svd_kernel[(min(nprog, batch),)](
            work,
            e2h,
            e2l,
            K=k,
            BLOCK=block,
            ROWS=rows,
            BATCH=batch,
            NPROG=nprog,
            TRANSPOSED_LOAD=0 if tall else 1,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
        _small_sturm_run(e2h, e2l, S, k, batch, nprog, n, _SVD_SMALL_BISECT_ITERS)
    else:
        # Row-chunked bidiagonalization of the (batch, k, rows) workspace.
        work = torch.zeros((batch, k, rows), dtype=torch.float32, device=dev)
        src = A.reshape(batch, M, N)
        if tall:
            src = src.transpose(-2, -1)
        work[:, :k, :] = src
        # One launch per (j, reflection-half): a kernel boundary is the only
        # fence ordering MTE2 stores against MTE3 loads.  The left reflection
        # runs for every j (the final one zeroes col k-1 -> W = [B; 0]).
        # MAX_ROWS must be a compile-time bound (a runtime bound miscompiles).
        max_rows = 1 << (rows - 1).bit_length()
        # One pair loop over j in [0, k-2) + two final left-only steps
        # (right(j) writes row j+1 and is valid only for j < k-2).
        for j in range(k):
            _bidiag_left_step_kernel[(min(nprog, batch),)](
                work,
                k,
                rows,
                j,
                BATCH=batch,
                NPROG=nprog,
                CHUNK=512,
                MAX_ROWS=max_rows,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=True,
            )
            if j + 2 < k:
                _bidiag_right_step_kernel[(min(nprog, batch),)](
                    work,
                    k,
                    rows,
                    j,
                    DO_RIGHT=True,
                    BATCH=batch,
                    NPROG=nprog,
                    CHUNK=512,
                    MAX_ROWS=max_rows,
                    num_warps=4,
                    num_stages=1,
                    enable_fp_fusion=True,
                )
        # GK seed + Sturm bisection on the bidiagonal corner (d at [i,i],
        # off-diagonal at (i+1)*rows+i = B[i,i+1], the SUPERdiagonal).
        n = 2 * k
        e2h = torch.empty((batch, n), dtype=torch.float32, device=dev)
        e2l = torch.empty((batch, n), dtype=torch.float32, device=dev)
        S = torch.empty((batch, k), dtype=torch.float32, device=dev)
        _gk_init_kernel[(min(nprog, batch),)](
            work,
            e2h,
            e2l,
            rows,
            K=k,
            BATCH=batch,
            NPROG=nprog,
            BLOCK=_SVD_BLOCKED_GK_BLOCK,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
        _small_sturm_run(e2h, e2l, S, k, batch, nprog, n, _SVD_BLOCKED_BISECT_ITERS)
    return S


# ===========================================================================
# k > 64 pure-Triton singular values (Phase 3)
# ===========================================================================
_SVD_BLOCKED_MAX_K = 512
_SVD_BLOCKED_MAX_ROWS = 2048
_SVD_BLOCKED_CHUNK = 512
# Bisection iterations (each halves the interval): SMALL (k <= 64) needs ~20
# — residual ~1e-5/sigma < atol + rtol*|norm|.  BLOCKED (k <= 512, hi <=
# ~1e2) keeps 27 so the summed nuc error stays in tolerance.
_SVD_SMALL_BISECT_ITERS = 20
_SVD_BLOCKED_BISECT_ITERS = 27
# gk_init/sturm lane width.  Fixed 512: df64 pairs keep ~10 live vectors per
# lane, so BLOCK = N (1024 at K=512) overflows UB under the multi-buffer
# flags; chunked loops cover K <= 512 in <= 2 trips.
_SVD_BLOCKED_GK_BLOCK = 512

_BLOCK_NPROG_CACHE = None


def _block_parallel_programs():
    global _BLOCK_NPROG_CACHE
    if _BLOCK_NPROG_CACHE is None:
        try:
            from triton.backends.ascend.driver import NPUUtils

            _BLOCK_NPROG_CACHE = max(int(NPUUtils().get_aivector_core_num()), 1)
        except Exception:
            _BLOCK_NPROG_CACHE = max(CORE_NUM, 1)
    return _BLOCK_NPROG_CACHE


# ===========================================================================
# Fast two-kernel Golub-Kahan loop (batch == 1, rows <= CHUNK)
# ===========================================================================
# Fuses a step into two launches instead of four (511 vs 1021 at k=256) —
# the per-launch host cost of JITFunction.run dominates the shipped loop.
#   LA_p: left reflection + the previous step's deferred right correction
#         (t -= u[c]*zz[rr]) + per-column partial of the right dot into PB.
#   RED_p: alpha store + cross-program PB reduction + row j+1 update.
# The LA_p->RED_p kernel boundary is the cross-program fence (RED_p reads
# only slots the just-finished LA_p wrote).  MAX_ROWS == CHUNK keeps every
# chunk loop single-trip.
_SVD_BLOCKED_FAST_RCHUNK = 16
_SVD_BLOCKED_FAST_UBLOCK = 512


@libentry()
@triton.jit
def _bidiag_la_p_kernel(
    W,
    U,
    ZZ,
    PB,
    K,
    ROWS,
    J,
    BATCH: tl.constexpr,
    NPROG: tl.constexpr,
    CHUNK: tl.constexpr,
    MAX_ROWS: tl.constexpr,
):
    # Grid-parallel left reflection + deferred right correction + per-column
    # partial of the right dot (see left-apply flatten note).  Empty-column
    # programs still store a zero pvec (RED_p sums every slot).  PB slots are
    # CHUNK-strided, written UNMASKED: a masked 512-lane pvec store miscompiles
    # (masked-off lanes sporadically overwrite adjacent slots).
    WTOT: tl.constexpr = BATCH * NPROG
    for w in range(tl.program_id(0), WTOT, NPROG):
        b = w // NPROG
        cc = w % NPROG
        wbase = W + b * K * ROWS
        ub = U + b * K
        zzb = ZZ + b * ROWS
        pbb = PB + (b * NPROG + cc) * CHUNK
        j = J
        x0 = tl.load(wbase + j * ROWS + j)
        sigmasq = 0.0
        rr = j + tl.arange(0, CHUNK)
        m = rr < ROWS
        for cs in range(0, MAX_ROWS, CHUNK):
            rr = j + cs + tl.arange(0, CHUNK)
            m = rr < ROWS
            x = tl.load(wbase + j * ROWS + rr, mask=m, other=0.0)
            sigmasq += tl.sum(x * x)
        sigma = tl.sqrt(sigmasq)
        alpha = tl.where(x0 >= 0.0, -sigma, sigma)
        vnorm2 = 2.0 * sigma * (sigma + tl.abs(x0))
        tau = tl.where(vnorm2 > 0.0, 2.0 / vnorm2, 0.0)
        pvec = tl.zeros([CHUNK], dtype=tl.float32)
        for c in range(j + 1 + cc, K, NPROG):
            ub_c = tl.load(ub + c)
            t = tl.load(wbase + c * ROWS + rr, mask=m, other=0.0)
            zz = tl.load(zzb + rr, mask=m, other=0.0)
            x = tl.load(wbase + j * ROWS + rr, mask=m, other=0.0)
            tc = t - ub_c * zz
            one = rr < j + 1
            uv = tl.where(one, x0 - alpha, x)
            wvc = tau * tl.sum(tc * uv)
            t0 = tl.load(wbase + c * ROWS + j)
            zz0 = tl.load(zzb + j)
            uval = (t0 - ub_c * zz0) - (x0 - alpha) * wvc
            tl.store(ub + c, uval)
            tnew = tc - uv * wvc
            tl.store(wbase + c * ROWS + rr, tnew, mask=m)
            pvec += tl.where(c >= j + 2, tnew * uval, 0.0)
        tl.store(pbb + tl.arange(0, CHUNK), pvec)


@libentry()
@triton.jit
def _bidiag_red_p_kernel(
    W,
    U,
    PB,
    ZZ,
    K,
    ROWS,
    J,
    BATCH: tl.constexpr,
    NPROG: tl.constexpr,
    RCHUNK: tl.constexpr,
    UBLOCK: tl.constexpr,
    NCHUNKS: tl.constexpr,
    CHUNK: tl.constexpr,
    MAX_ROWS: tl.constexpr,
):
    # LF alpha store + cross-program partial reduction + row j+1 update +
    # alpha2 store (the shipped right-apply/finalize pair in one launch).
    # u-chain scalars (u0/uraw) come from ubuf written by the just-finished
    # LA_p launch; all programs compute bit-identical values.  ps sums the
    # NPROG PB slots written by LA_p (CHUNK-strided, lanes rr - j hold the
    # partial for row rr); zz goes to zzbuf for LA_p_{j+1}'s deferred
    # correction.  RA parts masked by the shipped guard j + 2 < K.
    WTOT: tl.constexpr = BATCH * NCHUNKS
    for w in range(tl.program_id(0), WTOT, NPROG):
        b = w // NCHUNKS
        cc = w % NCHUNKS
        wbase = W + b * K * ROWS
        ub = U + b * K
        zzb = ZZ + b * ROWS
        pbb = PB + b * NPROG * CHUNK
        j = J
        x0 = tl.load(wbase + j * ROWS + j)
        sigmasq = 0.0
        for cs in range(0, MAX_ROWS, CHUNK):
            rr = j + cs + tl.arange(0, CHUNK)
            m = rr < ROWS
            x = tl.load(wbase + j * ROWS + rr, mask=m, other=0.0)
            sigmasq += tl.sum(x * x)
        sigma = tl.sqrt(sigmasq)
        alpha = tl.where(x0 >= 0.0, -sigma, sigma)
        tl.store(wbase + j * ROWS + j, alpha, mask=cc == 0)
        cols = tl.arange(0, UBLOCK)
        c2 = j + 2 + cols
        c2cl = tl.minimum(c2, K - 1)
        cm = c2 < K
        u0 = tl.load(ub + tl.minimum(j + 1, K - 1))
        uraw = tl.load(ub + c2cl, mask=cm, other=0.0)
        sigma2sq = u0 * u0 + tl.sum(uraw * uraw)
        sigma2 = tl.sqrt(sigma2sq)
        alpha2 = tl.where(u0 >= 0.0, -sigma2, sigma2)
        vnorm3 = 2.0 * sigma2 * (sigma2 + tl.abs(u0))
        tau2 = tl.where(vnorm3 > 0.0, 2.0 / vnorm3, 0.0)
        uadj = u0 - alpha2
        ra_guard = j + 2 < K
        rr = j + 1 + cc * RCHUNK + tl.arange(0, RCHUNK)
        rm = rr < ROWS
        rcl = tl.minimum(rr, ROWS - 1)
        ps = tl.zeros([RCHUNK], dtype=tl.float32)
        for p in range(0, NPROG):
            ps += tl.load(pbb + p * CHUNK + (rcl - j), mask=rm, other=0.0)
        trow = tl.load(
            wbase + tl.minimum(j + 1, K - 1) * ROWS + rcl,
            mask=rm,
            other=0.0,
        )
        zz = tau2 * (ps + trow * uadj)
        tl.store(zzb + rcl, zz, mask=rm & ra_guard)
        tl.store(
            wbase + tl.minimum(j + 1, K - 1) * ROWS + rcl,
            trow - zz * uadj,
            mask=rm & ra_guard,
        )
        tl.store(
            wbase + (j + 1) * ROWS + j,
            alpha2,
            mask=(cc == 0) & ra_guard,
        )


@triton.jit
def _w4_chain_step(e2h, e2l, qh, ql, xh, xl):
    # one step of the qd recurrence for ONE scalar chain (see the shipped
    # _gk_sturm_count_less -- identical df64 arithmetic)
    rh, rl = _df64_div_ds(e2h, e2l, qh, ql)
    qh2, ql2 = _df64_add(-xh, -xl, -rh, -rl)
    zero_q = (qh2 == 0.0) & (ql2 == 0.0)
    qh2 = tl.where(zero_q, -1.1754944e-38, qh2)
    ql2 = tl.where(zero_q, 0.0, ql2)
    return qh2, ql2, tl.where(qh2 < 0.0, 1, 0)


@triton.jit
def _w4_count_way4(
    E2H, E2L, base, N: tl.constexpr, x0h, x0l, x1h, x1l, x2h, x2l, x3h, x3l
):
    # four interleaved scalar qd chains sharing the same e2h/e2l scalar
    # loads: the scheduler overlaps chain A's division with chain B's adds
    # (real ILP where the SIMD-vector attempt stayed latency-bound).
    q0h, q0l = -x0h, -x0l
    q1h, q1l = -x1h, -x1l
    q2h, q2l = -x2h, -x2l
    q3h, q3l = -x3h, -x3l
    z0 = (q0h == 0.0) & (q0l == 0.0)
    q0h = tl.where(z0, -1.1754944e-38, q0h)
    q0l = tl.where(z0, 0.0, q0l)
    z1 = (q1h == 0.0) & (q1l == 0.0)
    q1h = tl.where(z1, -1.1754944e-38, q1h)
    q1l = tl.where(z1, 0.0, q1l)
    z2 = (q2h == 0.0) & (q2l == 0.0)
    q2h = tl.where(z2, -1.1754944e-38, q2h)
    q2l = tl.where(z2, 0.0, q2l)
    z3 = (q3h == 0.0) & (q3l == 0.0)
    q3h = tl.where(z3, -1.1754944e-38, q3h)
    q3l = tl.where(z3, 0.0, q3l)
    n0 = tl.where(q0h < 0.0, 1, 0)
    n1 = tl.where(q1h < 0.0, 1, 0)
    n2 = tl.where(q2h < 0.0, 1, 0)
    n3 = tl.where(q3h < 0.0, 1, 0)
    for i in range(1, N):
        e2h = tl.load(E2H + base + i - 1)
        e2l = tl.load(E2L + base + i - 1)
        q0h, q0l, d0 = _w4_chain_step(e2h, e2l, q0h, q0l, x0h, x0l)
        q1h, q1l, d1 = _w4_chain_step(e2h, e2l, q1h, q1l, x1h, x1l)
        q2h, q2l, d2 = _w4_chain_step(e2h, e2l, q2h, q2l, x2h, x2l)
        q3h, q3l, d3 = _w4_chain_step(e2h, e2l, q3h, q3l, x3h, x3l)
        n0 += d0
        n1 += d1
        n2 += d2
        n3 += d3
    return n0, n1, n2, n3


@libentry()
@triton.jit
def _sturm_sigmas_w4_kernel(
    E2H,
    E2L,
    S,
    K: tl.constexpr,
    NPROG: tl.constexpr,
    BISECT_ITERS: tl.constexpr,
    N: tl.constexpr,
    BATCH: tl.constexpr,
    NGROUPS: tl.constexpr,
):
    # Sturm bisection with four interleaved qd chains per program
    # (targets 4g..4g+3, g = w % NGROUPS).  Same df64 recurrence and
    # bisection bounds as the shipped sturm kernel, so the counts and the
    # final sigmas are identical (bit-for-bit, verified).
    WTOT: tl.constexpr = BATCH * NGROUPS
    for w in range(tl.program_id(0), WTOT, NPROG):
        b = w // NGROUPS
        g = w % NGROUPS
        base = b * N
        emax = 0.0
        for i in range(1, N):
            emax = tl.maximum(emax, tl.load(E2H + base + i - 1))
        emax = tl.sqrt(emax)
        hi0 = 2.0 * emax * (1.0 + 1e-9) + 1e-292
        j0 = tl.minimum(4 * g, K - 1)
        j1 = tl.minimum(4 * g + 1, K - 1)
        j2 = tl.minimum(4 * g + 2, K - 1)
        j3 = tl.minimum(4 * g + 3, K - 1)
        lo0 = 0.0
        hi1 = hi0
        lo1 = 0.0
        hi2 = hi0
        lo2 = 0.0
        hi3 = hi0
        lo3 = 0.0
        t0 = K + j0
        t1 = K + j1
        t2 = K + j2
        t3 = K + j3
        for it in range(BISECT_ITERS):
            mid0 = 0.5 * (lo0 + hi1)
            mid1 = 0.5 * (lo1 + hi2)
            mid2 = 0.5 * (lo2 + hi3)
            mid3 = 0.5 * (lo3 + hi0)
            x0h, x0l = _split_f32(mid0)
            x1h, x1l = _split_f32(mid1)
            x2h, x2l = _split_f32(mid2)
            x3h, x3l = _split_f32(mid3)
            c0, c1, c2, c3 = _w4_count_way4(
                E2H, E2L, base, N, x0h, x0l, x1h, x1l, x2h, x2l, x3h, x3l
            )
            if c0 >= t0 + 1:
                hi1 = mid0
            else:
                lo0 = mid0
            if c1 >= t1 + 1:
                hi2 = mid1
            else:
                lo1 = mid1
            if c2 >= t2 + 1:
                hi3 = mid2
            else:
                lo2 = mid2
            if c3 >= t3 + 1:
                hi0 = mid3
            else:
                lo3 = mid3
        tl.store(S + b * K + j0, 0.5 * (lo0 + hi1), mask=4 * g < K)
        tl.store(S + b * K + j1, 0.5 * (lo1 + hi2), mask=4 * g + 1 < K)
        tl.store(S + b * K + j2, 0.5 * (lo2 + hi3), mask=4 * g + 2 < K)
        tl.store(S + b * K + j3, 0.5 * (lo3 + hi0), mask=4 * g + 3 < K)


def _svdvals_blocked(A):
    *batch_dims, M, N = A.shape
    batch = math.prod(batch_dims)
    k, rows = min(M, N), max(M, N)
    dev = A.device

    work = torch.zeros((batch, k, rows), dtype=torch.float32, device=dev)
    src = A.reshape(batch, M, N)
    if M >= N:
        src = src.transpose(-2, -1)
    work[:, :k, :] = src

    max_rows = 1 << (rows - 1).bit_length()
    chunk = _SVD_BLOCKED_CHUNK
    nprog = _block_parallel_programs()

    if batch == 1 and rows <= chunk:
        # Fast two-kernel loop (LA_p + RED_p per step), MAX_ROWS == CHUNK so
        # every chunk loop is single-trip.
        ubuf = torch.zeros((batch, k), dtype=torch.float32, device=dev)
        zzbuf = torch.zeros((batch, rows), dtype=torch.float32, device=dev)
        pbuf = torch.zeros((batch, nprog, chunk), dtype=torch.float32, device=dev)
        rchunk = _SVD_BLOCKED_FAST_RCHUNK
        nchunks = (chunk + rchunk - 1) // rchunk
        for j in range(k - 1):
            _bidiag_la_p_kernel[(nprog,)](
                work,
                ubuf,
                zzbuf,
                pbuf,
                k,
                rows,
                j,
                BATCH=batch,
                NPROG=nprog,
                CHUNK=chunk,
                MAX_ROWS=chunk,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=True,
            )
            _bidiag_red_p_kernel[(min(nprog, batch * nchunks),)](
                work,
                ubuf,
                pbuf,
                zzbuf,
                k,
                rows,
                j,
                BATCH=batch,
                NPROG=nprog,
                RCHUNK=rchunk,
                UBLOCK=_SVD_BLOCKED_FAST_UBLOCK,
                NCHUNKS=nchunks,
                CHUNK=chunk,
                MAX_ROWS=chunk,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=True,
            )
        _bidiag_left_finalize_kernel[(min(nprog, batch),)](
            work,
            k,
            rows,
            k - 1,
            BATCH=batch,
            NPROG=nprog,
            CHUNK=chunk,
            MAX_ROWS=chunk,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=True,
        )
    else:
        # 4-kernel loop (batch > 1 / rows > CHUNK).  RA's NCHUNKS constexpr
        # varies with j.
        for j in range(k):
            if j < k - 1:
                _bidiag_left_apply_kernel[(min(nprog, batch * nprog),)](
                    work,
                    k,
                    rows,
                    j,
                    BATCH=batch,
                    NPROG=nprog,
                    CHUNK=chunk,
                    MAX_ROWS=max_rows,
                    num_warps=4,
                    num_stages=1,
                    enable_fp_fusion=True,
                )
            _bidiag_left_finalize_kernel[(min(nprog, batch),)](
                work,
                k,
                rows,
                j,
                BATCH=batch,
                NPROG=nprog,
                CHUNK=chunk,
                MAX_ROWS=max_rows,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=True,
            )
            if j + 2 < k:
                nchunks = (rows - (j + 1) + chunk - 1) // chunk
                _bidiag_right_apply_kernel[(min(nprog, batch * nchunks),)](
                    work,
                    k,
                    rows,
                    j,
                    BATCH=batch,
                    NPROG=nprog,
                    NCHUNKS=nchunks,
                    CHUNK=chunk,
                    MAX_ROWS=max_rows,
                    num_warps=4,
                    num_stages=1,
                    enable_fp_fusion=True,
                )
                _bidiag_right_finalize_kernel[(min(nprog, batch),)](
                    work,
                    k,
                    rows,
                    j,
                    BATCH=batch,
                    NPROG=nprog,
                    CHUNK=chunk,
                    num_warps=4,
                    num_stages=1,
                    enable_fp_fusion=True,
                )

    n = 2 * k
    e2h = torch.empty((batch, n), dtype=torch.float32, device=dev)
    e2l = torch.empty((batch, n), dtype=torch.float32, device=dev)
    S = torch.empty((batch, k), dtype=torch.float32, device=dev)
    _gk_init_kernel[(min(nprog, batch),)](
        work,
        e2h,
        e2l,
        rows,
        K=k,
        BATCH=batch,
        NPROG=nprog,
        BLOCK=_SVD_BLOCKED_GK_BLOCK,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    ngroups = (k + 3) // 4
    _sturm_sigmas_w4_kernel[(min(nprog, batch * ngroups),)](
        e2h,
        e2l,
        S,
        K=k,
        NPROG=nprog,
        BISECT_ITERS=_SVD_BLOCKED_BISECT_ITERS,
        N=n,
        BATCH=batch,
        NGROUPS=ngroups,
        num_warps=1,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return S


# ===========================================================================
# Entry point
# ===========================================================================

_SUPPORTED_NUMERIC = {2, -2}


def _ord2_norm(A, ord_val, dim, keepdim, dtype):
    d0, d1 = dim
    d0, d1 = d0 % A.ndim, d1 % A.ndim
    out_dtype = dtype if dtype is not None else A.dtype

    if A.ndim == 2 and d0 == 0 and d1 == 1:
        M, N = A.shape
        k, rows = min(M, N), max(M, N)
        mode = "max" if float(ord_val) > 0 else "min"
        if k == 1:
            Ain = A.float() if A.dtype in (torch.float16, torch.bfloat16) else A
            Ain = Ain.contiguous()
            result = torch.empty(1, dtype=torch.float32, device=A.device)
            blk_n = triton.next_power_of_2(min(M * N, 512))
            _fro_kernel[(1,)](
                Ain.reshape(1, M * N),
                result,
                0,
                M * N,
                1,
                blk_n,
                1,
                TILE_2D=False,
                USE_FP64=False,
                num_warps=8,
            )
            result = result.reshape(())
        elif k == 2 and rows <= _RANK2_BLOCK_R_MAX:
            result = _rank2_norm_fast(A, mode)
        elif k == 3 and rows <= _RANK3_BLOCK_R_MAX:
            result = _rank3_norm_fast(A, mode)
        elif 3 <= k <= _TINY_MAX_K and rows <= _TINY_MAX_ROWS:
            result = _svd_norm_tiny(A, mode)
        elif 2 < k <= 512 and rows <= 2048:
            if _use_gram(k, rows):
                result = _gram_norm_fast(A, mode)
            else:
                result = _dim1_reduce(_svdvals_for_norm(A), mode)
        else:
            raise NotImplementedError(
                f"FlagGems Ascend matrix_norm ord={ord_val}: "
                f"unsupported shape {A.shape}"
            )
        if keepdim:
            result = result.reshape(1, 1)
        if result.dtype != out_dtype:
            result = result.to(out_dtype)
        return result

    ndim = A.ndim
    remaining = [d for d in range(ndim) if d not in (d0, d1)]
    perm = remaining + [d0, d1]
    A_perm = A.permute(perm) if perm != list(range(ndim)) else A
    if dtype is not None:
        A_perm = A_perm.to(dtype)
    mode = "max" if float(ord_val) > 0 else "min"
    k, rows = min(A_perm.shape[-2], A_perm.shape[-1]), max(
        A_perm.shape[-2], A_perm.shape[-1]
    )
    if k == 2 and rows <= _RANK2_BLOCK_R_MAX:
        result = _rank2_norm_fast(A_perm, mode)
    elif k == 3 and rows <= _RANK3_BLOCK_R_MAX:
        result = _rank3_norm_fast(A_perm, mode)
    elif (
        3 <= k <= _TINY_MAX_K
        and rows <= _TINY_MAX_ROWS
        and math.prod(A_perm.shape[:-2]) <= _block_parallel_programs()
    ):
        result = _svd_norm_tiny(A_perm, mode)
    else:
        if _use_gram(k, rows):
            result = _gram_norm_fast(A_perm, mode)
        else:
            result = _dim1_reduce(_svdvals_for_norm(A_perm), mode)
    if result.dtype != out_dtype:
        result = result.to(out_dtype)
    if keepdim:
        out_shape = list(A.shape)
        out_shape[d0] = out_shape[d1] = 1
        result = result.reshape(out_shape)
    return result


def _nuc_norm(A, dim, keepdim=False, dtype=None):
    out_dtype = dtype if dtype is not None else A.dtype
    k, rows = min(A.shape[-2], A.shape[-1]), max(A.shape[-2], A.shape[-1])
    if k == 2 and rows <= _RANK2_BLOCK_R_MAX:
        result = _rank2_norm_fast(A, "sum")
    elif k == 3 and rows <= _RANK3_BLOCK_R_MAX:
        result = _rank3_norm_fast(A, "sum")
    elif (
        3 <= k <= _TINY_MAX_K
        and rows <= _TINY_MAX_ROWS
        and math.prod(A.shape[:-2]) <= _block_parallel_programs()
    ):
        result = _svd_norm_tiny(A, "sum")
    else:
        if _use_gram(k, rows):
            result = _gram_norm_fast(A, "sum")
        else:
            s = _svdvals_for_norm(A)
            if s.shape[-1] == 1:
                # k==1: drop the trailing size-1 dim (squeeze handles 2D,
                # where reshape(*()) raises on empty s.shape[:-1])
                result = s.squeeze(-1)
            else:
                result = _dim1_reduce(s, "sum")
    if result.dtype != out_dtype:
        result = result.to(out_dtype)
    if keepdim:
        d0, d1 = dim[0], dim[1]
        out_shape = list(A.shape)
        out_shape[d0] = out_shape[d1] = 1
        result = result.reshape(out_shape)
    return result


def linalg_matrix_norm(A, ord="fro", dim=(-2, -1), keepdim=False, dtype=None):
    logger.debug("GEMS ASCEND LINALG_MATRIX_NORM")

    d0, d1 = dim
    d0, d1 = d0 % A.ndim, d1 % A.ndim
    if d0 == d1:
        raise RuntimeError(
            f"linalg_matrix_norm: dims must be different, got ({dim[0]}, {dim[1]})"
        )
    wrapped_dim = [d0, d1]

    if A.dtype == torch.float64:
        raise RuntimeError(
            "FlagGems Ascend linalg_matrix_norm:: does no support float64"
        )
    if dtype is not None and dtype == torch.float64:
        raise RuntimeError(
            "FlagGems Ascend linalg_matrix_norm:: dtype does no support float64"
        )
    if isinstance(ord, str) and ord == "nuc":
        return _nuc_norm(A, wrapped_dim, keepdim=keepdim, dtype=dtype)

    ord_float = float(ord)
    if ord_float not in _SUPPORTED_NUMERIC:
        raise RuntimeError(
            f"FlagGems Ascend linalg_matrix_norm: Order {ord} not supported. "
            "Use 2, -2, nuc."
        )
    abs_ord = abs(ord_float)
    if abs_ord == 2.0:
        return _ord2_norm(A, ord_float, wrapped_dim, keepdim, dtype)
    raise NotImplementedError(
        f"FlagGems Ascend linalg_matrix_norm: unsupported ord '{ord}'"
    )
