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
    """One full Brent-Luk one-sided Jacobi sweep on the column-major work
    matrix (batch, KS, ROWS), single program per matrix (grid=(batch,)).

    KS = K + K % 2: odd K gets one zero dummy column so the ring column
    count is even; pairs touching the dummy rotate ~identity (t -> 0 on the
    zero column), so every real pair is still annihilated once per sweep.
    The j == 0 ring pair is written as its own block because scalar
    (x == const) / integer selects miscompile on this toolchain — the
    pairing here uses no selects at all.  Columns are loaded and rotated in
    CHUNK-sized slices so tall matrices (ROWS up to 2048) stay inside UB
    (full-column tiles overflow it); the ring pairs of one step are
    disjoint, so a single debug_barrier at the end of the step fences the
    MTE3 stores before the next step's MTE2 loads.
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
):
    # One program per (batch element, row-chunk) work item, 1D grid <= the
    # AIV block count striding the items (see the blocked-path header for
    # why grids must stay at/below the physical block count).
    # Golub-Kahan bidiagonalization of the
    # contiguous (ROWS, K) work tensor (A for tall inputs, A^T for wide
    # ones); rows beyond K are real data for tall inputs, so column
    # extraction runs over the full tile height.  The (BLOCK, BLOCK)
    # register tile is loaded UNMASKED with clamped addressing (min with
    # ROWS-1 / K-1 keeps every address in bounds — a masked 2D load
    # miscompiles on this toolchain) and the padding (rows >= ROWS or
    # cols >= K) is zeroed in-register on load, so the host passes the
    # caller's tensor directly: no padded workspace, no corner copy.
    # The GK tridiagonal seed [d0^2, s0^2, d1^2, s1^2, ...] (double-single)
    # is written to E2H/E2L in the same launch — a separate gk_init launch
    # would dominate at these launch-bound shapes.
    #
    # Everything stays in the LINEAR domain: the Sturm bisection that reads
    # E2 below computes sigma_min at full relative precision, which the
    # Gram-square route (A^T A) loses to fp32 round-off.
    #
    # Two-sided Householder reflections; only reshape-based outer products
    # (the only broadcast form that is numerically correct on this backend)
    # and stable primitives (axis-1 masked reductions, scalar-blend into
    # vectors via where, float-mask multiplies).  The right reflection takes
    # row j via a (BLOCK, 1) row mask + axis-0 reduction (a (1, BLOCK)
    # column mask reduces to a one-hot and silently turns the reflection
    # into a no-op — the matrix_rank branch has exactly that bug).
    #
    # enable_fp_fusion MUST be False: the fused tail calls _split_f32
    # (Veltkamp), which a contracted t - (t - a) degrades to plain fp32.
    for b in range(tl.program_id(0), BATCH, NPROG):
        batch = b
        rows = tl.arange(0, BLOCK)
        cols = tl.arange(0, BLOCK)
        rrow = tl.minimum(rows, ROWS - 1)
        ccol = tl.minimum(cols, K - 1)
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
    # One Golub-Kahan left Householder reflection step (column j): zero
    # column j at rows > j and update columns j+1..K-1 with 1D CHUNK-wide
    # vectors only (non-square 2D tiles with reshape-broadcast products
    # miscompile deterministically on this toolchain, square tiles overflow
    # UB inside this loop nest).  Column j is never written until the very
    # end, so the dot passes re-read a stable value: this kernel has NO
    # intra-kernel store->load dependency, which is what makes it correct
    # — the toolchain's MTE2-store / MTE3-load reordering is unordered
    # within a program and tl.debug_barrier is a no-op, so any in-kernel
    # producer/consumer round-trip races.  The only reliable fence is the
    # kernel boundary: the host launches one left step and one right step
    # per j.  1D grid <= the AIV block count striding the batch elements.
    #
    # Mask lesson: a TWO-bound mask (rr >= j) & (rr < ROWS) on a
    # fixed-offset vector miscompiles deterministically once the active
    # lane count drops below 8 (silently corrupting the masked
    # loads/stores, no error).  The safe form anchors the vector at j:
    # rr = j + arange(0, CHUNK) with a single upper-bound mask rr < ROWS
    # (verified bit-for-bit against numpy at every step).
    #
    # Loop-bound lesson: the row-chunk loop must have a COMPILE-TIME trip
    # count.  A runtime bound range(0, ROWS, CHUNK) miscompiles whenever
    # the trip count exceeds 1 (rows=1024/2048 corrupt, rows<=CHUNK fine).
    # MAX_ROWS = next_power_of_2(ROWS) is passed as constexpr; iterations
    # whose rows land at >= ROWS are fully masked off and cost only the
    # predicated-off lanes.
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
    # One Golub-Kahan right Householder reflection step (row j): zero
    # W[j, j+2:] with the alpha-adjusted form; 1D grid <= the AI core
    # count striding the batch elements.
    # (u = x - alpha*e, alpha = -sign(x0)*sigma) — the plain form (u = x)
    # only reflects (H x = -x) and zeroes nothing.  The superdiagonal
    # column j+1 is handled explicitly with the adjusted u (u0 - alpha2)
    # so no scalar select is needed; the other columns use scalar gathers
    # of row j.  Like the left step, this kernel has no intra-kernel
    # store->load round-trip: it only READS what the previous (left-step)
    # launch wrote, and each row's stores are disjoint from every other
    # row's loads.  DO_RIGHT is a constexpr guard (compiled away) so the
    # body never runs on an empty row slice — a 1-element reflection would
    # sign-flip the superdiagonal instead of doing nothing.
    #
    # Aliasing lesson: the apply must EXCLUDE row j itself.  Row j holds
    # the reflection vector u (the per-chunk scalar gathers read W[c, j]),
    # and if the apply also stored row j it would zero those entries mid-
    # kernel; with rows > CHUNK the next chunk's gathers would read the
    # zeroed values (this corrupted rows=1024/2048 before).  Row j is
    # instead finalized at the end: H maps it to alpha2 at column j+1 and
    # zeros at columns j+2..K-1, which is written directly (exact alpha2,
    # no cancellation through t - zz*u).
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
# Tiling-axes lesson: the serial Phase 2 kernels hold a runtime-bounded
# column loop whose trip count reaches ~K; for K > 64 the ascend autotuner
# tries to tile it and dies with "Number of parameters exceeds the number
# of available axes".  Grid-parallel kernels (one program per column /
# per row chunk) keep every kernel at <= 2 tiling parameters (one loop +
# the CHUNK vector) and compile for any K.  All launched kernels are
# @libentry()-wrapped: on this ascend fork a plain @triton.jit launch
# re-parses the kernel AST per call (~0.4 ms/launch), while libentry's
# per-shape cache launches in ~0.05 ms — decisive at these launch-bound
# shapes.  (The compile path is the same JITFunction either way.)
#
# Geometry lesson: the Phase 2/3 bidiagonalization produces a LOWER
# bidiagonal B (nonzero band = diagonal + SUBdiagonal -- the right
# reflection finalizes W[j+1, j] = alpha2).  The Golub-Kahan tridiagonal
# of order 2K (zero diagonal, off-diagonal [d0, s0, d1, s1, ..., d_{K-1}]
# with s = subdiagonal) has eigenvalues exactly +/- sigma_i.  Using the
# superdiagonal (all zeros) silently yields the eigenvalues of the wrong
# matrix -- verified: sigma errors ~46% until this fix.
#
# Precision: the Sturm count runs the qd recurrence in double-single fp32
# pairs (Veltkamp split + TwoSum/TwoProd; BiSheng rejects tl.float64, and
# native fp64 division is a slow software sequence).  Launched with
# enable_fp_fusion=False so the exact-pair identities survive.  Measured
# sigma error ~4e-6 at K=512 (30-80x inside the test tolerances).
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
    # Grid-parallel left reflection (see the blocked-path header): a 1D
    # grid of at most the AIV block count, each program striding over the
    # (batch element, column slot) work items w = pid, pid+NPROG, ...
    # Slot cc applies the step-j reflection to the columns c = j+1+cc,
    # j+1+cc+NPROG, ... of batch element b = w // NPROG.  The grid must
    # stay 1D and <= the AIV block count: a 2D (batch, NPROG) grid exceeds
    # the physical block count whenever batch >= 2, and the auto-map
    # wrapper that engages then does not fence MTE3 stores across kernel
    # boundaries, so the dependent next launch reads stale rows (the
    # batched k > 64 ord=2/-2/nuc corruption).  Slots with j+1+cc >= K
    # have an empty column range and only re-read row j (no stores).
    # Row j is untouched (finalized by a separate launch -- the kernel
    # boundary is the only reliable fence).  sigma/alpha/tau are computed
    # once per program from row j with the identical load order and
    # reduction tree, so every program sees bit-identical scalars.  Same
    # mask/loop-bound discipline as the Phase 2 kernels: vectors anchored
    # at the lower bound with a single upper-bound mask, compile-time
    # chunk trip count MAX_ROWS.
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
    # Row j of the workspace: H x_j = alpha * e_j.  Runs after the apply
    # launch; row j is not written by any apply program, so there is no
    # cross-program race.  1D grid <= the AIV block count striding the
    # batch elements (see the left-apply flatten note).
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
    # striding the (batch element, row chunk) work items (see the
    # left-apply flatten note).  Item (b, cs) applies the step-j
    # reflection to the workspace columns rr of its chunk (rr >= j+1; the
    # aliasing lesson -- row j holds the reflection vector and must not be
    # written while gathers of it are pending).  The u vector (column j
    # below row j+1) is read once per program via scalar gathers.
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
    # alpha2, no cancellation through t - zz*u), after all the gathers of
    # row j and column j are done.  1D grid <= the AIV block count striding
    # the batch elements (see the left-apply flatten note).
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
    # Golub-Kahan tridiagonal of order N = 2K for the lower-bidiagonal
    # corner B = W[:, :K, :K]: zero diagonal (implicit), off-diagonal
    # [d0, s0, d1, s1, ..., d_{K-1}] with d = diagonal and s = SUBdiagonal
    # of B.  Eigenvalues are exactly +/- sigma_i.  Each squared
    # off-diagonal is stored as a double-single fp32 pair (E2H/E2L),
    # interleaved the same way.  Reads B straight from W -- no separate
    # diagonal-extraction pass, no store->load round trip in-kernel.
    # BLOCK lanes are reused across chunks of 2*BLOCK interleaved
    # positions: the df64 pairs keep ~10 live vectors per lane, and
    # BLOCK = N (i.e. 1024 lanes at K=512) overflows the UB under the
    # multi-buffer compile options that TRITON_ALL_BLOCKS_PARALLEL adds.
    # 1D grid <= the AIV block count striding the batch elements.
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
    # 1D grid <= the AIV block count striding the (batch element, column
    # slot) work items w = pid, pid+NPROG, ... (see the left-apply
    # flatten note — a 2D (NPROG, batch) grid races with the dependent
    # reduction that reads S).  Item (b, cc) bisects eigenvalue indices
    # K+j for j = cc, cc+NPROG, ... (ascending order): the j-th value of
    # the positive half sorted ascending -- S[j] runs sigma_min ..
    # sigma_max, with zeros occupying the front slots for rank-deficient
    # matrices (exactly what the norm reductions need).  lo = 0 is a
    # valid lower bound for every target (count(0) = K + #{zero sigma}
    # <= K + j); Gershgorin gives hi = 2*max|g| above every eigenvalue.
    # The bisection loop is scalar-only (no stores inside the
    # data-dependent if), BISECT_ITERS is constexpr.
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
    # Max/min/sum over the last dim of S (batch, K).  Torch computational
    # ops are forbidden on this path (and the flag_gems dispatchers cost a
    # fixed ~0.17 ms each — dominant at these tiny shapes), so the
    # reduction runs in one fast launch: single masked BLOCK-lane load,
    # neutral `other` per mode.  1D grid <= the AIV block count striding the
    # batch elements (S is written by a grid-flattened producer — see the
    # sturm flatten note).  K <= 512.
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
    # Closed-form k=2 singular values with the norm reduction fused in:
    # both sigmas are scalars in-register, so max/min/sum costs nothing.
    # A separate reduction launch would dominate at the launch-bound
    # (2, 2048) / (2048, 2) benchmark shapes.  1D grid <= the AI core
    # count striding the batch elements.
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
        torch.npu.synchronize()
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

# Reused intermediate buffers for the small path, keyed by (batch, size,
# device): the benchmark harness loops one shape per run, and each
# torch.empty dispatch costs ~0.02 ms — at the launch-bound
# (8, 8) / (4, 32, 64) shapes the host side rivals the kernels.  Only
# INTERMEDIATE buffers are cached (E2 seeds) — S, the exposed output, is
# always fresh so successive calls never alias.
_SVD_WORKSPACE_CACHE = {}
_SVD_WORKSPACE_CACHE_MAX = 8


def _svdvals_small(A):
    """Pure-Triton singular values for 3 <= k <= 64, rows <= 2048 (fp32).

    rows <= 64: Golub-Kahan bidiagonalization of the caller's (rows, k)
        tensor loaded directly into a (BLOCK, BLOCK) register tile
        (UB-resident, batch parallel, clamped-address unmasked load),
        followed by Sturm bisection on the Golub-Kahan tridiagonal (the
        E2 seed is produced in the same bidiag launch).  The whole chain
        stays in the linear domain, so sigma_min keeps full relative
        precision (the Gram route squares the spectrum and loses it).
    rows >  64: row-chunked Golub-Kahan bidiagonalization of the
        column-major (batch, k, rows) workspace followed by the same
        Sturm bisection on the bidiagonal corner.
    """
    *batch_dims, M, N = A.shape
    batch = math.prod(batch_dims)
    k, rows = min(M, N), max(M, N)
    tall = M >= N
    dev = A.device
    nprog = _block_parallel_programs()

    if rows <= 64:
        block = max(triton.next_power_of_2(max(rows, k)), 32)
        # work holds the columns to be rotated: A for tall inputs, A^T for
        # wide ones; (batch, rows, k) row-major and contiguous — the
        # kernel reads it directly with clamped-address unmasked loads.
        work = (
            A.reshape(batch, rows, k)
            if tall
            else A.reshape(batch, M, N).transpose(-2, -1)
        ).contiguous()
        S = torch.empty((batch, k), dtype=torch.float32, device=dev)
        # GK tridiagonal + Sturm bisection on the bidiagonal B (upper
        # bidiagonal corner of the padded tile: d at [i, i], s at [i, i+1]).
        # The E2 seed (interleaved double-single squares) is produced inside
        # the bidiag launch itself — a separate gk_init launch would
        # dominate at these launch-bound shapes.
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
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
        snprog = min(nprog, k)
        _sturm_sigmas_kernel[(min(nprog, batch * snprog),)](
            e2h,
            e2l,
            S,
            K=k,
            NPROG=snprog,
            BISECT_ITERS=_SVD_SMALL_BISECT_ITERS,
            N=n,
            BATCH=batch,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    else:
        # Row-chunked bidiagonalization of the column-major workspace
        # (batch, k, rows), then ring sweeps on the k x k bidiagonal B.
        # The fresh zero buffer keeps the caller's tensor untouched.
        # Columns of A for tall inputs, rows of A (= columns of A^T) for
        # wide ones.  B's corner is copied into (batch, ks, k) — ks adds
        # one zero dummy column for odd k so the ring pairing needs no
        # scalar selects (which miscompile on this toolchain).
        work = torch.zeros((batch, k, rows), dtype=torch.float32, device=dev)
        src = A.reshape(batch, M, N)
        if tall:
            src = src.transpose(-2, -1)
        work[:, :k, :] = src
        # One launch per (j, reflection-half): the only fence that orders
        # the toolchain's MTE2 stores against MTE3 loads is a kernel
        # boundary, so every producer/consumer pair is split into its own
        # launch.  The left reflection runs for EVERY j including j = k-1
        # (dgebd2-style): the final one zeroes column k-1 below the
        # diagonal, making W = [B; 0] so that sigma(B) == sigma(A).
        # MAX_ROWS must be a compile-time upper bound on rows (the chunk
        # loop miscompiles under a runtime bound for trip counts > 1).
        max_rows = 1 << (rows - 1).bit_length()
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
        # GK tridiagonal + Sturm bisection on the bidiagonal corner of the
        # column-major workspace.  _gk_init_kernel reads d at [i, i] and
        # the bidiagonal off-diagonal at (i+1)*rows + i = B[i, i+1] (the
        # SUPERdiagonal these step kernels produce — same layout and
        # orientation as the blocked path).  No dummy column, no Jacobi.
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
        snprog = min(nprog, k)
        _sturm_sigmas_kernel[(min(nprog, batch * snprog),)](
            e2h,
            e2l,
            S,
            K=k,
            NPROG=snprog,
            BISECT_ITERS=_SVD_BLOCKED_BISECT_ITERS,
            N=n,
            BATCH=batch,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return S


# ===========================================================================
# k > 64 pure-Triton singular values (Phase 3)
# ===========================================================================
_SVD_BLOCKED_MAX_K = 512
_SVD_BLOCKED_MAX_ROWS = 2048
_SVD_BLOCKED_CHUNK = 512
# Bisection iterations.  hi <= 2*max|g|, so each iter halves the interval:
# the SMALL path (k <= 64) needs only ~20 iters — the residual ~1e-5 abs
# per sigma is far below atol + rtol*|norm| (~1e-3 for the nuc sums of the
# test shapes).  The BLOCKED path (k up to 512, hi up to ~1e2) keeps 27 so
# the summed nuc error stays inside tolerance.
_SVD_SMALL_BISECT_ITERS = 20
_SVD_BLOCKED_BISECT_ITERS = 27
# Lane width of the gk_init / sturm vector ops.  Fixed at 512 lanes: the
# df64 pairs keep ~10 live vectors per lane, so BLOCK = N (1024 at K=512)
# overflows the UB under the multi-buffer compile options that the
# TRITON_ALL_BLOCKS_PARALLEL flag adds (see constraint list).  With the
# chunked loops, 512 lanes covers every K <= 512 in at most 2 trips.
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
    _sturm_sigmas_kernel[(min(nprog, batch * nprog),)](
        e2h,
        e2l,
        S,
        K=k,
        NPROG=nprog,
        BISECT_ITERS=_SVD_BLOCKED_BISECT_ITERS,
        N=n,
        BATCH=batch,
        num_warps=4,
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
            torch.npu.synchronize()
            result = result.reshape(())
        elif k == 2 and rows <= _RANK2_BLOCK_R_MAX:
            result = _rank2_norm_fast(A, mode)
        elif 2 < k <= 512 and rows <= 2048:
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
    else:
        s = _svdvals_for_norm(A)
        if s.shape[-1] == 1:
            # k==1: drop the trailing size-1 dim (squeeze handles the 2D case,
            # where s.shape[:-1] is empty and reshape(*()) would raise)
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
