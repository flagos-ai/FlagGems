import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

SVDResult = namedtuple("SVDResult", ["U", "S", "V"])

# Maximum matrix dimension supported by native Triton SVD kernels.
# Tested correct up to N=1024; BLOCK_N must be power-of-2, so practical
# upper bound is 1024 (32 warps).
MAX_SVD_DIM = 1024

# Routing thresholds:
# k <= _JACOBI_THRESHOLD: Jacobi SVD (O(N^4), but fast for tiny matrices)
# _JACOBI_THRESHOLD < k <= MAX_SVD_DIM: Bidiagonal SVD (O(N^3))
# k > MAX_SVD_DIM: cuSOLVER fallback (safety net for very large matrices)
_JACOBI_THRESHOLD = 16


def _next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


@libentry()
@triton.jit
def jacobi_svd_kernel(
    A_ptr,
    A_work_ptr,
    V_work_ptr,
    U_ptr,
    S_ptr,
    V_ptr,
    # Input A strides
    batch_stride_A,
    m_stride_A,
    n_stride_A,
    # A_work strides (column-major: A_work[batch, col, row])
    aw_batch_stride,
    aw_col_stride,
    # V_work strides (column-major: V_work[batch, col, row])
    vw_batch_stride,
    vw_col_stride,
    # Output strides
    batch_stride_U,
    m_stride_U,
    k_stride_U,
    batch_stride_S,
    batch_stride_V,
    n_stride_V,
    k_stride_V,
    # Runtime dimension for dynamic loop bounds (prevents unrolling)
    N_dim,
    num_sweeps,
    # Constexpr dimensions for block sizing and output
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    out_k_U: tl.constexpr,
    out_k_V: tl.constexpr,
    compute_uv: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One-sided Jacobi SVD kernel with dynamic loops and in-kernel sort.

    Each program instance handles one matrix from the batch.
    Uses column-major scratch buffers in global memory for O(M) column
    access. The Jacobi pair loops use runtime bounds (N_dim, num_sweeps)
    to prevent full unrolling, avoiding instruction cache thrashing.
    Singular values are sorted in descending order within the kernel.
    """
    pid = tle.program_id(0)
    row_idx = tl.arange(0, BLOCK_M)
    row_mask = row_idx < M

    # Base pointers for this batch element's scratch buffers
    aw_base = A_work_ptr + pid * aw_batch_stride
    vw_base = V_work_ptr + pid * vw_batch_stride

    # Copy input A into A_work (column-major layout)
    for j in range(N):
        a_col = tl.load(
            A_ptr + pid * batch_stride_A + row_idx * m_stride_A + j * n_stride_A,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(aw_base + j * aw_col_stride + row_idx, a_col, mask=row_mask)

    # Initialize V_work = Identity (column-major layout)
    if compute_uv:
        v_row_idx = tl.arange(0, BLOCK_N)
        v_mask = v_row_idx < N
        for j in range(N):
            v_col = tl.where(
                v_row_idx == j,
                tl.full((BLOCK_N,), 1.0, dtype=tl.float32),
                tl.full((BLOCK_N,), 0.0, dtype=tl.float32),
            )
            tl.store(vw_base + j * vw_col_stride + v_row_idx, v_col, mask=v_mask)

    # Jacobi sweeps — dynamic loop bounds prevent code bloat
    sweep_converged = tl.full((), 0, dtype=tl.int32)

    for _sweep in range(num_sweeps):
        max_gamma = tl.full((), 0.0, dtype=tl.float32)

        for p in range(N_dim):
            for q in range(p + 1, N_dim):
                # Load columns p and q from A_work
                a_p = tl.load(
                    aw_base + p * aw_col_stride + row_idx,
                    mask=row_mask,
                    other=0.0,
                )
                a_q = tl.load(
                    aw_base + q * aw_col_stride + row_idx,
                    mask=row_mask,
                    other=0.0,
                )

                # Compute Gram matrix entries for the 2x2 subproblem
                alpha = tl.sum(a_p * a_p)
                beta = tl.sum(a_q * a_q)
                gamma = tl.sum(a_p * a_q)

                abs_gamma = tl.abs(gamma)
                threshold = 1e-7 * tl.sqrt(alpha * beta + 1e-30)
                converged = abs_gamma < threshold

                # Track max off-diagonal for sweep-level convergence
                max_gamma = tl.where(abs_gamma > max_gamma, abs_gamma, max_gamma)

                # Only rotate if pair not converged and sweep not done
                should_rotate = ~converged & (sweep_converged == 0)

                # Compute Jacobi rotation angle
                safe_gamma = tl.where(
                    converged, tl.full((), 1.0, dtype=tl.float32), gamma
                )
                zeta = (beta - alpha) / (2.0 * safe_gamma)
                sign_zeta = tl.where(
                    zeta >= 0,
                    tl.full((), 1.0, dtype=tl.float32),
                    tl.full((), -1.0, dtype=tl.float32),
                )
                t = sign_zeta / (tl.abs(zeta) + tl.sqrt(1.0 + zeta * zeta))
                c = 1.0 / tl.sqrt(1.0 + t * t)
                s = t * c

                # Identity rotation when skipping
                c = tl.where(should_rotate, c, tl.full((), 1.0, dtype=tl.float32))
                s = tl.where(should_rotate, s, tl.full((), 0.0, dtype=tl.float32))

                # Apply rotation to A columns
                new_a_p = c * a_p - s * a_q
                new_a_q = s * a_p + c * a_q
                tl.store(aw_base + p * aw_col_stride + row_idx, new_a_p, mask=row_mask)
                tl.store(aw_base + q * aw_col_stride + row_idx, new_a_q, mask=row_mask)

                # Apply rotation to V columns
                if compute_uv:
                    v_idx = tl.arange(0, BLOCK_N)
                    v_m = v_idx < N
                    v_p = tl.load(
                        vw_base + p * vw_col_stride + v_idx, mask=v_m, other=0.0
                    )
                    v_q = tl.load(
                        vw_base + q * vw_col_stride + v_idx, mask=v_m, other=0.0
                    )
                    new_v_p = c * v_p - s * v_q
                    new_v_q = s * v_p + c * v_q
                    tl.store(vw_base + p * vw_col_stride + v_idx, new_v_p, mask=v_m)
                    tl.store(vw_base + q * vw_col_stride + v_idx, new_v_q, mask=v_m)

        # After each sweep, check if globally converged
        sweep_converged = tl.where(
            max_gamma < 1e-6,
            tl.full((), 1, dtype=tl.int32),
            sweep_converged,
        )

    # Extract singular values as column norms of A_work
    s_idx = tl.arange(0, BLOCK_N)
    s_mask = s_idx < K
    S_vals = tl.full((BLOCK_N,), 0.0, dtype=tl.float32)
    for j in range(N):
        a_col_j = tl.load(
            aw_base + j * aw_col_stride + row_idx, mask=row_mask, other=0.0
        )
        norm_sq = tl.sum(a_col_j * a_col_j)
        S_vals = tl.where(s_idx == j, tl.sqrt(norm_sq), S_vals)

    # --- In-kernel descending sort by computing ranks ---
    ranks = tl.zeros((BLOCK_N,), dtype=tl.int32)
    for i in range(N):
        s_i = tl.sum(tl.where(s_idx == i, S_vals, tl.zeros((BLOCK_N,), tl.float32)))
        i_val = tl.full((BLOCK_N,), i, dtype=tl.int32)
        beats = ((s_i > S_vals) | ((s_i == S_vals) & (i_val < s_idx))) & (s_idx < N)
        ranks = ranks + beats.to(tl.int32)

    # Output S in sorted descending order
    sorted_S = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for j in range(N):
        s_j = tl.sum(tl.where(s_idx == j, S_vals, tl.zeros((BLOCK_N,), tl.float32)))
        rank_j = tl.sum(tl.where(s_idx == j, ranks, tl.zeros((BLOCK_N,), tl.int32)))
        sorted_S = tl.where(s_idx == rank_j, s_j, sorted_S)
    tl.store(S_ptr + pid * batch_stride_S + s_idx, sorted_S, mask=s_mask)

    # Output U and V columns in sorted order
    if compute_uv:
        for j in range(N):
            rank_j = tl.sum(tl.where(s_idx == j, ranks, tl.zeros((BLOCK_N,), tl.int32)))

            a_col_j = tl.load(
                aw_base + j * aw_col_stride + row_idx, mask=row_mask, other=0.0
            )
            s_j = tl.sum(tl.where(s_idx == j, S_vals, tl.zeros((BLOCK_N,), tl.float32)))
            safe_s_j = tl.where(s_j > 1e-10, s_j, tl.full((), 1.0, dtype=tl.float32))
            u_col_j = a_col_j / safe_s_j

            u_ptrs = (
                U_ptr
                + pid * batch_stride_U
                + row_idx * m_stride_U
                + rank_j * k_stride_U
            )
            tl.store(u_ptrs, u_col_j, mask=row_mask & (rank_j < out_k_U))

            v_out_idx = tl.arange(0, BLOCK_N)
            v_col_j = tl.load(
                vw_base + j * vw_col_stride + v_out_idx,
                mask=v_out_idx < N,
                other=0.0,
            )
            v_ptrs = (
                V_ptr
                + pid * batch_stride_V
                + v_out_idx * n_stride_V
                + rank_j * k_stride_V
            )
            tl.store(v_ptrs, v_col_j, mask=(v_out_idx < N) & (rank_j < out_k_V))


# ---------------------------------------------------------------------------
# Bidiagonal SVD kernel: Householder bidiagonalization + Golub-Kahan QR
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def bidiag_svd_kernel(
    A_ptr,
    A_work_ptr,
    U_work_ptr,
    V_work_ptr,
    diag_ptr,
    superdiag_ptr,
    tau_left_ptr,
    tau_right_ptr,
    U_ptr,
    S_ptr,
    V_ptr,
    # Input A strides
    batch_stride_A,
    m_stride_A,
    n_stride_A,
    # A_work strides (column-major: [batch, col, row])
    aw_batch_stride,
    aw_col_stride,
    # U_work strides (column-major: [batch, col, row])
    uw_batch_stride,
    uw_col_stride,
    # V_work strides (column-major: [batch, col, row])
    vw_batch_stride,
    vw_col_stride,
    # Scratch buffer strides (diag/superdiag/tau: [batch, N])
    scratch_batch_stride,
    # Output strides
    batch_stride_U,
    m_stride_U,
    k_stride_U,
    batch_stride_S,
    batch_stride_V,
    n_stride_V,
    k_stride_V,
    # Runtime dims (prevents unrolling)
    N_dim,
    M_dim,
    max_qr_iters,
    # Constexpr dims
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    out_k_U: tl.constexpr,
    out_k_V: tl.constexpr,
    compute_uv: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Bidiagonal SVD: Householder bidiagonalization + Golub-Kahan implicit QR.

    One program per batch element. Operates on column-major scratch buffers.
    Phase 1: Reduce A to bidiagonal form via Householder reflections.
    Phase 2: Iterative QR with Wilkinson shift to find singular values.
    Phase 3: Back-transform to get U and V.
    Phase 4: Sort singular values descending and output.
    """
    pid = tle.program_id(0)
    row_idx = tl.arange(0, BLOCK_M)
    col_idx = tl.arange(0, BLOCK_N)
    row_mask = row_idx < M
    col_mask = col_idx < N

    aw_base = A_work_ptr + pid * aw_batch_stride
    uw_base = U_work_ptr + pid * uw_batch_stride
    vw_base = V_work_ptr + pid * vw_batch_stride
    d_base = diag_ptr + pid * scratch_batch_stride
    e_base = superdiag_ptr + pid * scratch_batch_stride
    tl_base = tau_left_ptr + pid * scratch_batch_stride
    tr_base = tau_right_ptr + pid * scratch_batch_stride

    # ---- Copy A into A_work (column-major) via 2D block ----
    a_ptr_2d = (
        A_ptr
        + pid * batch_stride_A
        + row_idx[:, None] * m_stride_A
        + col_idx[None, :] * n_stride_A
    )
    a_mask_2d = row_mask[:, None] & col_mask[None, :]
    A_block = tl.load(a_ptr_2d, mask=a_mask_2d, other=0.0).to(tl.float32)
    aw_ptr_2d = aw_base + col_idx[None, :] * aw_col_stride + row_idx[:, None]
    tl.store(aw_ptr_2d, A_block, mask=a_mask_2d)

    # ================================================================
    # Phase 1: Householder Bidiagonalization
    # ================================================================
    for k in range(N_dim):
        # --- LEFT Householder: zero out A_work[k+1:M, k] ---
        # Load column k from row k onward
        v_left = tl.load(
            aw_base + k * aw_col_stride + row_idx,
            mask=row_mask & (row_idx >= k),
            other=0.0,
        )

        # Compute norm of v_left[k:M]
        norm_sq = tl.sum(tl.where(row_idx >= k, v_left * v_left, 0.0))
        norm_val = tl.sqrt(norm_sq)

        # Extract a_kk (diagonal element)
        a_kk = tl.sum(tl.where(row_idx == k, v_left, 0.0))

        # sign choice: d[k] = -sign(a_kk) * norm
        sign_akk = tl.where(
            a_kk >= 0.0,
            tl.full((), 1.0, dtype=tl.float32),
            tl.full((), -1.0, dtype=tl.float32),
        )
        d_k = -sign_akk * norm_val
        tl.store(d_base + k, d_k)

        # Householder vector: v[k] = a_kk - d_k, v[k+1:] = a[k+1:]
        v_left = tl.where(row_idx == k, a_kk - d_k, v_left)
        # v_left is zero for row_idx < k

        # tau = 2 / (v^T v), handle zero vector
        vtv = tl.sum(tl.where(row_idx >= k, v_left * v_left, 0.0))
        tau_k = tl.where(vtv > 1e-30, 2.0 / vtv, 0.0)
        tl.store(tl_base + k, tau_k)

        # Store Householder vector in A_work column k (rows k..M-1 only).
        # IMPORTANT: only write rows >= k to preserve right Householder
        # vectors stored at A_work[row < k, col = k] by earlier iterations.
        tl.store(
            aw_base + k * aw_col_stride + row_idx,
            v_left,
            mask=row_mask & (row_idx >= k),
        )

        # Apply left reflector to columns k+1..N-1 via 2D block operation
        # A[:, k+1:N] -= tau * v * (v^T @ A[:, k+1:N])
        trail_ptr = (
            aw_base + (col_idx[None, :] + k + 1) * aw_col_stride + row_idx[:, None]
        )
        trail_mask = row_mask[:, None] & ((col_idx[None, :] + k + 1) < N)
        A_trail = tl.load(trail_ptr, mask=trail_mask, other=0.0)
        v_masked = tl.where(
            (row_idx >= k)[:, None],
            v_left[:, None],
            tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32),
        )
        dots = tl.sum(v_masked * A_trail, axis=0)
        A_trail = A_trail - tau_k * v_masked * dots[None, :]
        tl.store(trail_ptr, A_trail, mask=trail_mask)

        # --- RIGHT Householder: zero out A_work[k, k+2:N] ---
        if k < N - 2:
            # Load row k using strided vector load (replaces scalar gather)
            w_right = tl.load(
                aw_base + col_idx * aw_col_stride + k,
                mask=col_mask & (col_idx >= (k + 1)),
                other=0.0,
            )

            # Norm of row[k+1:N]
            r_norm_sq = tl.sum(tl.where(col_idx >= (k + 1), w_right * w_right, 0.0))
            r_norm = tl.sqrt(r_norm_sq)

            # a_{k,k+1}
            a_k_kp1 = tl.sum(tl.where(col_idx == (k + 1), w_right, 0.0))
            sign_r = tl.where(
                a_k_kp1 >= 0.0,
                tl.full((), 1.0, dtype=tl.float32),
                tl.full((), -1.0, dtype=tl.float32),
            )
            e_k = -sign_r * r_norm
            tl.store(e_base + k, e_k)

            # Householder vector for right reflection
            w_right = tl.where(col_idx == (k + 1), a_k_kp1 - e_k, w_right)

            wtw = tl.sum(tl.where(col_idx >= (k + 1), w_right * w_right, 0.0))
            tau_r_k = tl.where(wtw > 1e-30, 2.0 / wtw, 0.0)
            tl.store(tr_base + k, tau_r_k)

            # Store w using strided vector store (replaces scalar scatter)
            tl.store(
                aw_base + col_idx * aw_col_stride + k,
                w_right,
                mask=col_mask & (col_idx >= (k + 1)),
            )

            # Apply right reflector via 2D block: A -= tau * (A @ w) * w^T
            # Load A_work[0:M, 0:N] as 2D block, mask rows > k and cols >= k+1
            right_ptr = aw_base + col_idx[None, :] * aw_col_stride + row_idx[:, None]
            right_mask = (
                (row_idx[:, None] > k)
                & row_mask[:, None]
                & (col_idx[None, :] >= (k + 1))
                & col_mask[None, :]
            )
            A_right = tl.load(right_ptr, mask=right_mask, other=0.0)
            # p = A @ w (per-row dot product with w_right)
            p = tl.sum(A_right * w_right[None, :], axis=1)
            # A -= tau * p * w^T
            A_right = A_right - tau_r_k * p[:, None] * w_right[None, :]
            tl.store(right_ptr, A_right, mask=right_mask)

        elif k == N - 2:
            # Last superdiagonal element: just read it
            e_k = tl.load(aw_base + (k + 1) * aw_col_stride + k)
            tl.store(e_base + k, e_k)

    # ================================================================
    # Phase 2: Golub-Kahan Implicit QR on bidiagonal (d, e)
    # ================================================================
    # Initialize U2 = I(N x N) and V2 = I(N x N) via 2D identity
    n_idx = tl.arange(0, BLOCK_N)
    n_mask = n_idx < N
    eye_2d = tl.where(
        n_idx[:, None] == n_idx[None, :],
        tl.full((BLOCK_N, BLOCK_N), 1.0, dtype=tl.float64),
        tl.full((BLOCK_N, BLOCK_N), 0.0, dtype=tl.float64),
    )
    eye_mask = n_mask[:, None] & n_mask[None, :]
    uw_ptr_2d = uw_base + n_idx[None, :] * uw_col_stride + n_idx[:, None]
    vw_ptr_2d = vw_base + n_idx[None, :] * vw_col_stride + n_idx[:, None]
    tl.store(uw_ptr_2d, eye_2d, mask=eye_mask)
    tl.store(vw_ptr_2d, eye_2d, mask=eye_mask)

    # NOTE: No pre-QR sign flipping. The Golub-Kahan QR algorithm handles
    # signed bidiagonal entries correctly. Flipping d and e signs independently
    # is incorrect because row/column sign flips interact through the bidiagonal
    # structure. Signs are fixed post-convergence instead.

    # QR iterations — use machine-epsilon-scale convergence for float64 QR
    # The bidiagonal QR runs in float64, so eps ≈ 1e-7 gives near-optimal
    # singular vectors while minimizing iteration count.
    eps = 1e-7
    all_converged = tl.full((), 0, dtype=tl.int32)

    for _qr_iter in range(max_qr_iters):
        if all_converged == 0:
            # Vectorized deflation: load e[] and d[] as vectors
            e_vec = tl.load(e_base + n_idx, mask=n_idx < (N - 1), other=0.0)
            d_vec = tl.load(d_base + n_idx, mask=n_mask, other=0.0)
            d_next = tl.load(d_base + n_idx + 1, mask=n_idx < (N - 1), other=0.0)

            # Deflate: set e[k]=0 where |e[k]| < eps*(|d[k]|+|d[k+1]|)
            thresh = eps * (tl.abs(d_vec) + tl.abs(d_next))
            defl = (tl.abs(e_vec) < thresh) & (n_idx < (N - 1))
            e_vec = tl.where(
                defl,
                tl.zeros((BLOCK_N,), dtype=tl.float64),
                e_vec,
            )
            tl.store(e_base + n_idx, e_vec, mask=n_idx < (N - 1))

            # Find hi: largest (k+1) where e[k] != 0
            active = (e_vec != 0.0) & (n_idx < (N - 1))
            hi_vals = tl.where(
                active, (n_idx + 1).to(tl.int32), tl.zeros((BLOCK_N,), tl.int32)
            )
            hi = tl.max(hi_vals, axis=0)

            if hi == 0:
                all_converged = tl.full((), 1, dtype=tl.int32)
            else:
                # Find lo: largest (k+1) where e[k]==0 and k < hi
                zero_below = (e_vec == 0.0) & (n_idx < hi) & (n_idx < (N - 1))
                lo_vals = tl.where(
                    zero_below,
                    (n_idx + 1).to(tl.int32),
                    tl.zeros((BLOCK_N,), tl.int32),
                )
                lo = tl.max(lo_vals, axis=0)

                if lo < hi:
                    # Wilkinson shift from bottom-right 2x2 of B^T B
                    d_hi = tl.load(d_base + hi)
                    d_hi_m1 = tl.load(d_base + hi - 1)
                    e_hi_m1 = tl.load(e_base + hi - 1)

                    a22 = d_hi * d_hi + e_hi_m1 * e_hi_m1
                    a11 = d_hi_m1 * d_hi_m1
                    if hi - 2 >= lo:
                        e_hi_m2 = tl.load(e_base + hi - 2)
                        a11 = a11 + e_hi_m2 * e_hi_m2
                    a12 = d_hi_m1 * e_hi_m1

                    avg = 0.5 * (a11 + a22)
                    diff_val = 0.5 * (a11 - a22)
                    disc = tl.sqrt(diff_val * diff_val + a12 * a12)
                    lam1 = avg + disc
                    lam2 = avg - disc
                    shift = tl.where(
                        tl.abs(lam1 - a22) < tl.abs(lam2 - a22), lam1, lam2
                    )

                    # Initial bulge
                    d_lo = tl.load(d_base + lo)
                    e_lo = tl.load(e_base + lo)
                    y = d_lo * d_lo - shift
                    z = d_lo * e_lo

                    # Pre-load first V2/U2 columns and scalar carries.
                    # Scalar carries for d[k] and e[k] avoid re-reading
                    # values just written in the previous step from global
                    # memory, saving ~2 scalar loads per bulge chase step.
                    if compute_uv:
                        v_carry = tl.load(
                            vw_base + lo * vw_col_stride + n_idx,
                            mask=n_mask,
                            other=0.0,
                        )
                        u_carry = tl.load(
                            uw_base + lo * uw_col_stride + n_idx,
                            mask=n_mask,
                            other=0.0,
                        )

                    d_carry = tl.load(d_base + lo)
                    e_carry = tl.load(e_base + lo)

                    # Bulge chase from lo to hi-1
                    for k in range(lo, hi):
                        # Right Givens
                        r = tl.sqrt(y * y + z * z)
                        safe_r = tl.where(r > 1e-30, r, 1.0)
                        c = y / safe_r
                        s = -z / safe_r

                        d_k = d_carry
                        d_kp1 = tl.load(d_base + k + 1)
                        e_k = e_carry

                        if k > lo:
                            tl.store(e_base + k - 1, r)

                        new_dk = c * d_k - s * e_k
                        tmp1 = s * d_k + c * e_k
                        tmp2 = -s * d_kp1
                        new_dkp1 = c * d_kp1

                        # Accumulate V2 (carry: avoid re-reading just-written col)
                        if compute_uv:
                            vk = v_carry
                            vkp1 = tl.load(
                                vw_base + (k + 1) * vw_col_stride + n_idx,
                                mask=n_mask,
                                other=0.0,
                            )
                            new_vk = c * vk - s * vkp1
                            v_carry = s * vk + c * vkp1
                            tl.store(
                                vw_base + k * vw_col_stride + n_idx,
                                new_vk,
                                mask=n_mask,
                            )

                        # Left Givens
                        r2 = tl.sqrt(new_dk * new_dk + tmp2 * tmp2)
                        safe_r2 = tl.where(r2 > 1e-30, r2, 1.0)
                        c2 = new_dk / safe_r2
                        s2 = -tmp2 / safe_r2

                        new_ek2 = c2 * tmp1 - s2 * new_dkp1
                        new_dkp1_2 = s2 * tmp1 + c2 * new_dkp1

                        # Store d[k], carry d[k+1] and e[k] forward
                        tl.store(d_base + k, r2)
                        d_carry = new_dkp1_2
                        e_carry = new_ek2

                        # Accumulate U2 (carry: avoid re-reading just-written col)
                        if compute_uv:
                            uk = u_carry
                            ukp1 = tl.load(
                                uw_base + (k + 1) * uw_col_stride + n_idx,
                                mask=n_mask,
                                other=0.0,
                            )
                            new_uk = c2 * uk - s2 * ukp1
                            u_carry = s2 * uk + c2 * ukp1
                            tl.store(
                                uw_base + k * uw_col_stride + n_idx,
                                new_uk,
                                mask=n_mask,
                            )

                        # Next bulge
                        if k < hi - 1:
                            e_kp1 = tl.load(e_base + k + 1)
                            y = new_ek2
                            z = -s2 * e_kp1
                            e_carry = c2 * e_kp1

                    # Flush carried scalars after bulge chase
                    tl.store(d_base + hi, d_carry)
                    tl.store(e_base + hi - 1, e_carry)

                    # Store last carried columns
                    if compute_uv:
                        tl.store(
                            vw_base + hi * vw_col_stride + n_idx,
                            v_carry,
                            mask=n_mask,
                        )
                        tl.store(
                            uw_base + hi * uw_col_stride + n_idx,
                            u_carry,
                            mask=n_mask,
                        )

    # Ensure all singular values are non-negative
    for k in range(N_dim):
        d_k = tl.load(d_base + k)
        if d_k < 0.0:
            tl.store(d_base + k, -d_k)
            if compute_uv:
                u_col_k = tl.load(
                    uw_base + k * uw_col_stride + n_idx, mask=n_mask, other=0.0
                )
                tl.store(uw_base + k * uw_col_stride + n_idx, -u_col_k, mask=n_mask)

    # ================================================================
    # Phase 3: Back-transformation
    # ================================================================
    if compute_uv:
        # U = Q_L * [U2; 0] where Q_L = H_0 * H_1 * ... * H_{N-1}
        # V = Q_R * V2      where Q_R = G_0 * G_1 * ... * G_{N-3}
        # Apply reflectors in reverse order to U2/V2.

        # Copy U2 (N x N) expanded to M x N into output U via 2D block
        u_base_pid = U_ptr + pid * batch_stride_U
        u_ptr_2d = (
            u_base_pid + row_idx[:, None] * m_stride_U + col_idx[None, :] * k_stride_U
        )
        u_mask_2d = row_mask[:, None] & col_mask[None, :]

        uw_read_2d = uw_base + col_idx[None, :] * uw_col_stride + row_idx[:, None]
        uw_read_mask = (row_idx[:, None] < N) & row_mask[:, None] & col_mask[None, :]
        U_init = tl.load(uw_read_2d, mask=uw_read_mask, other=0.0).to(tl.float32)
        tl.store(u_ptr_2d, U_init, mask=u_mask_2d)

        for k_rev in range(N_dim):
            k = N_dim - 1 - k_rev
            v_k = tl.load(
                aw_base + k * aw_col_stride + row_idx,
                mask=row_mask & (row_idx >= k),
                other=0.0,
            )
            tau_k = tl.load(tl_base + k)

            U_block = tl.load(u_ptr_2d, mask=u_mask_2d, other=0.0)
            v_masked = tl.where(
                (row_idx >= k)[:, None],
                v_k[:, None],
                tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32),
            )
            dots = tl.sum(v_masked * U_block, axis=0)
            U_block = U_block - tau_k * v_masked * dots[None, :]
            tl.store(u_ptr_2d, U_block, mask=u_mask_2d)

        # Copy V2 into output V via 2D block
        v_base_pid = V_ptr + pid * batch_stride_V
        v_ptr_2d = (
            v_base_pid + col_idx[:, None] * n_stride_V + col_idx[None, :] * k_stride_V
        )
        v_mask_2d = col_mask[:, None] & col_mask[None, :]

        vw_read_2d = vw_base + col_idx[None, :] * vw_col_stride + col_idx[:, None]
        V_init = tl.load(vw_read_2d, mask=v_mask_2d, other=0.0).to(tl.float32)
        tl.store(v_ptr_2d, V_init, mask=v_mask_2d)

        for k_rev in range(N_dim):
            k = N_dim - 3 - k_rev
            if k >= 0:
                # Strided vector load for right Householder vector
                w_k = tl.load(
                    aw_base + col_idx * aw_col_stride + k,
                    mask=col_mask & (col_idx >= (k + 1)),
                    other=0.0,
                )
                tau_r_k = tl.load(tr_base + k)

                V_block = tl.load(v_ptr_2d, mask=v_mask_2d, other=0.0)
                # dots = w^T @ V (per-column dot products) → BLOCK_N vector
                w_masked = tl.where(
                    (col_idx >= (k + 1))[:, None],
                    w_k[:, None],
                    tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32),
                )
                dots = tl.sum(w_masked * V_block, axis=0)
                V_block = V_block - tau_r_k * w_masked * dots[None, :]
                tl.store(v_ptr_2d, V_block, mask=v_mask_2d)

    # ================================================================
    # Phase 4: Sort singular values descending and output
    # ================================================================
    s_idx = tl.arange(0, BLOCK_N)
    s_mask = s_idx < K

    # Load all singular values as a vector (replaces N-iteration scalar loop)
    S_vals = tl.load(d_base + s_idx, mask=s_mask, other=0.0)

    # Compute ranks via 2D comparison matrix (replaces N-iteration loop)
    # ranks[j] = number of elements i where S[i] > S[j] or tied with i < j
    valid_2d = (s_idx[:, None] < N) & (s_idx[None, :] < N)
    beats_2d = (
        (S_vals[:, None] > S_vals[None, :])
        | ((S_vals[:, None] == S_vals[None, :]) & (s_idx[:, None] < s_idx[None, :]))
    ) & valid_2d
    ranks = tl.sum(beats_2d.to(tl.int32), axis=0)

    # Output S sorted via scatter store (replaces N-iteration loop)
    tl.store(
        S_ptr + pid * batch_stride_S + ranks,
        S_vals.to(tl.float32),
        mask=s_mask,
    )

    # Permute U and V columns via gather (replaces 7 N-iteration loops)
    if compute_uv:
        # Compute inverse permutation: inv_ranks[i] = source col for output pos i
        valid_j = s_idx[None, :] < N
        perm_match = (ranks[None, :] == s_idx[:, None]) & valid_j
        inv_ranks = tl.sum(
            tl.where(
                perm_match,
                s_idx[None, :],
                tl.zeros((BLOCK_N, BLOCK_N), tl.int32),
            ),
            axis=1,
        )

        # Gather-permute U columns (load from source, store to dest)
        u_gather_ptr = (
            U_ptr
            + pid * batch_stride_U
            + row_idx[:, None] * m_stride_U
            + inv_ranks[None, :] * k_stride_U
        )
        U_sorted = tl.load(u_gather_ptr, mask=u_mask_2d, other=0.0)
        tl.store(u_ptr_2d, U_sorted, mask=u_mask_2d)

        # Gather-permute V columns
        v_gather_ptr = (
            V_ptr
            + pid * batch_stride_V
            + col_idx[:, None] * n_stride_V
            + inv_ranks[None, :] * k_stride_V
        )
        V_sorted = tl.load(v_gather_ptr, mask=v_mask_2d, other=0.0)
        tl.store(v_ptr_2d, V_sorted, mask=v_mask_2d)


def _svd_fallback(input, some, compute_uv):
    """Fallback to cuSOLVER for matrices exceeding kernel thresholds.

    Calls aten::_linalg_svd directly to minimize dispatch overhead.
    cuSOLVER's divide-and-conquer SVD is asymptotically faster for larger
    matrices and uses multiple SMs for a single matrix decomposition.
    """
    if compute_uv:
        U, S, Vh = torch.ops.aten._linalg_svd.default(input, not some, True)
        V = Vh.mH
    else:
        _, S, _ = torch.ops.aten._linalg_svd.default(input, not some, False)
        m, n = input.shape[-2], input.shape[-1]
        k = min(m, n)
        batch_shape = input.shape[:-2]
        if some:
            U = torch.zeros(*batch_shape, m, k, device=input.device, dtype=input.dtype)
            V = torch.zeros(*batch_shape, n, k, device=input.device, dtype=input.dtype)
        else:
            U = torch.zeros(*batch_shape, m, m, device=input.device, dtype=input.dtype)
            V = torch.zeros(*batch_shape, n, n, device=input.device, dtype=input.dtype)
    return SVDResult(U, S, V)


def svd(input, some=True, compute_uv=True):
    """Compute the Singular Value Decomposition of a matrix.

    Implements SVD using native Triton kernels:
    - One-sided Jacobi for small matrices (min(m,n) <= 16)
    - Householder bidiagonalization + Golub-Kahan QR for larger matrices
      (16 < min(m,n) <= 1024)
    - cuSOLVER fallback only for very large matrices (min(m,n) > 1024)

    Args:
        input: Input tensor of shape (..., m, n).
        some: If True, return reduced SVD. If False, return full SVD.
        compute_uv: If True, compute U and V. If False, only compute S
            and return zero-filled U, V.

    Returns:
        Named tuple (U, S, V) where:
            - U: Left singular vectors
            - S: Singular values in descending order
            - V: Right singular vectors (not transposed)
    """
    logger.debug("GEMS SVD")
    assert input.ndim >= 2, f"Input must have at least 2 dimensions, got {input.ndim}"

    orig_m, orig_n = input.shape[-2], input.shape[-1]
    k = min(orig_m, orig_n)

    # Fall back to cuSOLVER for matrices beyond our kernel's maximum dimension.
    if k > MAX_SVD_DIM:
        return _svd_fallback(input, some, compute_uv)

    batch_shape = input.shape[:-2]
    m, n = orig_m, orig_n
    batch_size = 1
    for s in batch_shape:
        batch_size *= s

    # Handle empty batch
    if batch_size == 0:
        if some:
            U_out = torch.zeros(
                *batch_shape, orig_m, k, device=input.device, dtype=input.dtype
            )
            V_out = torch.zeros(
                *batch_shape, orig_n, k, device=input.device, dtype=input.dtype
            )
        else:
            U_out = torch.zeros(
                *batch_shape, orig_m, orig_m, device=input.device, dtype=input.dtype
            )
            V_out = torch.zeros(
                *batch_shape, orig_n, orig_n, device=input.device, dtype=input.dtype
            )
        S_out = torch.empty(*batch_shape, k, device=input.device, dtype=input.dtype)
        return SVDResult(U_out, S_out, V_out)

    A = input.reshape(batch_size, m, n)
    if not A.is_contiguous():
        A = A.contiguous()

    # Handle m < n by transposing: SVD(A^T) gives V, S, U
    need_transpose = m < n
    if need_transpose:
        A = A.transpose(-2, -1).contiguous()
        m, n = n, m

    # After transpose normalization, m >= n, so k = n
    if some:
        out_k_U = k
        out_k_V = k
    else:
        out_k_U = m
        out_k_V = n

    S_out = torch.empty(batch_size, k, device=input.device, dtype=input.dtype)
    if some and compute_uv:
        U_out = torch.empty(
            batch_size, m, out_k_U, device=input.device, dtype=input.dtype
        )
        V_out = torch.empty(
            batch_size, n, out_k_V, device=input.device, dtype=input.dtype
        )
    else:
        U_out = torch.zeros(
            batch_size, m, out_k_U, device=input.device, dtype=input.dtype
        )
        V_out = torch.zeros(
            batch_size, n, out_k_V, device=input.device, dtype=input.dtype
        )

    BLOCK_M = _next_power_of_2(m)
    BLOCK_N = _next_power_of_2(n)
    grid = (batch_size,)

    if k <= _JACOBI_THRESHOLD:
        # --- Jacobi SVD path (small matrices) ---
        A_work = torch.empty(batch_size, n, m, device=input.device, dtype=torch.float32)
        V_work = torch.empty(batch_size, n, n, device=input.device, dtype=torch.float32)
        num_sweeps = 6
        num_warps_val = 1

        with torch_device_fn.device(input.device):
            jacobi_svd_kernel[grid](
                A,
                A_work,
                V_work,
                U_out,
                S_out,
                V_out,
                A.stride(0),
                A.stride(1),
                A.stride(2),
                A_work.stride(0),
                A_work.stride(1),
                V_work.stride(0),
                V_work.stride(1),
                U_out.stride(0),
                U_out.stride(1),
                U_out.stride(2),
                S_out.stride(0),
                V_out.stride(0),
                V_out.stride(1),
                V_out.stride(2),
                n,
                num_sweeps,
                M=m,
                N=n,
                K=k,
                out_k_U=out_k_U,
                out_k_V=out_k_V,
                compute_uv=compute_uv,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                num_warps=num_warps_val,
                num_stages=1,
            )
    else:
        # --- Bidiagonal SVD path (16 < k <= 1024) ---
        A_work = torch.empty(batch_size, n, m, device=input.device, dtype=torch.float32)
        U_work = torch.empty(batch_size, n, n, device=input.device, dtype=torch.float64)
        V_work = torch.empty(batch_size, n, n, device=input.device, dtype=torch.float64)
        diag = torch.zeros(batch_size, n, device=input.device, dtype=torch.float64)
        superdiag = torch.zeros(batch_size, n, device=input.device, dtype=torch.float64)
        tau_left = torch.zeros(batch_size, n, device=input.device, dtype=torch.float32)
        tau_right = torch.zeros(batch_size, n, device=input.device, dtype=torch.float32)

        max_qr_iters = 3 * n
        # Use a single warp to avoid cross-warp synchronization issues in
        # the sequential Householder / QR loops that access shared state
        # (A_work, diag, superdiag) within the same iteration.
        num_warps_val = 1

        with torch_device_fn.device(input.device):
            bidiag_svd_kernel[grid](
                A,
                A_work,
                U_work,
                V_work,
                diag,
                superdiag,
                tau_left,
                tau_right,
                U_out,
                S_out,
                V_out,
                # Input A strides
                A.stride(0),
                A.stride(1),
                A.stride(2),
                # A_work strides
                A_work.stride(0),
                A_work.stride(1),
                # U_work strides
                U_work.stride(0),
                U_work.stride(1),
                # V_work strides
                V_work.stride(0),
                V_work.stride(1),
                # Scratch strides
                diag.stride(0),
                # Output strides
                U_out.stride(0),
                U_out.stride(1),
                U_out.stride(2),
                S_out.stride(0),
                V_out.stride(0),
                V_out.stride(1),
                V_out.stride(2),
                # Runtime dims
                n,
                m,
                max_qr_iters,
                # Constexpr dims
                M=m,
                N=n,
                K=k,
                out_k_U=out_k_U,
                out_k_V=out_k_V,
                compute_uv=compute_uv,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                num_warps=num_warps_val,
                num_stages=1,
            )

    # Undo transpose: swap U and V back
    if need_transpose:
        U_out, V_out = V_out, U_out

    S_out = S_out.reshape(*batch_shape, k)

    if compute_uv:
        if need_transpose:
            U_out = U_out.reshape(*batch_shape, orig_m, out_k_V)
            V_out = V_out.reshape(*batch_shape, orig_n, out_k_U)
        else:
            U_out = U_out.reshape(*batch_shape, orig_m, out_k_U)
            V_out = V_out.reshape(*batch_shape, orig_n, out_k_V)
    else:
        if some:
            U_out = torch.zeros(
                *batch_shape, orig_m, k, device=input.device, dtype=input.dtype
            )
            V_out = torch.zeros(
                *batch_shape, orig_n, k, device=input.device, dtype=input.dtype
            )
        else:
            U_out = torch.zeros(
                *batch_shape, orig_m, orig_m, device=input.device, dtype=input.dtype
            )
            V_out = torch.zeros(
                *batch_shape, orig_n, orig_n, device=input.device, dtype=input.dtype
            )

    return SVDResult(U_out, S_out, V_out)
