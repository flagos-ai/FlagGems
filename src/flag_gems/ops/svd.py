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
MAX_SVD_DIM = 64

# Routing thresholds:
# k <= _JACOBI_THRESHOLD: Jacobi SVD (O(N^4), but fast for tiny matrices)
# _JACOBI_THRESHOLD < k <= MAX_SVD_DIM: Bidiagonal SVD (O(N^3))
# k > MAX_SVD_DIM: cuSOLVER fallback
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

    # ---- Copy A into A_work (column-major) ----
    for j in range(N):
        a_col = tl.load(
            A_ptr + pid * batch_stride_A + row_idx * m_stride_A + j * n_stride_A,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(aw_base + j * aw_col_stride + row_idx, a_col, mask=row_mask)

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

        # Apply left reflector to remaining columns j = k+1..N-1
        # A[:,j] -= tau * (v^T A[:,j]) * v
        for j in range(k + 1, N_dim):
            a_col_j = tl.load(
                aw_base + j * aw_col_stride + row_idx,
                mask=row_mask,
                other=0.0,
            )
            dot = tl.sum(tl.where(row_idx >= k, v_left * a_col_j, 0.0))
            a_col_j = tl.where(row_idx >= k, a_col_j - tau_k * dot * v_left, a_col_j)
            tl.store(aw_base + j * aw_col_stride + row_idx, a_col_j, mask=row_mask)

        # --- RIGHT Householder: zero out A_work[k, k+2:N] ---
        if k < N - 2:
            # Load row k from columns k+1..N-1 (strided access)
            w_right = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for jj in range(k + 1, N_dim):
                val = tl.load(aw_base + jj * aw_col_stride + k)
                w_right = tl.where(col_idx == jj, val, w_right)

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

            # Store w in A_work row k, cols k+1..N-1
            for jj in range(k + 1, N_dim):
                w_jj = tl.sum(
                    tl.where(col_idx == jj, w_right, tl.zeros((BLOCK_N,), tl.float32))
                )
                tl.store(aw_base + jj * aw_col_stride + k, w_jj)

            # Apply right reflector column-wise: A -= tau * (A * w) * w^T
            # Step 1: p = A[k+1:M, k+1:N] * w (length-M vector)
            p = tl.zeros((BLOCK_M,), dtype=tl.float32)
            for jj in range(k + 1, N_dim):
                w_jj = tl.load(aw_base + jj * aw_col_stride + k)
                a_col = tl.load(
                    aw_base + jj * aw_col_stride + row_idx,
                    mask=row_mask,
                    other=0.0,
                )
                p = tl.where(row_idx > k, p + w_jj * a_col, p)

            # Step 2: A[:, j] -= tau * w[j] * p
            for jj in range(k + 1, N_dim):
                w_jj = tl.load(aw_base + jj * aw_col_stride + k)
                a_col = tl.load(
                    aw_base + jj * aw_col_stride + row_idx,
                    mask=row_mask,
                    other=0.0,
                )
                a_col = tl.where(
                    row_idx > k,
                    a_col - tau_r_k * w_jj * p,
                    a_col,
                )
                tl.store(
                    aw_base + jj * aw_col_stride + row_idx,
                    a_col,
                    mask=row_mask,
                )

        elif k == N - 2:
            # Last superdiagonal element: just read it
            e_k = tl.load(aw_base + (k + 1) * aw_col_stride + k)
            tl.store(e_base + k, e_k)

    # ================================================================
    # Phase 2: Golub-Kahan Implicit QR on bidiagonal (d, e)
    # ================================================================
    # Initialize U2 = I(N x N) and V2 = I(N x N)
    n_idx = tl.arange(0, BLOCK_N)
    n_mask = n_idx < N
    for j in range(N_dim):
        u_col = tl.where(
            n_idx == j,
            tl.full((BLOCK_N,), 1.0, dtype=tl.float64),
            tl.full((BLOCK_N,), 0.0, dtype=tl.float64),
        )
        tl.store(uw_base + j * uw_col_stride + n_idx, u_col, mask=n_mask)
        v_col = tl.where(
            n_idx == j,
            tl.full((BLOCK_N,), 1.0, dtype=tl.float64),
            tl.full((BLOCK_N,), 0.0, dtype=tl.float64),
        )
        tl.store(vw_base + j * vw_col_stride + n_idx, v_col, mask=n_mask)

    # NOTE: No pre-QR sign flipping. The Golub-Kahan QR algorithm handles
    # signed bidiagonal entries correctly. Flipping d and e signs independently
    # is incorrect because row/column sign flips interact through the bidiagonal
    # structure. Signs are fixed post-convergence instead.

    # QR iterations
    eps = 1e-10
    all_converged = tl.full((), 0, dtype=tl.int32)

    for _qr_iter in range(max_qr_iters):
        if all_converged == 0:
            # Deflate negligible superdiagonal elements
            for k in range(N_dim - 1):
                e_k = tl.load(e_base + k)
                d_k = tl.load(d_base + k)
                d_kp1 = tl.load(d_base + k + 1)
                thresh = eps * (tl.abs(d_k) + tl.abs(d_kp1))
                if tl.abs(e_k) < thresh:
                    tl.store(e_base + k, 0.0)

            # Find hi: largest index where e[hi-1] != 0
            hi = 0
            for k in range(N_dim - 1):
                e_k = tl.load(e_base + k)
                if e_k != 0.0:
                    hi = k + 1

            if hi == 0:
                all_converged = tl.full((), 1, dtype=tl.int32)
            else:
                # Find lo: walk backwards from hi-1 to find first zero e
                lo = 0
                for k in range(N_dim - 1):
                    # Forward scan: find last zero e below hi
                    if k < hi:
                        e_k = tl.load(e_base + k)
                        if e_k == 0.0:
                            lo = k + 1

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

                    # Pre-load first V2/U2 columns for register carry.
                    # This avoids cross-warp read-after-write hazards:
                    # instead of re-reading column k+1 from global memory
                    # (which another warp may not have finished writing),
                    # we carry the rotated value in registers.
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

                    # Bulge chase from lo to hi-1
                    for k in range(lo, hi):
                        # Right Givens
                        r = tl.sqrt(y * y + z * z)
                        safe_r = tl.where(r > 1e-30, r, 1.0)
                        c = y / safe_r
                        s = -z / safe_r

                        d_k = tl.load(d_base + k)
                        d_kp1 = tl.load(d_base + k + 1)
                        e_k = tl.load(e_base + k)

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

                        tl.store(d_base + k, r2)
                        new_ek2 = c2 * tmp1 - s2 * new_dkp1
                        new_dkp1_2 = s2 * tmp1 + c2 * new_dkp1
                        tl.store(d_base + k + 1, new_dkp1_2)
                        tl.store(e_base + k, new_ek2)

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
                            tl.store(e_base + k + 1, c2 * e_kp1)

                    # Store last carried columns after bulge chase
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

        # Copy U2 (N x N) expanded to M x N into output U
        for j in range(N_dim):
            u_col_out = tl.where(
                row_idx < N,
                tl.load(
                    uw_base + j * uw_col_stride + row_idx,
                    mask=row_mask & (row_idx < N),
                    other=0.0,
                ),
                tl.zeros((BLOCK_M,), dtype=tl.float64),
            ).to(tl.float32)
            tl.store(
                U_ptr + pid * batch_stride_U + row_idx * m_stride_U + j * k_stride_U,
                u_col_out,
                mask=row_mask,
            )

        # Apply H_{N-1}, H_{N-2}, ..., H_0 to U columns
        for k_rev in range(N_dim):
            k = N_dim - 1 - k_rev
            v_k = tl.load(
                aw_base + k * aw_col_stride + row_idx,
                mask=row_mask & (row_idx >= k),
                other=0.0,
            )
            tau_k = tl.load(tl_base + k)

            for j in range(N_dim):
                u_col = tl.load(
                    U_ptr
                    + pid * batch_stride_U
                    + row_idx * m_stride_U
                    + j * k_stride_U,
                    mask=row_mask,
                    other=0.0,
                )
                dot = tl.sum(tl.where(row_idx >= k, v_k * u_col, 0.0))
                u_col = tl.where(row_idx >= k, u_col - tau_k * dot * v_k, u_col)
                tl.store(
                    U_ptr
                    + pid * batch_stride_U
                    + row_idx * m_stride_U
                    + j * k_stride_U,
                    u_col,
                    mask=row_mask,
                )

        # Copy V2 into output V
        for j in range(N_dim):
            v_col_out = tl.load(
                vw_base + j * vw_col_stride + col_idx,
                mask=col_mask,
                other=0.0,
            ).to(tl.float32)
            tl.store(
                V_ptr + pid * batch_stride_V + col_idx * n_stride_V + j * k_stride_V,
                v_col_out,
                mask=col_mask,
            )

        # Apply right reflectors: k = N-3 down to 0
        for k_rev in range(N_dim):
            k = N_dim - 3 - k_rev
            if k >= 0:
                w_k = tl.zeros((BLOCK_N,), dtype=tl.float32)
                for jj in range(k + 1, N_dim):
                    val = tl.load(aw_base + jj * aw_col_stride + k)
                    w_k = tl.where(col_idx == jj, val, w_k)

                tau_r_k = tl.load(tr_base + k)

                for j in range(N_dim):
                    v_col = tl.load(
                        V_ptr
                        + pid * batch_stride_V
                        + col_idx * n_stride_V
                        + j * k_stride_V,
                        mask=col_mask,
                        other=0.0,
                    )
                    dot = tl.sum(tl.where(col_idx >= (k + 1), w_k * v_col, 0.0))
                    v_col = tl.where(
                        col_idx >= (k + 1),
                        v_col - tau_r_k * dot * w_k,
                        v_col,
                    )
                    tl.store(
                        V_ptr
                        + pid * batch_stride_V
                        + col_idx * n_stride_V
                        + j * k_stride_V,
                        v_col,
                        mask=col_mask,
                    )

    # ================================================================
    # Phase 4: Sort singular values descending and output
    # ================================================================
    s_idx = tl.arange(0, BLOCK_N)
    s_mask = s_idx < K

    # Load final singular values from diag (float64)
    S_vals = tl.zeros((BLOCK_N,), dtype=tl.float64)
    for j in range(N_dim):
        d_j = tl.load(d_base + j)
        S_vals = tl.where(s_idx == j, d_j, S_vals)

    # Compute ranks for descending sort
    ranks = tl.zeros((BLOCK_N,), dtype=tl.int32)
    for i in range(N_dim):
        s_i = tl.sum(tl.where(s_idx == i, S_vals, tl.zeros((BLOCK_N,), tl.float64)))
        i_val = tl.full((BLOCK_N,), i, dtype=tl.int32)
        beats = ((s_i > S_vals) | ((s_i == S_vals) & (i_val < s_idx))) & (s_idx < N)
        ranks = ranks + beats.to(tl.int32)

    # Output S sorted (convert to float32 for output)
    sorted_S = tl.zeros((BLOCK_N,), dtype=tl.float64)
    for j in range(N_dim):
        s_j = tl.sum(tl.where(s_idx == j, S_vals, tl.zeros((BLOCK_N,), tl.float64)))
        rank_j = tl.sum(tl.where(s_idx == j, ranks, tl.zeros((BLOCK_N,), tl.int32)))
        sorted_S = tl.where(s_idx == rank_j, s_j, sorted_S)
    tl.store(S_ptr + pid * batch_stride_S + s_idx, sorted_S.to(tl.float32), mask=s_mask)

    # Permute U and V columns by rank
    if compute_uv:
        for j in range(N_dim):
            rank_j = tl.sum(tl.where(s_idx == j, ranks, tl.zeros((BLOCK_N,), tl.int32)))

            # Read U column j, write to rank_j position
            u_col = tl.load(
                U_ptr + pid * batch_stride_U + row_idx * m_stride_U + j * k_stride_U,
                mask=row_mask,
                other=0.0,
            )
            # We need to permute in-place which is tricky. Instead, store
            # to A_work as temp buffer then copy back.
            tl.store(aw_base + rank_j * aw_col_stride + row_idx, u_col, mask=row_mask)

        # Copy permuted U back
        for j in range(N_dim):
            if j < out_k_U:
                u_col = tl.load(
                    aw_base + j * aw_col_stride + row_idx,
                    mask=row_mask,
                    other=0.0,
                )
                tl.store(
                    U_ptr
                    + pid * batch_stride_U
                    + row_idx * m_stride_U
                    + j * k_stride_U,
                    u_col,
                    mask=row_mask,
                )

        # Permute V columns
        for j in range(N_dim):
            rank_j = tl.sum(tl.where(s_idx == j, ranks, tl.zeros((BLOCK_N,), tl.int32)))
            v_col = tl.load(
                V_ptr + pid * batch_stride_V + col_idx * n_stride_V + j * k_stride_V,
                mask=col_mask,
                other=0.0,
            )
            # Store to U_work as temp
            tl.store(uw_base + rank_j * uw_col_stride + col_idx, v_col, mask=col_mask)

        for j in range(N_dim):
            if j < out_k_V:
                v_col = tl.load(
                    uw_base + j * uw_col_stride + col_idx,
                    mask=col_mask,
                    other=0.0,
                ).to(tl.float32)
                tl.store(
                    V_ptr
                    + pid * batch_stride_V
                    + col_idx * n_stride_V
                    + j * k_stride_V,
                    v_col,
                    mask=col_mask,
                )


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

    Implements SVD using:
    - One-sided Jacobi for small matrices (min(m,n) <= 16)
    - Householder bidiagonalization + Golub-Kahan QR for medium matrices
      (16 < min(m,n) <= 64)
    - torch.linalg.svd fallback for larger matrices (min(m,n) > 64)

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
        # --- Bidiagonal SVD path (medium matrices, 16 < k <= 64) ---
        A_work = torch.empty(batch_size, n, m, device=input.device, dtype=torch.float32)
        U_work = torch.empty(batch_size, n, n, device=input.device, dtype=torch.float64)
        V_work = torch.empty(batch_size, n, n, device=input.device, dtype=torch.float64)
        diag = torch.zeros(batch_size, n, device=input.device, dtype=torch.float64)
        superdiag = torch.zeros(batch_size, n, device=input.device, dtype=torch.float64)
        tau_left = torch.zeros(batch_size, n, device=input.device, dtype=torch.float32)
        tau_right = torch.zeros(batch_size, n, device=input.device, dtype=torch.float32)

        max_qr_iters = 30 * n
        # The QR bulge chase uses register-carried columns (v_carry, u_carry)
        # to avoid cross-warp read-after-write hazards. This allows using
        # multiple warps safely. Use enough warps to cover the block size.
        num_warps_val = max(1, max(BLOCK_M, BLOCK_N) // 32)

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
