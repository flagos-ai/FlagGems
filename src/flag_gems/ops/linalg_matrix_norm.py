"""linalg_matrix_norm -- matrix norm with Triton kernels and C++ wrapper.

    ord=2/-2    → max/min singular value (fp64 QR + Jacobi or bidiag+DBDSQR)
    ord=1/-1    → max/min absolute column sum
    ord=inf/-inf → max/min absolute row sum
    ord="fro"   → Frobenius norm = sqrt(Σ Aᵢⱼ²)
    ord="nuc"   → nuclear norm = Σ σₖ (sum of singular values)

7 Triton kernels (shared with C++ wrapper via triton_jit):
    _fro_kernel                       _abs_norm_kernel
    _householder_qr_r_kernel          _parallel_jacobi_step_kernel
    _rank2_svals_kernel               _bidiag_kernel
    _fused_dbdsqr_kernel
"""

import logging
import math

import torch
import triton
import triton.language as tl

# FlagGems computational operators (replace torch.amax / torch.sqrt).
from flag_gems.ops.amax import amax
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

# ===========================================================================
# Kernel: _fro_kernel -- unified Frobenius norm (sqrt(Σx²)).
# Grid=(batch,) for TILE_2D=False (per-row), Grid=(grid_m×grid_n,) for TILE_2D=True (tiled).
# ===========================================================================


@libentry()
@triton.jit
def _fro_kernel(
    X,
    Out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GRID_N: tl.constexpr,
    TILE_2D: tl.constexpr,
):
    """Unified Frobenius norm kernel: sqrt(Σ x²).

    Two modes selected by TILE_2D:
      TILE_2D=False (1D per-row):
        Grid=(batch,).  Each program handles one row / one matrix,
        loops over N elements in BLOCK_N chunks, stores sqrt(Σx²)
        to Out[pid].  Used for batched inputs and small 2D matrices.

      TILE_2D=True (2D tiled):
        Grid=(grid_m × grid_n,).  Each program processes one
        BLOCK_M×BLOCK_N tile, atomically adds Σx² to Out[0].
        Host takes sqrt.  Used for large single matrices (M·N > 65536).
    """
    c_dtype = X.dtype.element_ty
    if c_dtype != tl.float64:
        c_dtype = tl.float32
    # Always accumulate sum-of-squares in fp64: fp32 summation error grows
    # as N * eps ≈ 4e-3 (before sqrt) for N=65536 in the 1D path, and
    # fp32 atomic_add across tiles gives ~2e-6 in the 2D path.  fp64
    # accumulation eliminates both noise floors so the Frobenius norm
    # meets CPU-LAPACK comparison tolerances regardless of path.
    acc_dtype = tl.float64
    if TILE_2D:
        # --- 2D tiled mode: one matrix, many tile-blocks ---
        pid = tl.program_id(0)
        pid_m = pid // GRID_N
        pid_n = pid % GRID_N
        row_start = pid_m * BLOCK_M
        col_start = pid_n * BLOCK_N
        rows = row_start + tl.arange(0, BLOCK_M)[:, None]
        cols = col_start + tl.arange(0, BLOCK_N)[None, :]
        mask = (rows < M) & (cols < N)
        x = tl.load(X + rows * N + cols, mask=mask, other=0.0).to(acc_dtype)
        tile_sum = tl.sum(x * x)
        tl.atomic_add(Out, tile_sum)
    else:
        # --- 1D per-row mode: batch rows, each program = one row ---
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK_N)
        acc = tl.zeros([BLOCK_N], dtype=acc_dtype)
        for start in range(0, N, BLOCK_N):
            idx = start + offs
            mask = idx < N
            x = tl.load(X + pid * N + idx, mask=mask, other=0.0).to(acc_dtype)
            acc += tl.where(mask, x * x, 0.0)
        total = tl.sqrt(tl.sum(acc))
        tl.store(Out + pid, total.to(Out.dtype.element_ty))


# ===========================================================================
# Kernel: _rank2_svals_kernel -- closed-form SVD for k=2.
# Used by _svdvals_for_norm and C++ SVD dispatch.  No iteration needed.
# BLOCK_B=1 → one matrix per program (regular).
# BLOCK_B>1 → BLOCK_B matrices per program (vectorized, for tiny rows).
# ===========================================================================

_RANK2_BLOCK_R_MAX = 2048


def _select_dbdsqr_params(k):
    """Autotune DBDSQR parameters by k: larger k needs more iterations.
    Returns (MAX_ITERS, num_warps)."""
    if k <= 32:
        return 30, 1
    elif k <= 64:
        return 50, 1
    elif k <= 128:
        return 100, 4
    else:
        return 200, 4


def _svd_shape(A):
    """Return (batch, M, N) for an SVD-shaped tensor."""
    if A.ndim < 2:
        return 0, 0, 0
    batch = 1
    for d in A.shape[:-2]:
        batch *= d
    return batch, A.shape[-2], A.shape[-1]


@libentry()
@triton.jit
def _rank2_svals_kernel(
    A,
    S,
    BATCH: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    TALL: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Closed-form SVD for k=2. BLOCK_B matrices per program."""
    pid = tl.program_id(0)
    eps = 1.0e-20
    c_dtype = A.type.element_ty
    if c_dtype != tl.float64:
        c_dtype = tl.float32

    if BLOCK_B == 1:
        # --- Single matrix per program (regular path) ---
        offs = tl.arange(0, BLOCK_R)
        if TALL:
            mask = offs < M
            base = A + pid * M * N
            x = tl.load(base + offs * N, mask=mask, other=0.0).to(c_dtype)
            y = tl.load(base + offs * N + 1, mask=mask, other=0.0).to(c_dtype)
        else:
            mask = offs < N
            base = A + pid * M * N
            x = tl.load(base + offs, mask=mask, other=0.0).to(c_dtype)
            y = tl.load(base + N + offs, mask=mask, other=0.0).to(c_dtype)

        aa = tl.sum(x * x)
        bbv = tl.sum(y * y)
        ab = tl.sum(x * y)
        diff = aa - bbv
        root = tl.sqrt(diff * diff + 4.0 * ab * ab)
        l0 = tl.maximum(0.0, 0.5 * (aa + bbv + root))
        det = tl.maximum(0.0, aa * bbv - ab * ab)
        l1 = tl.where(l0 > eps, det / l0, 0.0)

        sbase = S + pid * 2
        tl.store(sbase, tl.sqrt(l0))
        tl.store(sbase + 1, tl.sqrt(l1))

    else:
        # --- BLOCK_B matrices per program (vectorized, tiny rows) ---
        b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        r = tl.arange(0, BLOCK_R)
        bb = b[:, None]
        rr = r[None, :]
        bmask = b < BATCH

        if TALL:
            mask = (bb < BATCH) & (rr < M)
            base = A + bb * M * N + rr * N
            x = tl.load(base, mask=mask, other=0.0).to(c_dtype)
            y = tl.load(base + 1, mask=mask, other=0.0).to(c_dtype)
        else:
            mask = (bb < BATCH) & (rr < N)
            base = A + bb * M * N + rr
            x = tl.load(base, mask=mask, other=0.0).to(c_dtype)
            y = tl.load(base + N, mask=mask, other=0.0).to(c_dtype)

        aa = tl.sum(x * x, axis=1)
        bbv = tl.sum(y * y, axis=1)
        ab = tl.sum(x * y, axis=1)
        diff = aa - bbv
        root = tl.sqrt(diff * diff + 4.0 * ab * ab)
        l0 = tl.maximum(0.0, 0.5 * (aa + bbv + root))
        det = tl.maximum(0.0, aa * bbv - ab * ab)
        l1 = tl.where(l0 > 1.0e-20, det / l0, 0.0)
        tl.store(S + b * 2, tl.sqrt(l0), mask=bmask)
        tl.store(S + b * 2 + 1, tl.sqrt(l1), mask=bmask)


# ===========================================================================
# Householder QR kernel -- pure Triton left-Householder QR (R-only).
# Replaces torch.linalg.qr for A of shape (M, N) with M >= N, K = min(M,N).
# Outputs upper-triangular R (K×K).  Does NOT compute Q.
# Grid=(batch,).  Each program handles one matrix independently.
# ===========================================================================


@libentry()
@triton.jit
def _householder_qr_r_kernel(
    A_ptr,
    R_out,
    M,
    N,
    K,
    stride_b,
    stride_m,
    stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_FP64: tl.constexpr = False,
):
    """Left-Householder QR, R-only.  Grid=(batch,).

    For A (M×N, M≥N), computes QR → stores upper-triangular R (K×K, K=N)
    into R_out (row-major, shape (batch, K, K)).  A is overwritten with
    Householder vectors (caller ignores them).

    Uses BLOCK_M for row-tiling and BLOCK_N for column-tiling during the
    trailing-submatrix update.  BLOCK_N=32 works well on most GPUs.
    """
    pid = tl.program_id(0)
    eps = 1.0e-30
    DTYPE = tl.float64 if USE_FP64 else tl.float32

    a_base = A_ptr + pid * stride_b

    for j in range(K):
        # ================================================================
        # Phase 1: Householder reflector for column j
        #   x = A[j:M, j]
        #   α = -sign(x₀)·‖x‖
        #   v = x - α·e₁  (unnormalized),   v[0] = x₀ - α
        #   τ = 2 / ‖v‖²
        #
        # LAPACK convention: when M-j == 1 (column has only the diagonal
        # element), the subdiagonal is empty → xLARFG returns TAU=0 and
        # leaves the diagonal unchanged.  R[j,j] = A[j,j] as-is.
        # ================================================================
        has_subdiag = M - j > 1
        if has_subdiag:
            x0 = tl.load(a_base + j * stride_m + j).to(DTYPE)

            # Tiled reduction: ‖x‖² = Σ_{i=j}^{M-1} A[i, j]²
            x_norm_sq = tl.zeros([BLOCK_M], dtype=DTYPE)
            for r_start in range(j, M, BLOCK_M):
                r_offs = r_start + tl.arange(0, BLOCK_M)
                r_mask = r_offs < M
                x_vals = tl.load(
                    a_base + r_offs * stride_m + j,
                    mask=r_mask,
                    other=0.0,
                ).to(DTYPE)
                x_norm_sq += tl.where(r_mask, x_vals * x_vals, 0.0)
            x_norm = tl.sqrt(tl.sum(x_norm_sq))

            # α = -sign(x₀) · ‖x‖
            sign_x0 = tl.where(x0 >= 0.0, 1.0, -1.0)
            alpha = -sign_x0 * x_norm

            v0 = x0 - alpha
            # Stable ‖v‖²: 2‖x‖² - 2x₀α = 2‖x‖(‖x‖ + |x₀|)
            v_norm_sq = 2.0 * x_norm * (x_norm + tl.abs(x0))
            tau = tl.where(v_norm_sq > eps, 2.0 / v_norm_sq, 0.0)

            # Store α back to A[j,j] -- this is R[j,j] after the reflector.
            tl.store(a_base + j * stride_m + j, alpha)

            # ============================================================
            # Phase 2: Apply reflector to trailing submatrix
            #   A[j:M, j+1:N] -= τ·v·(vᵀ·A[j:M, j+1:N])
            #
            #   v = [v0, A[j+1,j], …, A[M-1,j]]  (UNnormalized, v[0]=v₀)
            #   τ = 2 / ‖v‖²  already computed above.
            #
            #   Two-pass per column-tile:
            #     Pass 1 -- compute w[c] = Σ_m v[m]·A[m, c]
            #     Pass 2 -- A[m, c] -= τ·v[m]·w[c]
            # ============================================================
            if tau > 0.0:
                for c_start in range(j + 1, N, BLOCK_N):
                    c_offs = c_start + tl.arange(0, BLOCK_N)
                    c_mask = c_offs < N

                    # ---- Pass 1: w = vᵀ @ A[j:M, c_tile] ----
                    w = tl.zeros([BLOCK_N], dtype=DTYPE)
                    for r_start in range(j, M, BLOCK_M):
                        r_offs = r_start + tl.arange(0, BLOCK_M)
                        r_mask = r_offs < M

                        # v[r] -- unnormalized: v[j]=v₀, v[r>j]=A[r,j] (unchanged)
                        v_r = tl.where(
                            r_mask & (r_offs == j),
                            v0,
                            tl.load(
                                a_base + r_offs * stride_m + j,
                                mask=r_mask & (r_offs > j),
                                other=0.0,
                            ).to(DTYPE),
                        )

                        # A[r_tile, c_tile]
                        a_tile = tl.load(
                            a_base + r_offs[:, None] * stride_m + c_offs[None, :],
                            mask=r_mask[:, None] & c_mask[None, :],
                            other=0.0,
                        ).to(DTYPE)

                        # w[c] += v[r] · A[r, c]
                        w += tl.sum(v_r[:, None] * a_tile, axis=0)

                    # ---- Pass 2: A -= τ·v·w ----
                    for r_start in range(j, M, BLOCK_M):
                        r_offs = r_start + tl.arange(0, BLOCK_M)
                        r_mask = r_offs < M

                        v_r = tl.where(
                            r_mask & (r_offs == j),
                            v0,
                            tl.load(
                                a_base + r_offs * stride_m + j,
                                mask=r_mask & (r_offs > j),
                                other=0.0,
                            ).to(DTYPE),
                        )

                        a_tile = tl.load(
                            a_base + r_offs[:, None] * stride_m + c_offs[None, :],
                            mask=r_mask[:, None] & c_mask[None, :],
                            other=0.0,
                        ).to(DTYPE)

                        a_tile -= tau * v_r[:, None] * w[None, :]

                        tl.store(
                            a_base + r_offs[:, None] * stride_m + c_offs[None, :],
                            a_tile,
                            mask=r_mask[:, None] & c_mask[None, :],
                        )

    # ====================================================================
    # Phase 3: Extract upper triangle of A into R_out.
    # R[i, j] = A[i, j] for 0 ≤ i ≤ j < K.
    # ====================================================================
    for i in range(K):
        for c_start in range(i, K, BLOCK_N):
            c_offs = c_start + tl.arange(0, BLOCK_N)
            c_mask = c_offs < K
            vals = tl.load(
                a_base + i * stride_m + c_offs,
                mask=c_mask,
                other=0.0,
            )
            tl.store(
                R_out + pid * K * K + i * K + c_offs,
                vals,
                mask=c_mask,
            )


# ===========================================================================
# gesvd kernels -- fp64 Householder bidiagonalization + fused DBDSQR.
# cuSOLVER-aligned: bidiag in fp64 matches GEBRD, DBDSQR tol 2.2e-14 converges.
# ===========================================================================


@libentry()
@triton.jit
def _bidiag_kernel(
    R,
    D,
    E,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Bidiagonalize upper triangular R (k×k) via LAPACK GEBRD algorithm.

    All computation in fp64 (matching cuSOLVER GEBRD).  The input R is
    expected to be fp64 (caller converts from fp32 QR output).  Outputs
    D and E are fp64, compatible with DBDSQR tolerance (2.2e-14).

    Grid=(batch,).  R is modified in-place."""
    pid = tl.program_id(0)
    eps = 1.0e-30
    idx = tl.arange(0, BLOCK_K)
    dtype = R.dtype.element_ty

    i = 0
    while i < K - 1:
        # === Left reflector: zero R[i+1:, i] (subdiagonal of column i) ===
        col_mask = (idx + i) < K
        x = tl.load(R + pid * K * K + (i + idx) * K + i, mask=col_mask, other=0.0).to(
            dtype
        )
        x0 = tl.sum(tl.where(idx == 0, x, 0.0))
        x_sq = tl.where(col_mask, x * x, 0.0)
        x_nrm = tl.sqrt(tl.sum(x_sq))

        sign_x0 = tl.where(x0 >= 0.0, 1.0, -1.0)
        alpha_l = -sign_x0 * x_nrm
        u0 = x0 - alpha_l
        u = tl.where(col_mask, tl.where(idx == 0, u0, x), 0.0)
        beta_l = tl.sum(tl.where(col_mask, u * u, 0.0))
        inv_nrm_l = tl.rsqrt(tl.maximum(beta_l, eps))
        v_l = tl.where(col_mask, u * inv_nrm_l, 0.0)

        # Apply left reflector H_L = I - 2*v_l*v_l^T
        r = i
        while r < K:
            row_mask = (idx + i) < K
            row_r = tl.load(
                R + pid * K * K + (i + idx) * K + r, mask=row_mask, other=0.0
            ).to(dtype)
            dot_r = tl.sum(tl.where(row_mask, v_l * row_r, 0.0))
            new_row = tl.where(row_mask, row_r - 2.0 * dot_r * v_l, row_r)
            tl.store(R + pid * K * K + (i + idx) * K + r, new_row, mask=row_mask)
            r += 1

        # Store d[i] = R[i,i] after left reflector
        d_val = tl.load(R + pid * K * K + i * K + i).to(dtype)
        tl.store(D + pid * K + i, d_val)

        # === Right reflector: zero R[i, i+2:] (far super-diagonal of row i) ===
        if i + 1 < K:
            w = K - i - 1
            row_mask_r = idx < w
            y = tl.load(
                R + pid * K * K + i * K + (i + 1 + idx), mask=row_mask_r, other=0.0
            ).to(dtype)

            y_sq = tl.where(row_mask_r, y * y, 0.0)
            y_nrm = tl.sqrt(tl.sum(y_sq))
            y0 = tl.sum(tl.where(idx == 0, y, 0.0))
            sign_y0 = tl.where(y0 >= 0.0, 1.0, -1.0)
            e_i = -sign_y0 * y_nrm

            u0_r = y0 - e_i
            u_r = tl.where(row_mask_r, tl.where(idx == 0, u0_r, y), 0.0)
            beta_r = tl.sum(tl.where(row_mask_r, u_r * u_r, 0.0))
            inv_nrm_r = tl.rsqrt(tl.maximum(beta_r, eps))
            v_r = tl.where(row_mask_r, u_r * inv_nrm_r, 0.0)

            # Store super-diagonal e[i]
            tl.store(E + pid * (K - 1) + i, e_i)

            # Apply right reflector H_R = I - 2*v_r*v_r^T
            r = i
            while r < K:
                row_r = tl.load(
                    R + pid * K * K + r * K + (i + 1 + idx), mask=row_mask_r, other=0.0
                ).to(dtype)
                dot_r = tl.sum(tl.where(row_mask_r, row_r * v_r, 0.0))
                new_row = tl.where(row_mask_r, row_r - 2.0 * dot_r * v_r, row_r)
                tl.store(
                    R + pid * K * K + r * K + (i + 1 + idx), new_row, mask=row_mask_r
                )
                r += 1

        i += 1

    # Store last diagonal
    d_last = tl.load(R + pid * K * K + (K - 1) * K + (K - 1)).to(dtype)
    tl.store(D + pid * K + (K - 1), d_last)


logger = logging.getLogger(__name__)

_SUPPORTED_NUMERIC = {1, -1, 2, -2, float("inf"), -float("inf")}


# ===========================================================================
# Unified abs-norm kernel -- parameterized by SUM_AXIS, IS_MIN, TILED, BATCHED.
#
# SUM_AXIS=0  → 1-norm style  (reduce along rows, per-column result)
# SUM_AXIS=1  → inf-norm style (reduce along cols, per-row result)
# IS_MIN=False → max (for positive ord), IS_MIN=True → min (for negative ord)
# TILED=True   → 2D grid, atomic_add to partial buffer (large single matrix)
# TILED=False  → 1D stripe, atomic_max/atomic_min directly to Out
# BATCHED=True → 2D grid=(batch, grid_dim), output to Out[batch_idx]
#                (only used with TILED=False; TILED+BATCHED is not combined)
#
# Replaces: _fused_abs_tiled_kernel + _fused_abs_single_kernel
#           + _reduce_kernel + _batched_abs_multi_kernel
# ===========================================================================


@libentry()
@triton.jit
def _abs_norm_kernel(
    X,
    Out,
    Partial,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GRID_N: tl.constexpr,
    SUM_AXIS: tl.constexpr,
    IS_MIN: tl.constexpr,
    TILED: tl.constexpr,
    BATCHED: tl.constexpr = False,
):
    # Accumulate in the input dtype: fp64→fp64, others→fp32 (matching
    # _fro_kernel convention, preserves fp64 precision through the reduction).
    c_dtype = X.dtype.element_ty
    if c_dtype != tl.float64:
        c_dtype = tl.float32

    if TILED:
        # --- 2D tiled: atomic_add to Partial (single matrix, not batched) ---
        pid = tl.program_id(0)
        pid_m = pid // GRID_N
        pid_n = pid % GRID_N
        row_start = pid_m * BLOCK_M
        col_start = pid_n * BLOCK_N
        rows = row_start + tl.arange(0, BLOCK_M)[:, None]
        cols = col_start + tl.arange(0, BLOCK_N)[None, :]
        row_mask = rows < M
        col_mask = cols < N
        x = tl.load(X + rows * N + cols, mask=row_mask & col_mask, other=0.0).to(
            c_dtype
        )

        if SUM_AXIS == 0:
            col_sum = tl.sum(tl.abs(x), axis=0)
            off = col_start + tl.arange(0, BLOCK_N)
            tl.atomic_add(Partial + off, col_sum)
        else:
            row_sum = tl.sum(tl.abs(x), axis=1)
            off = row_start + tl.arange(0, BLOCK_M)
            tl.atomic_add(Partial + off, row_sum)

    else:
        # --- 1D stripe: atomic_max / atomic_min to Out[batch_idx] ---
        if BATCHED:
            batch_idx = tl.program_id(0)
            block_idx = tl.program_id(1)
            base = batch_idx * M * N
            out_ptr = Out + batch_idx
        else:
            batch_idx = 0
            block_idx = tl.program_id(0)
            base = 0
            out_ptr = Out

        if SUM_AXIS == 0:
            # 1norm: per-column stripe
            block_start = block_idx * BLOCK_N
            cols = block_start + tl.arange(0, BLOCK_N)
            col_mask = cols < N
            acc = tl.zeros([BLOCK_N], dtype=c_dtype)
            for row_start in range(0, M, BLOCK_M):
                rows = row_start + tl.arange(0, BLOCK_M)[:, None]
                mask = (rows < M) & col_mask[None, :]
                x = tl.load(
                    X + base + rows * N + cols[None, :], mask=mask, other=0.0
                ).to(c_dtype)
                acc += tl.sum(tl.abs(x), axis=0)
            # atomic_max / atomic_min require fp32 on most hardware;
            # convert to fp32 for the final reduction.
            acc_f32 = acc.to(tl.float32)
            if IS_MIN:
                acc_f32 = tl.where(col_mask, acc_f32, float("inf"))
                tl.atomic_min(out_ptr, tl.min(acc_f32))
            else:
                tl.atomic_max(out_ptr, tl.max(acc_f32))

        else:
            # infnorm: per-row stripe
            row_start = block_idx * BLOCK_M
            rows = row_start + tl.arange(0, BLOCK_M)[:, None]
            row_mask = rows < M
            acc = tl.zeros([BLOCK_M, 1], dtype=c_dtype)
            for col_start in range(0, N, BLOCK_N):
                cols = col_start + tl.arange(0, BLOCK_N)[None, :]
                mask = row_mask & (cols < N)
                x = tl.load(X + base + rows * N + cols, mask=mask, other=0.0).to(
                    c_dtype
                )
                acc += tl.sum(tl.abs(x), axis=1)[:, None]
            acc_f32 = acc.to(tl.float32)
            if IS_MIN:
                acc_f32 = tl.where(row_mask, acc_f32, float("inf"))
                tl.atomic_min(out_ptr, tl.min(acc_f32))
            else:
                tl.atomic_max(out_ptr, tl.max(acc_f32))


# --- 2D tiled Frobenius -- single-launch with atomic_add to scalar ----------
# ===========================================================================
# Host dispatch -- mirrors the C++ dispatch in lib/linalg_matrix_norm.cpp.
#
# C++ and Python use the same function names (Python prefixed with _):
#
#   C++                       Python
#   ────────────────────────  ───────────────────────
#   fro_norm                  _fro_norm
#   nuc_norm                  _nuc_norm
#   ord1_norm                 _ord1_norm
#   ordinf_norm               _ordinf_norm
#   svdvals_hybrid            _svdvals_hybrid
#   svdvals_rank2             _svdvals_rank2
#   (inline dispatch)         _ord2_norm
#   (inline dispatch)         _svdvals_for_norm
# ===========================================================================


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _amin(x, dim, keepdim=False):
    """Drop-in for ``torch.amin`` -- works for all float dtypes.

    FlagGems ``amax`` requires *dim* as a list/tuple, not a plain int.
    """
    if isinstance(dim, int):
        dim = [dim]
    return -amax(-x, dim=dim, keepdim=keepdim)


# ---------------------------------------------------------------------------
# Per-ord helper functions -- each mirrors a path in the C++ dispatch
# ---------------------------------------------------------------------------


def _fro_norm(A, dim, keepdim, dtype):
    """Frobenius norm (ord=\"fro\").  Mirrors C++ ``fro_norm``.

    Uses the unified ``_fro_kernel``: TILE_2D=False for small/batched
    (per-row L2, grid=(batch,)), TILE_2D=True for large 2D (tiled + atomic,
    host sqrt).
    """
    d0, d1 = dim
    out_dtype = dtype if dtype is not None else A.dtype

    # Simple 2D case -- same kernel path as C++ fro_norm.
    if A.ndim == 2 and d0 == 0 and d1 == 1:
        M, N = A.shape
        total = M * N
        if total <= 65536:
            flat = A.reshape(1, total)
            tmp = torch.empty(1, dtype=out_dtype, device=A.device)
            _fro_kernel[(1,)](
                flat, tmp, 0, total, 1, 512, 1, TILE_2D=False, num_warps=8
            )
            result = tmp.view(())
        else:
            if M <= 1024 and N <= 1024:
                BM, BN = 32, 32
            elif N >= 8 * M or M >= 8 * N:
                BM, BN = 128, 128
            else:
                BM, BN = 128, 32
            grid_m = triton.cdiv(M, BM)
            grid_n = triton.cdiv(N, BN)
            grid_size = int(grid_m * grid_n)
            # Accumulate in fp64 for the tiled atomic_add path so that
            # both the per-tile sum and the cross-tile reduction use fp64.
            # The _fro_kernel internally uses fp64 when TILE_2D=True.
            acc_dtype = torch.float64
            out = torch.zeros((), dtype=acc_dtype, device=A.device)
            _fro_kernel[(grid_size,)](
                A, out, M, N, BM, BN, grid_n, TILE_2D=True, num_warps=8
            )
            result = torch.sqrt(out).to(out_dtype)
        if result.dtype != out_dtype:
            result = result.to(out_dtype)
        if keepdim:
            result = result.reshape(1, 1)
        return result

    # Batched case -- same kernel as C++ wrapper (mirrors linalg_matrix_norm_str).
    ndim = A.ndim
    all_dims = list(range(ndim))
    remaining = [d for d in all_dims if d not in (d0, d1)]
    perm = remaining + [d0, d1]
    A_perm = A.permute(perm)
    batch = 1
    for i in range(A_perm.ndim - 2):
        batch *= A_perm.size(i)
    mat_size = A_perm.size(-2) * A_perm.size(-1)
    # A_perm is non-contiguous; reshape on a non-contiguous (…,1) view can
    # yield a strided (batch, mat_size) view instead of a copy, breaking the
    # _fro_kernel's contiguous linear indexing.  Force a contiguous copy.
    flat = A_perm.reshape(batch, mat_size).contiguous()

    result = torch.empty(batch, dtype=out_dtype, device=flat.device)
    blk_n = triton.next_power_of_2(min(mat_size, 512))
    _fro_kernel[(batch,)](
        flat, result, 0, mat_size, 1, blk_n, 1, TILE_2D=False, num_warps=8
    )

    if result.dtype != out_dtype:
        result = result.to(out_dtype)
    if keepdim:
        out_shape = list(A.shape)
        out_shape[d0] = 1
        out_shape[d1] = 1
        result = result.reshape(out_shape)
    else:
        batch_shape = [A.size(i) for i in range(ndim) if i != d0 and i != d1]
        result = result.reshape(batch_shape)
    return result


def _ord2_norm(A, ord_val, dim, keepdim, dtype):
    """Spectral norm (ord=2 / ord=-2).  Mirrors C++ inline dispatch.

    Permutes target dims to last two positions, computes singular values
    via ``_svdvals_for_norm``, then takes max (ord=2) or min (ord=-2).
    """
    d0, d1 = dim
    out_dtype = dtype if dtype is not None else A.dtype

    # Move target dims to the last two positions.
    ndim = A.ndim
    all_dims = list(range(ndim))
    remaining = [d for d in all_dims if d not in (d0, d1)]
    perm = remaining + [d0, d1]
    A_perm = A.permute(perm) if perm != all_dims else A
    if dtype is not None:
        A_perm = A_perm.to(dtype)

    s = _svdvals_for_norm(A_perm)
    fn = amax if ord_val > 0 else _amin
    result = fn(s, dim=[-1], keepdim=False)

    if result.dtype != out_dtype:
        result = result.to(out_dtype)
    if keepdim:
        out_shape = list(A.shape)
        out_shape[d0] = 1
        out_shape[d1] = 1
        result = result.reshape(out_shape)
    return result


def _choose_fast_tile(M, N):
    """Shared tile size selection. Returns (BM, BN, grid_m, grid_n)."""
    if M <= 1024 and N <= 1024:
        BM, BN = 32, 32
    elif N >= 8 * M or M >= 8 * N:
        BM, BN = min(M, 128), min(N, 128)
    else:
        BM, BN = 128, 32
    if BM > M:
        BM = triton.next_power_of_2(M)  # tl.arange requires pow2
    if BN > N:
        BN = triton.next_power_of_2(N)
    grid_m = triton.cdiv(M, BM)
    grid_n = triton.cdiv(N, BN)
    return BM, BN, int(grid_m), int(grid_n)


def _batched_kernel_dispatch(A, dim, ord_val, out_dtype, keepdim):
    """Mirrors the C++ batched path in ``linalg_matrix_norm`` exactly.

    Permutes so the two matrix dims are last, reshapes to (batch, M, N),
    then dispatches to the same Triton kernels used by the C++ wrapper.
    """
    d0, d1 = dim
    ndim = A.ndim
    all_dims = list(range(ndim))
    remaining = [d for d in all_dims if d != d0 and d != d1]
    perm = remaining + [d0, d1]
    A_perm = A.permute(perm)
    batch = 1
    for i in range(A_perm.ndim - 2):
        batch *= A_perm.size(i)
    mat_M = A_perm.size(-2)
    mat_N = A_perm.size(-1)
    Ab = A_perm.reshape(batch, mat_M, mat_N).contiguous()

    is_min = ord_val < 0
    abs_ord = abs(float(ord_val))

    if math.isinf(abs_ord):
        # --- inf/-inf: multi-block per matrix (row-parallel) ---
        tile_m = 16
        grid_dim = triton.cdiv(mat_M, tile_m)
        blk_dim = triton.next_power_of_2(min(mat_N, 256))
        init_val = float("inf") if is_min else 0.0
        result = torch.full((batch,), init_val, dtype=torch.float32, device=Ab.device)
        _dummy = torch.empty(1, device=Ab.device)
        _abs_norm_kernel[(batch, grid_dim)](
            Ab,
            result,
            _dummy,
            mat_M,
            mat_N,
            tile_m,
            blk_dim,
            1,
            SUM_AXIS=1,
            IS_MIN=is_min,
            TILED=False,
            BATCHED=True,
            num_warps=8,
        )
    elif abs_ord == 1.0:
        # --- 1/-1: multi-block per matrix (column-parallel) ---
        tile_n_raw = min(mat_N, 128)
        tile_n = triton.next_power_of_2(tile_n_raw)
        grid_dim = triton.cdiv(mat_N, tile_n_raw)
        blk_dim = triton.next_power_of_2(min(mat_M, 32))
        init_val = float("inf") if is_min else 0.0
        result = torch.full((batch,), init_val, dtype=torch.float32, device=Ab.device)
        _dummy = torch.empty(1, device=Ab.device)
        _abs_norm_kernel[(batch, grid_dim)](
            Ab,
            result,
            _dummy,
            mat_M,
            mat_N,
            blk_dim,
            tile_n,
            1,
            SUM_AXIS=0,
            IS_MIN=is_min,
            TILED=False,
            BATCHED=True,
            num_warps=8,
        )
    else:
        raise RuntimeError(f"_batched_kernel_dispatch: unsupported ord {ord_val}")

    if result.dtype != out_dtype:
        result = result.to(out_dtype)

    if keepdim:
        out_shape = list(A.shape)
        out_shape[d0] = 1
        out_shape[d1] = 1
        result = result.reshape(out_shape)
    else:
        batch_shape = [A.size(i) for i in range(ndim) if i != d0 and i != d1]
        result = result.reshape(batch_shape)
    return result


def _ord1_norm(A, ord_val, dim, keepdim, dtype):
    """1-norm (ord=1 / ord=-1).  Mirrors C++ ``ord1_norm``.

    Uses ``_abs_norm_kernel``: TILED=True (2D grid → partial buffer → host
    max/min) for large matrices, TILED=False (1D stripe → atomic scalar)
    for small matrices.  Batched via ``_batched_kernel_dispatch`` →
    ``_abs_norm_kernel`` BATCHED=True.
    """
    d0, d1 = dim
    out_dtype = dtype if dtype is not None else A.dtype
    is_min = ord_val < 0

    # --- Simple 2D matrix path  ---
    if A.ndim == 2 and d0 == 0 and d1 == 1:
        M, N = A.shape

        BM, BN, grid_m, grid_n = _choose_fast_tile(M, N)

        # Accumulation dtype: fp64 for fp64 inputs, fp32 otherwise.
        acc_dtype = A.dtype if A.dtype == torch.float64 else torch.float32

        if grid_m * grid_n >= 128:
            partial = torch.zeros(N, dtype=acc_dtype, device=A.device)
            _abs_norm_kernel[(grid_m * grid_n,)](
                A,
                partial,
                partial,
                M,
                N,
                BM,
                BN,
                grid_n,
                SUM_AXIS=0,
                IS_MIN=is_min,
                TILED=True,
                num_warps=8,
            )
            result = (partial.min() if is_min else partial.max()).view(())
        else:
            init_val = float("inf") if is_min else 0.0
            out = torch.full((), init_val, dtype=torch.float32, device=A.device)
            _dummy = torch.empty(1, device=A.device)
            _abs_norm_kernel[(grid_n,)](
                A,
                out,
                _dummy,
                M,
                N,
                BM,
                BN,
                1,
                SUM_AXIS=0,
                IS_MIN=is_min,
                TILED=False,
                num_warps=8,
            )
            result = out.to(out_dtype).view(())

        if keepdim:
            result = result.reshape(1, 1)
        if result.dtype != out_dtype:
            result = result.to(out_dtype)
        return result

    # --- Batched path -- same kernels as the C++ wrapper ---
    return _batched_kernel_dispatch(A, dim, ord_val, out_dtype, keepdim)


def _ordinf_norm(A, ord_val, dim, keepdim, dtype):
    """Infinity-norm (ord=inf / ord=-inf).  Mirrors C++ ``ordinf_norm``.

    Uses ``_abs_norm_kernel``: same TILED/BATCHED dispatch as ``_ord1_norm``,
    but with SUM_AXIS=1 (row-wise reduction).
    """
    d0, d1 = dim
    out_dtype = dtype if dtype is not None else A.dtype
    is_min = ord_val < 0

    # --- Simple 2D matrix path ---
    if A.ndim == 2 and d0 == 0 and d1 == 1:
        M, N = A.shape

        # Few rows/cols: Triton parallelism limited → direct PyTorch.
        if M <= 2 or N <= 2:
            row_sums = torch.sum(torch.abs(A), 1)
            result = row_sums.min() if is_min else row_sums.max()
            if result.dtype != out_dtype:
                result = result.to(out_dtype)
            if keepdim:
                result = result.reshape(1, 1)
            return result

        acc_dtype = A.dtype if A.dtype == torch.float64 else torch.float32

        BM, BN, grid_m, grid_n = _choose_fast_tile(M, N)

        if grid_m * grid_n >= 512:
            partial = torch.zeros(M, dtype=acc_dtype, device=A.device)
            _abs_norm_kernel[(grid_m * grid_n,)](
                A,
                partial,
                partial,
                M,
                N,
                BM,
                BN,
                grid_n,
                SUM_AXIS=1,
                IS_MIN=is_min,
                TILED=True,
                num_warps=8,
            )
            result = (partial.min() if is_min else partial.max()).view(())
        else:
            init_val = float("inf") if is_min else 0.0
            out = torch.full((), init_val, dtype=torch.float32, device=A.device)
            _dummy = torch.empty(1, device=A.device)
            _abs_norm_kernel[(grid_m,)](
                A,
                out,
                _dummy,
                M,
                N,
                BM,
                BN,
                1,
                SUM_AXIS=1,
                IS_MIN=is_min,
                TILED=False,
                num_warps=8,
            )
            result = out.to(out_dtype).view(())

        if keepdim:
            result = result.reshape(1, 1)
        if result.dtype != out_dtype:
            result = result.to(out_dtype)
        return result

    # --- Batched path -- same kernels as the C++ wrapper ---
    return _batched_kernel_dispatch(A, dim, ord_val, out_dtype, keepdim)


def _nuc_norm(A, dim, keepdim=False, dtype=None):
    """Nuclear norm (ord='nuc').  Mirrors C++ ``nuc_norm``."""
    d0, d1 = dim

    # Move target dims to last two positions.
    ndim = A.ndim
    all_dims = list(range(ndim))
    remaining = [d for d in all_dims if d not in (d0, d1)]
    perm = remaining + [d0, d1]
    A_perm = A.permute(perm) if perm != all_dims else A
    if dtype is not None:
        A_perm = A_perm.to(dtype)

    *batch_dims, M, N = A_perm.shape
    s = _svdvals_for_norm(A_perm)  # (..., K)
    result = s.sum(dim=-1, keepdim=False)

    if keepdim:
        d0_sorted, d1_sorted = sorted([d0, d1])
        result = result.unsqueeze(d0_sorted).unsqueeze(d1_sorted)
    return result


# ===========================================================================
# Main entry point
# ===========================================================================


def linalg_matrix_norm(A, ord="fro", dim=(-2, -1), keepdim=False, dtype=None):
    """Matrix norm -- main entry point.

    Mirrors the C++ dispatch in ``lib/linalg_matrix_norm.cpp``.  The C++
    wrapper is preferred when available (lower Python overhead); the Python
    helpers provide an equivalent fallback for backends without the C++
    extension.

    Dispatch order::

        1. validate inputs + dtype guard for SVD-based ords
        2. string ord: "fro" → C++ (if available) else _fro_norm
                       "nuc" → C++ (if available) else _nuc_norm
        3. numeric ord: C++ linalg_matrix_norm (if available and ord handled)
                        else per-ord Python helper:
                          abs==2  → _ord2_norm
                          abs==1  → _ord1_norm
                          isinf   → _ordinf_norm
    """
    logger.debug("GEMS LINALG_MATRIX_NORM")

    # --- validate -----------------------------------------------------------
    if A.ndim < 2:
        raise RuntimeError(
            f"linalg.matrix_norm: A must be at least 2-D, got shape {A.shape}"
        )
    dim = list(dim)
    if len(dim) != 2:
        raise RuntimeError(f"linalg.matrix_norm: dim must be a 2-tuple, got {dim}")
    dim = [d % A.ndim for d in dim]
    if dim[0] == dim[1]:
        raise RuntimeError(
            f"linalg.matrix_norm: dims must be different, got ({dim[0]}, {dim[1]})"
        )

    # --- dtype guard for SVD-based ords -------------------------------------
    _svd_ord = (isinstance(ord, str) and ord == "nuc") or (
        not isinstance(ord, str) and abs(float(ord)) == 2
    )
    if _svd_ord and A.dtype in (torch.float16, torch.bfloat16):
        A = A.float()  # upcast to fp32 for SVD

    # --- string ord: fro / nuc ----------------------------------------------
    if isinstance(ord, str):
        if ord == "fro":
            return _fro_norm(A, dim, keepdim, dtype)
        if ord == "nuc":
            return _nuc_norm(A, dim=dim, keepdim=keepdim, dtype=dtype)
        raise RuntimeError(
            f"linalg.matrix_norm: Order '{ord}' not supported. " "Use 'fro' or 'nuc'."
        )

    # --- numeric ord --------------------------------------------------------
    ord_val = float(ord)
    if ord_val not in _SUPPORTED_NUMERIC:
        raise RuntimeError(
            f"linalg.matrix_norm: Order {ord} not supported. "
            "Use 1, -1, 2, -2, inf, -inf."
        )

    abs_ord = abs(ord_val)
    if abs_ord == 2.0:
        return _ord2_norm(A, ord_val, dim, keepdim, dtype)
    if abs_ord == 1.0:
        return _ord1_norm(A, ord_val, dim, keepdim, dtype)
    if math.isinf(abs_ord):
        return _ordinf_norm(A, ord_val, dim, keepdim, dtype)

    raise RuntimeError(f"linalg.matrix_norm: Order {ord} not supported.")


# ===========================================================================
# ===========================================================================
# Parallel Jacobi step kernel -- Brent-Luk ordering, grid=(batch, K/2).
# ===========================================================================


@libentry()
@triton.jit
def _parallel_jacobi_step_kernel(
    A_WORK,
    K,
    ROWS,
    STEP,
    BLOCK_R: tl.constexpr,
):
    """Brent-Luk parallel Jacobi step.  Grid=(batch, K/2).

    Compute dtype is inferred from the work buffer: fp64 buffers give the
    cuSOLVER-gesvdj-matching ~1e-7 residual floor; fp32 buffers (native-dtype
    mode) compute in fp32 to match PyTorch CUDA f32 gesvdj."""
    pid0 = tl.program_id(0)  # batch
    j = tl.program_id(1)  # pair index in [0, K/2)
    rows = tl.arange(0, BLOCK_R)
    rmask = rows < ROWS
    km1 = K - 1
    kh = K // 2
    dtype = A_WORK.dtype.element_ty

    # Normalise integer types: tl.program_id returns int32, but when called
    # from C++ TritonJIT the scalar args (K, ROWS, STEP) may be int64.
    # Force j and STEP to K's type so both if/else branches agree.
    j = j + (K - K)  # K - K = 0 in K's type → promotes j
    step_val = STEP + (K - K)  # promote STEP to K's type

    # Brent-Luk pair assignment for step s
    if j == 0:
        p = step_val
        q = km1  # pivot column
    else:
        p = (step_val + j) % km1
        q = (step_val - j + km1) % km1

    valid = j < kh
    aw = A_WORK + pid0 * K * ROWS
    ap = tl.load(aw + p * ROWS + rows, mask=rmask & valid, other=0.0).to(dtype)
    aq = tl.load(aw + q * ROWS + rows, mask=rmask & valid, other=0.0).to(dtype)

    alpha = tl.sum(ap * ap)
    beta = tl.sum(aq * aq)
    gamma = tl.sum(ap * aq)
    # Use max(|alpha|,|beta|) instead of sqrt(alpha*beta) for the
    # threshold: tl.maximum is bit-exact on all GPU architectures,
    # whereas tl.sqrt and fp64 multiply can differ by 1-2 ULPs across
    # SM versions, causing the active/inactive decision to flip for
    # off-diagonal values near the threshold boundary.
    threshold = 1.0e-15 * tl.maximum(tl.abs(alpha), tl.abs(beta))
    active = tl.abs(gamma) > threshold
    safe_gamma = tl.where(active, gamma, 1.0)
    tau = (beta - alpha) / (2.0 * safe_gamma)
    sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
    t = sign_tau / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
    c = tl.rsqrt(1.0 + t * t)
    s_rot = t * c
    c = tl.where(active, c, 1.0)
    s_rot = tl.where(active, s_rot, 0.0)

    new_ap = c * ap - s_rot * aq
    new_aq = s_rot * ap + c * aq
    tl.store(aw + p * ROWS + rows, new_ap, mask=rmask & valid)
    tl.store(aw + q * ROWS + rows, new_aq, mask=rmask & valid)


# ===========================================================================
# SVD host wrappers — _svdvals_rank2, _svdvals_hybrid, _svdvals_for_norm
# ===========================================================================


def _svdvals_rank2(input):
    """Closed-form singular values for k=2 matrices.  Mirrors C++ ``svdvals_rank2``."""
    batch, m, n = _svd_shape(input)
    a = input.contiguous().reshape(batch, m, n)
    s = torch.empty((batch, 2), dtype=input.dtype, device=input.device)
    largest = max(m, n)
    block_r = triton.next_power_of_2(largest)
    with torch_device_fn.device(input.device):
        if largest <= 16 and batch >= 16:
            block_b = (
                2 if largest <= 2 else (2 if m >= n else 8) if largest == 16 else 16
            )
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


# ===========================================================================
# DBDSQR fused kernel -- on-device Golub-Kahan QR (fp64, single launch, zero CPU sync).
# ===========================================================================


@libentry()
@triton.jit
def _fused_dbdsqr_kernel(
    D,
    E,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EPS: tl.constexpr,
    MAX_ITERS: tl.constexpr,
    BLOCK_SWEEPS: tl.constexpr,
):
    """Fused DBDSQR: on-device Golub-Kahan QR iteration.

    Single kernel launch per batch element (grid=(batch,)).  All convergence
    checking, block-finding, and zero-shift sweeps happen inside the kernel
    -- zero CPU sync, zero sub-launches.  Matches LAPACK DBDSQR convention.
    Compute dtype is inferred from the D buffer (fp64 default; fp32 in
    native-dtype mode with the fp32 EPS tolerance)."""
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK_K)
    dmask = idx < K
    emask = idx < K - 1
    dtype = D.dtype.element_ty
    eps_val = 1.0e-30

    # Load full bidiagonal into registers
    d = tl.load(D + pid * K + idx, mask=dmask, other=0.0).to(dtype)
    z = tl.zeros([BLOCK_K], dtype=dtype)
    e = tl.where(
        emask, tl.load(E + pid * (K - 1) + idx, mask=emask, other=0.0).to(dtype), z
    )

    # LAPACK BDSQR tolerance
    tol = max(10.0, min(100.0, EPS ** (-1.0 / 8.0))) * EPS

    for _ in range(MAX_ITERS):
        converged = True
        ii = 0
        while ii < K - 1:
            ei = tl.sum(tl.where(idx == ii, e, z))
            di = tl.sum(tl.where(idx == ii, d, z))
            di1 = tl.sum(tl.where(idx == ii + 1, d, z))

            elem_ok = tl.abs(ei) <= tol * (tl.abs(di) + tl.abs(di1))
            if elem_ok:
                e = tl.where(idx == ii, 0.0, e)
                ii += 1
            else:
                ll = ii
                mm = ii + 1
                while mm < K:
                    em1 = tl.sum(tl.where(idx == mm - 1, e, z))
                    dm1 = tl.sum(tl.where(idx == mm - 1, d, z))
                    dm = tl.sum(tl.where(idx == mm, d, z))
                    if tl.abs(em1) > tol * (tl.abs(dm1) + tl.abs(dm)):
                        mm += 1
                    else:
                        mm = K  # sentinel exit

                for _ in range(BLOCK_SWEEPS):
                    one = tl.full([1], 1.0, dtype=dtype)
                    zero = tl.full([1], 0.0, dtype=dtype)
                    cs = tl.sum(one)
                    oldcs = tl.sum(one)
                    oldsn = tl.sum(zero)
                    p = ll
                    while p < mm - 1:
                        dp = tl.sum(tl.where(idx == p, d, z))
                        ep = tl.sum(tl.where(idx == p, e, z))
                        dp1 = tl.sum(tl.where(idx == p + 1, d, z))

                        fv = dp * cs
                        gv = ep
                        rv = tl.sqrt(fv * fv + gv * gv + eps_val)
                        cs_new = tl.where(rv > 1e-30, fv / rv, tl.sum(one))
                        sn = tl.where(rv > 1e-30, gv / rv, tl.sum(zero))
                        if p > ll:
                            e = tl.where(idx == p - 1, oldsn * rv, e)

                        f2 = oldcs * rv
                        g2 = dp1 * sn
                        r2 = tl.sqrt(f2 * f2 + g2 * g2 + eps_val)
                        oldcs_new = tl.where(r2 > 1e-30, f2 / r2, tl.sum(one))
                        oldsn_new = tl.where(r2 > 1e-30, g2 / r2, tl.sum(zero))
                        d = tl.where(idx == p, r2, d)

                        cs = cs_new
                        oldcs = oldcs_new
                        oldsn = oldsn_new
                        p += 1

                    d_mm1 = tl.sum(tl.where(idx == mm - 1, d, z))
                    h = d_mm1 * cs
                    d = tl.where(idx == mm - 1, h * oldcs, d)
                    e = tl.where(idx == mm - 2, h * oldsn, e)

                ii = mm
                converged = False

        if converged:
            pass  # no break in Triton; remaining iters are no-ops

    tl.store(D + pid * K + idx, d, mask=dmask)
    tl.store(E + pid * (K - 1) + idx, e, mask=emask)


def _svdvals_hybrid(input):
    """Hybrid SVD: Jacobi on triangular R, DBDSQR for k=3.
    Mirrors C++ ``svdvals_hybrid``.

      1. Householder QR on A → triangular R (k×k), fp64 for k≥4
      2. Parallel Brent-Luk Jacobi on R (k≥4).
         fp64: 60/50 sweeps (cross-architecture deterministic)
         fp32: 30/40 sweeps (performance-optimised)
      3. DBDSQR fallback (k=3): QR + bidiag + Golub-Kahan iteration
      4. Sort descending"""

    batch, m, n = _svd_shape(input)
    k = min(m, n)
    tall = m >= n
    a = input.contiguous().reshape(batch, m, n)
    device = input.device

    # 1. QR: A → triangular R (k×k), always fp64 for k≥4 to avoid precision
    # loss in tall/wide matrices (fp32 Householder QR loses ~3e-3 on max SV).
    block_m = triton.next_power_of_2(min(max(m, n) if m >= n else n, 256))
    block_n = 32
    qr_use_fp64 = k >= 4

    if tall:
        a_qr = a.float().clone() if not qr_use_fp64 else a.double().clone()
        M_qr, N_qr = m, n
    else:
        a_qr = a.transpose(-2, -1).contiguous()
        a_qr = a_qr.float().clone() if not qr_use_fp64 else a_qr.double().clone()
        M_qr, N_qr = n, m
    block_m = triton.next_power_of_2(min(M_qr, 256))

    Rf = torch.zeros(
        (batch, k, k),
        dtype=torch.float64 if qr_use_fp64 else torch.float32,
        device=device,
    )

    with torch_device_fn.device(device):
        _householder_qr_r_kernel[(batch,)](
            a_qr,
            Rf,
            M_qr,
            N_qr,
            k,
            stride_b=a_qr.stride(0),
            stride_m=a_qr.stride(-2),
            stride_n=a_qr.stride(-1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            USE_FP64=qr_use_fp64,
            num_warps=4,
        )

    # 2. Jacobi SVD on R (k≥4).  Sweeps tuned by dtype:
    #      fp64: 60/50 sweeps — drowns out ULPs-level tl.sum/tl.sqrt
    #            differences across GPU SM versions so the final column
    #            norms are identical regardless of architecture.
    #      fp32: 30/40 sweeps — precision floor is higher, fewer sweeps
    #            suffice and cross-device differences stay within atol.
    use_jacobi = k >= 4

    if use_jacobi:
        # fp64: extra sweeps to drown out ULPs-level differences in tl.sum
        # and tl.sqrt across GPU SM versions.  On SM 8.0 the MUFU and
        # reduction strategies differ from SM 9.0; running enough sweeps
        # drives the off-diagonal deep below the convergence floor so the
        # final column norms are identical regardless of architecture.
        if input.dtype == torch.float64:
            _JACOBI_SWEEPS = 60 if k <= 48 else 50
        else:
            _JACOBI_SWEEPS = 30 if k <= 48 else 40
        block_r = triton.next_power_of_2(k)

        # R → column-major for Jacobi (dtype flows from Rf)
        a_work = Rf.transpose(1, 2).contiguous()

        with torch_device_fn.device(device):
            for _ in range(_JACOBI_SWEEPS):
                for step in range(k - 1):
                    _parallel_jacobi_step_kernel[(batch, k // 2)](
                        a_work,
                        k,
                        k,
                        step,
                        BLOCK_R=block_r,
                        num_warps=1 if block_r <= 64 else 4,
                    )
                # SM 8.0 (A100) workaround: synchronise + clone after each
                # sweep.  On A100, Triton kernel launches may not act as
                # implicit memory fences; a synchronize followed by a
                # clone (which launches a device-side memcpy kernel) forces
                # all column writes from the sweep to be globally visible
                # before the next sweep's kernels read them.  Without this,
                # results are non-deterministic run-to-run on A100.
                torch_device_fn.synchronize()
                a_work = a_work.clone()

        # Extract singular values: after convergence, column norms = σᵢ.
        # We use norm(dim=-1) instead of torch.bmm + diagonal + sqrt
        # because torch.bmm (cuBLAS) is non-deterministic across GPU
        # architectures — a few ULPs difference can cause spurious
        # fallback to DBDSQR on some devices (~3% of calls).
        col_norms = a_work.norm(dim=-1)
        s_sorted = col_norms.topk(k, dim=-1, largest=True).values
        return s_sorted.reshape(*input.shape[:-2], k).to(input.dtype)

    # 3. DBDSQR: QR + bidiag + Golub-Kahan iteration.
    #    Used for: k=3, and fp64 with k ≤ 128.
    work_dtype = torch.float64
    block_k = triton.next_power_of_2(k)
    d_f64 = torch.zeros((batch, k), dtype=work_dtype, device=device)
    e_f64 = torch.zeros((batch, k - 1), dtype=work_dtype, device=device)

    with torch_device_fn.device(device):
        # QR on original A (step 1 QR corrupted a_qr with Householder vectors)
        if tall:
            a_qr_f64 = a.to(work_dtype).clone()
        else:
            a_qr_f64 = a.transpose(-2, -1).contiguous().to(work_dtype)
        R_f64 = torch.zeros((batch, k, k), dtype=work_dtype, device=device)
        _householder_qr_r_kernel[(batch,)](
            a_qr_f64,
            R_f64,
            M_qr,
            N_qr,
            k,
            stride_b=a_qr_f64.stride(0),
            stride_m=a_qr_f64.stride(-2),
            stride_n=a_qr_f64.stride(-1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            USE_FP64=True,
            num_warps=4,
        )
        # bidiag on R (dtype flows from R)
        _bidiag_kernel[(batch,)](
            R_f64,
            d_f64,
            e_f64,
            K=k,
            BLOCK_K=block_k,
            num_warps=1 if block_k <= 64 else 4,
        )

    # Synchronise: bidiag kernel writes d_f64/e_f64 on a Triton-managed
    # stream; DBDSQR must not read them until the writes are complete,
    # otherwise partially-updated bidiagonal data causes non-deterministic
    # convergence behaviour (~3 % of calls).
    torch_device_fn.synchronize()

    # DBDSQR: fused on-device Golub-Kahan QR iteration
    block_k_dbdsqr = triton.next_power_of_2(k)
    max_iters, num_w = _select_dbdsqr_params(k)

    with torch_device_fn.device(device):
        _fused_dbdsqr_kernel[(batch,)](
            d_f64,
            e_f64,
            K=k,
            BLOCK_K=block_k_dbdsqr,
            EPS=2.220446049250313e-16,  # fp64 machine epsilon
            MAX_ITERS=max_iters,
            BLOCK_SWEEPS=50,
            num_warps=num_w,
        )

    s_out = d_f64.abs().to(input.dtype)

    s_sorted = s_out.topk(k, dim=-1, largest=True).values
    return s_sorted.reshape(*input.shape[:-2], k).to(input.dtype)


def _svdvals_for_norm(A):
    """SVD dispatch for ord=2/-2/nuc.  Returns (..., K).

    Precision strategy (matches PyTorch CUDA gesvdj / gesvd):
      fp64 → fp64,  fp32 → fp32,  fp16/bf16 → upcast to fp32.
    """
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

    # --- rank-1: single singular value = Frobenius norm via _fro_kernel ---
    if k == 1:
        flat = A.reshape(batch, M * N)
        s = torch.empty(batch, 1, dtype=torch.float32, device=A.device)
        blk_n = triton.next_power_of_2(min(M * N, 512))
        _fro_kernel[(batch,)](
            flat, s, 0, M * N, 1, blk_n, 1, TILE_2D=False, num_warps=8
        )
        return s.reshape(*batch_dims, 1).to(in_dtype)

    # --- rank-2 closed form ------------------------------------------------
    if k == 2 and rows <= _RANK2_BLOCK_R_MAX:
        return _svdvals_rank2(A).to(in_dtype)

    # --- gesvd: all k≥3 through _svdvals_hybrid --------------------
    if 2 < k <= 512 and rows <= 2048:
        return _svdvals_hybrid(A).to(in_dtype)
    # --- unsupported -------------------------------------------------------
    raise NotImplementedError(
        f"FlagGems svdvals: unsupported matrix shape. "
        f"Got batch={batch}, m={M}, n={N} (k={k}, rows={rows}). "
        f"Supported: k=1 (L2 norm), k==2 with rows<={_RANK2_BLOCK_R_MAX}, "
        f"or 2<k<=512 with rows<=2048."
    )
