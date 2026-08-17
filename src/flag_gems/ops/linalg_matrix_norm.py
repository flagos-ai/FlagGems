"""linalg_matrix_norm -- NVIDIA/CUDA (generic) matrix norm.

    ord=2/-2     -> max/min singular value (fp64 QR + Jacobi or bidiag+DBDSQR)
    ord=1/-1     -> max/min absolute column sum
    ord=inf/-inf -> max/min absolute row sum
    ord="fro"    -> Frobenius norm = sqrt(sum A_ij^2)
    ord="nuc"    -> nuclear norm = sum sigma_k (sum of singular values)

7 shared Triton kernels (also imported by the per-vendor backend overrides):
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

# FlagGems CUDA-native computational operators (replace torch.abs/torch.all/torch.max/
# torch.min/torch.norm/torch.sqrt/torch.sum/torch.topk/torch.amax with FlagGems equivalents).
from flag_gems.ops.abs import abs as gems_abs
from flag_gems.ops.all import all as gems_all
from flag_gems.ops.amax import amax
from flag_gems.ops.max import max as gems_max
from flag_gems.ops.max import max_dim
from flag_gems.ops.min import min as gems_min
from flag_gems.ops.sqrt import sqrt as gems_sqrt
from flag_gems.ops.sum import sum_dim
from flag_gems.ops.topk import topk as gems_topk
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
    USE_FP64: tl.constexpr = True,
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
    # Accumulate sum-of-squares in fp64 when available: fp32 summation error
    # grows as N * eps ≈ 4e-3 (before sqrt) for N=65536 in the 1D path, and
    # fp32 atomic_add across tiles gives ~2e-6 in the 2D path.  fp64
    # accumulation eliminates both noise floors so the Frobenius norm
    # meets CPU-LAPACK comparison tolerances regardless of path.
    # Backends without fp64 support (iluvatar, ascend, etc.) use fp32.
    acc_dtype = tl.float64 if USE_FP64 else tl.float32
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
_TSQR_MIN_ASPECT_RATIO = 8  # Use TSQR when M/N >= 8


def _is_nofp64_backend():
    """Whether the current device backend lacks fp64 compute support.

    NVIDIA CUDA supports fp64; the per-vendor backend files handle their own
    fp64 policy, so this generic path never uses fp32-only accumulation."""
    return False


def _use_fp64_acc(input_dtype):
    """Whether to use fp64 for internal accumulation in reduction kernels.

    - bf16/f16 → fp32 accumulator (fp64 is overkill for half precision)
    - f32      → fp64 if backend supports it, else fp32
    - f64      → fp64
    """
    if input_dtype == torch.float64:
        return True
    if input_dtype == torch.float32:
        return not _is_nofp64_backend()
    return False  # fp16/bf16 → fp32


def _acc_dtype(input_dtype):
    """Accumulator dtype for host-side reduction buffers.

    Mirrors ``_use_fp64_acc`` but returns a concrete torch dtype for use
    with ``torch.zeros / torch.full``.
    """
    if _use_fp64_acc(input_dtype):
        return torch.float64
    return torch.float32


def _select_dbdsqr_params(k):
    """Autotune DBDSQR parameters by k: larger k needs more iterations.

    Returns (MAX_ITERS, num_warps, BLOCK_SWEEPS).  BLOCK_SWEEPS controls the
    inner Golub-Kahan QR sweep count inside ``_fused_dbdsqr_kernel``.

    THEAD uses more conservative parameters: extra sweep budget adds robustness
    for k=3 (the only k that normally reaches DBDSQR on thead; k ≥ 4 goes
    through _svdvals_gram_jacobi).
    """
    if k <= 32:
        return 30, 1, 50
    elif k <= 64:
        return 50, 1, 50
    elif k <= 128:
        return 100, 4, 50
    else:
        return 200, 4, 50


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

        # Explicit barrier: step j+1's phase-1 loads (and the final phase-3
        # extraction) read A entries written by this step's phase-2 stores.
        # The dependency is genuinely cross-thread: phase 2 writes column
        # j+1 as part of a 2-D [BLOCK_M, BLOCK_N] tile, while phase 1 of
        # step j+1 re-reads that column as a 1-D [BLOCK_M] vector -- Triton
        # assigns the same matrix element to different threads in those two
        # layouts, so a thread can load a value some other thread stored.
        # No barrier is needed across column tiles (each tile's pass-1 reads
        # columns untouched by the previous tile's pass-2), but between
        # steps the write→read ordering must be explicit: on NVIDIA/thead
        # the timing happens to mask the race, while the MetaX backend
        # corrupts R intermittently under load (bit-identical inputs →
        # bitwise-different R in a fraction of launches, deviations starting
        # exactly at a random step's diagonal).
        tl.debug_barrier()

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
    USE_FP64: tl.constexpr = False,
):
    # Accumulate in fp64 when USE_FP64=True for better precision on f32/f64
    # inputs.  bf16/f16 inputs always use fp32 (USE_FP64=False).
    # Final atomic_min/atomic_max still require fp32 on most hardware.
    acc_dtype = tl.float64 if USE_FP64 else tl.float32

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
            acc_dtype
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
            acc = tl.zeros([BLOCK_N], dtype=acc_dtype)
            for row_start in range(0, M, BLOCK_M):
                rows = row_start + tl.arange(0, BLOCK_M)[:, None]
                mask = (rows < M) & col_mask[None, :]
                x = tl.load(
                    X + base + rows * N + cols[None, :], mask=mask, other=0.0
                ).to(acc_dtype)
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
            acc = tl.zeros([BLOCK_M, 1], dtype=acc_dtype)
            for col_start in range(0, N, BLOCK_N):
                cols = col_start + tl.arange(0, BLOCK_N)[None, :]
                mask = row_mask & (cols < N)
                x = tl.load(X + base + rows * N + cols, mask=mask, other=0.0).to(
                    acc_dtype
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
        use_fp64 = _use_fp64_acc(A.dtype)
        if total <= 65536:
            flat = A.reshape(1, total)
            tmp = torch.empty(1, dtype=out_dtype, device=A.device)
            _fro_kernel[(1,)](
                flat,
                tmp,
                0,
                total,
                1,
                512,
                1,
                TILE_2D=False,
                USE_FP64=use_fp64,
                num_warps=8,
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
            out = torch.zeros((), dtype=_acc_dtype(A.dtype), device=A.device)
            _fro_kernel[(grid_size,)](
                A,
                out,
                M,
                N,
                BM,
                BN,
                grid_n,
                TILE_2D=True,
                USE_FP64=use_fp64,
                num_warps=8,
            )
            result = gems_sqrt(out).to(out_dtype)
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
        flat,
        result,
        0,
        mat_size,
        1,
        blk_n,
        1,
        TILE_2D=False,
        USE_FP64=_use_fp64_acc(A.dtype),
        num_warps=8,
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
    use_fp64 = _use_fp64_acc(Ab.dtype)

    if math.isinf(abs_ord):
        # --- inf/-inf: multi-block per matrix (row-parallel) ---
        tile_m = 16
        grid_dim = triton.cdiv(mat_M, tile_m)
        blk_dim = triton.next_power_of_2(min(mat_N, 256))
        init_val = float("inf") if is_min else 0.0
        # Output buffer must be fp32: tl.atomic_min / tl.atomic_max only
        # support fp32.  USE_FP64 controls internal summation precision.
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
            USE_FP64=use_fp64,
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
            USE_FP64=use_fp64,
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

        use_fp64 = _use_fp64_acc(A.dtype)

        if grid_m * grid_n >= 128:
            partial = torch.zeros(N, dtype=_acc_dtype(A.dtype), device=A.device)
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
                USE_FP64=use_fp64,
                num_warps=8,
            )
            result = (gems_min(partial) if is_min else gems_max(partial)).view(())
        else:
            init_val = float("inf") if is_min else 0.0
            # Output buffer must be fp32: tl.atomic_min / tl.atomic_max only
            # support fp32.  Precision gain comes from internal fp64 summation
            # when USE_FP64=True, not from the output dtype.
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
                USE_FP64=use_fp64,
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

        # Few rows/cols: Triton parallelism limited → direct computation.
        if M <= 2 or N <= 2:
            row_sums = sum_dim(gems_abs(A), dim=(1,))
            result = gems_min(row_sums) if is_min else gems_max(row_sums)
            if result.dtype != out_dtype:
                result = result.to(out_dtype)
            if keepdim:
                result = result.reshape(1, 1)
            return result

        use_fp64 = _use_fp64_acc(A.dtype)

        BM, BN, grid_m, grid_n = _choose_fast_tile(M, N)

        if grid_m * grid_n >= 512:
            partial = torch.zeros(M, dtype=_acc_dtype(A.dtype), device=A.device)
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
                USE_FP64=use_fp64,
                num_warps=8,
            )
            result = (gems_min(partial) if is_min else gems_max(partial)).view(())
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
                USE_FP64=use_fp64,
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
    result = sum_dim(s, dim=(-1,), keepdim=False)

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
            f"linalg_matrix_norm: A must be at least 2-D, got shape {A.shape}"
        )
    dim = list(dim)
    if len(dim) != 2:
        raise RuntimeError(f"linalg_matrix_norm: dim must be a 2-tuple, got {dim}")
    dim = [d % A.ndim for d in dim]
    if dim[0] == dim[1]:
        raise RuntimeError(
            f"linalg_matrix_norm: dims must be different, got ({dim[0]}, {dim[1]})"
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
            f"linalg_matrix_norm: Order '{ord}' not supported. " "Use 'fro' or 'nuc'."
        )

    # --- numeric ord --------------------------------------------------------
    ord_val = float(ord)
    if ord_val not in _SUPPORTED_NUMERIC:
        raise RuntimeError(
            f"linalg_matrix_norm: Order {ord} not supported. "
            "Use 1, -1, 2, -2, inf, -inf."
        )

    abs_ord = abs(ord_val)
    if abs_ord == 2.0:
        return _ord2_norm(A, ord_val, dim, keepdim, dtype)
    if abs_ord == 1.0:
        return _ord1_norm(A, ord_val, dim, keepdim, dtype)
    if math.isinf(abs_ord):
        return _ordinf_norm(A, ord_val, dim, keepdim, dtype)

    raise RuntimeError(f"linalg_matrix_norm: Order {ord} not supported.")


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


def _tsqr_r(a, M, N, device):
    """Two-level TSQR for tall matrices (M >= N, M/N >= 8).

    Stage 1: Partition A into ceil(M/BLOCK_ROWS) chunks, QR each independently.
    Stage 2: Stack R-factors vertically, QR the stack once.

    WARNING: consumes ``a`` as in-place scratch (Householder vectors overwrite
    the input).  Callers must not reuse ``a`` after calling.

    Args:
        a: (batch, M, N) in fp64, tall (M >= N), contiguous
        M, N: matrix dimensions
        device: CUDA device

    Returns:
        R_final: (batch, N, N) in fp64, upper-triangular, or None if
        num_chunks <= 1 (caller should fall back to single-block QR).
    """
    batch = a.shape[0]

    # Choose partition size: > N so stack is smaller than original;
    # target ~M/16 rows per chunk but at least 2*N for numerical benefit.
    # block_rows may be non-power-of-2 after the max() clamp; that is fine
    # because only BLOCK_M (the kernel tile size, always power-of-2) is used
    # for tl.arange inside the kernel.
    target_rows = max(N * 2, M // 16)
    block_rows = triton.next_power_of_2(min(target_rows, max(N * 2, M // 2)))
    block_rows = max(block_rows, N + 8)  # ensure > N

    num_chunks = triton.cdiv(M, block_rows)
    if num_chunks <= 1:
        return None  # degenerate: single chunk -> use single-block QR

    padded_M = num_chunks * block_rows

    # Pad A if needed (zero rows don't affect R)
    if padded_M > M:
        pad = torch.zeros(batch, padded_M - M, N, dtype=torch.float64, device=device)
        a = torch.cat([a, pad], dim=1)

    # Stage 1: QR on each chunk
    # Reshape (batch, padded_M, N) -> (batch * num_chunks, block_rows, N)
    A_chunks = a.reshape(batch * num_chunks, block_rows, N).contiguous()
    R_chunks = torch.zeros(batch * num_chunks, N, N, dtype=torch.float64, device=device)

    # Fence before the chunk QR reads A_chunks: the reshape/contiguous (and
    # the torch.cat pad above) are torch ops, and on sm8.0-class devices a
    # Triton launch does not reliably observe prior writes without a full
    # device sync.  Without this fence the chunk QR occasionally reads a
    # partially-copied matrix under load -- observed on MetaX as rare R
    # deviations of up to ~7e-2 (~2/30 runs), which the Jacobi stage then
    # amplifies into test failures.
    torch_device_fn.synchronize()

    block_m_qr = triton.next_power_of_2(min(block_rows, 256))
    with torch_device_fn.device(device):
        _householder_qr_r_kernel[(batch * num_chunks,)](
            A_chunks,
            R_chunks,
            block_rows,
            N,
            N,
            stride_b=A_chunks.stride(0),
            stride_m=A_chunks.stride(1),
            stride_n=A_chunks.stride(2),
            BLOCK_M=block_m_qr,
            BLOCK_N=32,
            USE_FP64=True,
            num_warps=4,
        )
    torch_device_fn.synchronize()

    # Stage 2: Stack R-factors and QR the stack.
    # R_chunks: (batch * num_chunks, N, N) -> (batch, num_chunks * N, N)
    # No power-of-2 padding needed: the kernel's BLOCK_M tile size is
    # separately power-of-2, and the row count (M arg) supports masking.
    R_stack = R_chunks.reshape(batch, num_chunks * N, N).contiguous()
    stack_rows = num_chunks * N

    # Fence before the stack QR reads R_stack: same class of failure as the
    # A_chunks fence above -- the contiguous copy is a torch op whose writes
    # are not reliably visible to the immediately following Triton launch on
    # sm8.0-class devices.
    torch_device_fn.synchronize()

    R_final = torch.zeros(batch, N, N, dtype=torch.float64, device=device)
    block_m_stack = triton.next_power_of_2(min(stack_rows, 256))
    with torch_device_fn.device(device):
        _householder_qr_r_kernel[(batch,)](
            R_stack,
            R_final,
            stack_rows,
            N,
            N,
            stride_b=R_stack.stride(0),
            stride_m=R_stack.stride(1),
            stride_n=R_stack.stride(2),
            BLOCK_M=block_m_stack,
            BLOCK_N=32,
            USE_FP64=True,
            num_warps=4,
        )

    return R_final


# ===========================================================================
# Two-sided symmetric Jacobi kernels (THEAD Gram-Jacobi SVD path).
# On-device eigensolver for the fp64 Gram matrix -- replaces the former CPU
# torch.linalg.eigvalsh approach (_svdvals_gram_eigh, removed).
# ===========================================================================


def _qr_rf(a, batch, m, n, k, device, tall, qr_use_fp64, block_m, block_n):
    """QR factorisation producing the triangular R factor (k×k).

    Builds fresh scratch from ``a`` (the QR kernel overwrites its input with
    Householder vectors), then runs either the two-level TSQR (tall,
    high-aspect-ratio inputs) or a single-block Householder QR.  Returns
    (Rf, tsqr_applied).
    """
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

    _tsqr_applied = False
    # Fence before the QR kernel reads a_qr: its backing copies
    # (transpose/contiguous/clone) are torch ops, and on sm8.0-class devices
    # a Triton launch does not reliably observe prior writes without a full
    # device sync (see the QR->Jacobi fence note below).  Without this, the
    # QR factorisation can read a partially-copied matrix under load -- a
    # rare but real corruption (observed once as a ~1.5e-3 sigma_max error
    # on the k=3 keepdim test during a full-suite run).
    torch_device_fn.synchronize()
    with torch_device_fn.device(device):
        # TSQR disabled for THEAD: two-level QR introduces R-factor
        # perturbations that degrade DBDSQR convergence for k ≥ 32.
        if qr_use_fp64 and M_qr // N_qr >= _TSQR_MIN_ASPECT_RATIO:
            R_tsqr = _tsqr_r(a_qr, M_qr, N_qr, device)
            if R_tsqr is not None:
                Rf = R_tsqr
                _tsqr_applied = True
            else:
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
        else:
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
    return Rf, _tsqr_applied


def _qr_r_dbdsqr(a, batch, m, n, k, device, tall, work_dtype, block_m, block_n):
    """Single Householder QR pass on original A for the DBDSQR section.

    Re-factors A in ``work_dtype`` (used when step 1's Rf dtype does not
    match the DBDSQR work dtype, i.e. k=3 on fp64-capable backends).  The
    QR kernel overwrites its input, so fresh scratch is built per call.
    """
    M_qr, N_qr = (m, n) if tall else (n, m)
    if tall:
        a_qr_dbdsqr = a.to(work_dtype).clone()
    else:
        a_qr_dbdsqr = a.transpose(-2, -1).contiguous().to(work_dtype)
    R_dbdsqr = torch.zeros((batch, k, k), dtype=work_dtype, device=device)
    # Fence before the QR kernel reads a_qr_dbdsqr (same class of failure as
    # the a_qr-copy fence in _qr_rf).
    torch_device_fn.synchronize()
    with torch_device_fn.device(device):
        _householder_qr_r_kernel[(batch,)](
            a_qr_dbdsqr,
            R_dbdsqr,
            M_qr,
            N_qr,
            k,
            stride_b=a_qr_dbdsqr.stride(0),
            stride_m=a_qr_dbdsqr.stride(-2),
            stride_n=a_qr_dbdsqr.stride(-1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            USE_FP64=(work_dtype == torch.float64),
            num_warps=4,
        )
    return R_dbdsqr


def _svdvals_hybrid(input):
    """Hybrid SVD: Gram-Jacobi (thead, hygon) / QR+Jacobi+DBDSQR (others).

    Per-backend dispatch:

      THEAD (PingTouGe)
        k ≥ 4:  _svdvals_gram_jacobi (fp64 Gram GEMM on device + parallel
                two-sided Jacobi eigensolver in Triton).  No CPU LAPACK.
        k = 3:  DBDSQR fallback (same as generic, fp64).

      HYGON (DCU)
        k ≥ 3:  _svdvals_gram_jacobi, same as THEAD.  The Triton Householder
                QR kernel is nondeterministic on hygon, so the QR/Jacobi/
                DBDSQR stages are bypassed entirely (odd k uses a dummy
                column in the Brent-Luk pairing, PAIR_K = k + 1).

      ILUVATAR (CoreX)
        k ≥ 4:  fp32 QR + fp32 Jacobi (15–20 sweeps).
                Falls back to fp32 DBDSQR if Jacobi doesn't converge.
        k = 3:  fp32 DBDSQR directly.

      CUDA / MetaX / other
        k > 10:      QR (fp64) + Jacobi (60/50 sweeps fp64, 30/40 sweeps fp32).
        k ∈ (3, 10]: DBDSQR (fp64 or fp32).
        Jacobi → DBDSQR fallback on non-convergence.
    """

    batch, m, n = _svd_shape(input)
    k = min(m, n)
    tall = m >= n
    a = input.contiguous().reshape(batch, m, n)
    device = input.device

    # ---- 1. QR: A → triangular R (k×k) -------------------------------------
    # fp64 for k ≥ 4 on backends that support it (CUDA, MetaX, THEAD).
    # ILUVATAR / Ascend use fp32 throughout (_is_nofp64_backend → True).
    M_qr, N_qr = (m, n) if tall else (n, m)
    block_m = triton.next_power_of_2(min(M_qr, 256))
    block_n = 32
    qr_use_fp64 = k >= 4 and not _is_nofp64_backend()

    Rf, _tsqr_applied = _qr_rf(
        a, batch, m, n, k, device, tall, qr_use_fp64, block_m, block_n
    )

    # Fence before Jacobi reads Rf: on SM 8.0 (A100) Triton kernel launches
    # do not reliably act as memory fences under memory pressure, so a
    # subsequent read of Rf (via Rf.transpose().contiguous() below) can
    # observe stale data.  A full device sync guarantees the QR output is
    # visible before the Jacobi sweep reads it.  Mirrors the per-sweep
    # synchronize()+clone() fence already used inside the Jacobi loop.
    torch_device_fn.synchronize()

    # ---- 2. Jacobi SVD on R ------------------------------------------------
    # THEAD: Jacobi only for 10 < k < 32 (narrow window where Jacobi is
    #   faster than DBDSQR but small enough that thead's fp64 noise is
    #   tolerable).  For k ≥ 4, THEAD is normally caught by the Gram-Jacobi
    #   early-exit above and only reaches here if that breaks down.
    # ILUVATAR: Jacobi for all k ≥ 4, lower sweep counts (fp32 precision
    #   floor is higher, so additional sweeps don't help).
    # OTHER: Jacobi for k > 10.
    # Per-backend Jacobi dispatch.  TSQR matrices always retain Jacobi:
    # DBDSQR doesn't drown out the ULPs-level differences from the two-level
    # QR partitioning, so final singular values vary non-deterministically
    # across runs on SM 8.0 GPUs.  Jacobi's 60 sweeps drive the off-diagonal
    # deep below the convergence floor, producing identical results regardless
    # of GPU arch.
    use_jacobi = k > 10 or _tsqr_applied

    if use_jacobi:
        # Per-backend Jacobi sweep counts
        if Rf.dtype == torch.float64:
            _JACOBI_SWEEPS = 60 if k <= 48 else 50
        else:
            _JACOBI_SWEEPS = 30 if k <= 48 else 40

        block_r = triton.next_power_of_2(k)
        a_work = Rf.transpose(1, 2).contiguous()  # column-major for Jacobi

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
                torch_device_fn.synchronize()
                a_work = a_work.clone()

        # Convergence check: Gram off-diagonal ≤ tol × max diagonal
        gram = a_work @ a_work.transpose(1, 2)
        k_idx = torch.arange(k, device=device)
        diag = gram[:, k_idx, k_idx]
        off_mask = ~torch.eye(k, dtype=torch.bool, device=device)
        max_off = max_dim(gems_abs(gram[:, off_mask]), dim=-1).values
        max_diag = max_dim(gems_abs(diag), dim=-1).values
        rel_tol = 1e-6 if Rf.dtype == torch.float64 else 5e-4
        jacobi_ok = bool(gems_all(max_off <= rel_tol * max_diag).item())

        if jacobi_ok:
            if Rf.dtype == torch.float64:
                col_norms = a_work.norm(dim=-1)
            else:
                col_norms = gems_sqrt((a_work * a_work).sum(dim=-1).clamp(min=0.0))
            s_sorted = gems_topk(col_norms, k, dim=-1, largest=True)[0]
            return s_sorted.reshape(*input.shape[:-2], k).to(input.dtype)
        # Non-converged → fall through to DBDSQR

    # ---- 3. DBDSQR: bidiagonalisation + Golub-Kahan QR ----------------------
    # THEAD reuses Rf from step 1 (normally only reaches here for k=3, where
    # Rf is clean; k ≥ 4 arrives only if Gram-Jacobi breaks down).
    # ILUVATAR re-does QR from original A (step 1 was in fp32, but
    # work_dtype is also fp32, so reuse is fine).
    # OTHER backends re-do QR from original A because step 1's a_qr is
    # corrupted by in-place Householder vectors.
    work_dtype = torch.float32 if _is_nofp64_backend() else torch.float64
    block_k = triton.next_power_of_2(k)
    d_out = torch.zeros((batch, k), dtype=work_dtype, device=device)
    e_out = torch.zeros((batch, k - 1), dtype=work_dtype, device=device)

    with torch_device_fn.device(device):
        # Reuse Rf from step 1 whenever dtypes match — preserves TSQR precision
        # for tall matrices (step 1's QR already applied two-level partitioning).
        # Re-doing single-block QR would lose that precision and re-introduce
        # non-deterministic run-to-run variation for M/N >= 8 on SM 8.0 GPUs.
        # THEAD already unconditionally reuses Rf; iluvatar always takes the
        # Jacobi path above for k >= 4, so this fallback only matters for
        # CUDA/MetaX k <= 10 (no Jacobi).
        if Rf.dtype == work_dtype:
            R_dbdsqr = Rf if Rf.dtype == work_dtype else Rf.to(work_dtype)
        else:
            # Dtype mismatch (e.g. qr_use_fp64=False but work_dtype=fp64):
            # re-do QR on original A with correct dtype.  Only hits k=3 where
            # TSQR was never applied (k < 4 disables fp64 QR), so no precision
            # is lost.
            R_dbdsqr = _qr_r_dbdsqr(
                a, batch, m, n, k, device, tall, work_dtype, block_m, block_n
            )
        # Fence before the bidiagonalisation kernel reads R_dbdsqr: the
        # Rf.to(work_dtype) copy above is a torch op, and without a full
        # device sync the kernel may read it before the copy's writes are
        # visible on sm8.0-class devices (same class of failure as the
        # a_qr-copy fence above).
        torch_device_fn.synchronize()
        _bidiag_kernel[(batch,)](
            R_dbdsqr,
            d_out,
            e_out,
            K=k,
            BLOCK_K=block_k,
            num_warps=1 if block_k <= 64 else 4,
        )

    torch_device_fn.synchronize()

    block_k_dbdsqr = triton.next_power_of_2(k)
    max_iters, num_w, block_sweeps = _select_dbdsqr_params(k)

    with torch_device_fn.device(device):
        _fused_dbdsqr_kernel[(batch,)](
            d_out,
            e_out,
            K=k,
            BLOCK_K=block_k_dbdsqr,
            EPS=(
                1.1920928955078125e-07
                if _is_nofp64_backend()
                else 2.220446049250313e-16
            ),
            MAX_ITERS=max_iters,
            BLOCK_SWEEPS=block_sweeps,
            num_warps=num_w,
        )

    # Fence before the torch-side reads of d_out (gems_abs): Triton->torch
    # handoff needs the same full device sync as the other stage boundaries
    # on sm8.0-class devices.
    torch_device_fn.synchronize()

    s_out = gems_abs(d_out).to(input.dtype)
    s_sorted = gems_topk(s_out, k, dim=-1, largest=True)[0]
    return s_sorted.reshape(*input.shape[:-2], k).to(input.dtype)


def _svdvals_for_norm(A):
    """SVD dispatch for ord=2/-2/nuc.  Returns (..., K).

    Precision strategy (matches PyTorch CUDA gesvdj / gesvd):
      fp64 → fp64,  fp32 → fp32,  fp16/bf16 → upcast to fp32.
      ILUVATAR / Ascend: fp64 inputs raise (no fp64 hardware support).
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
            flat,
            s,
            0,
            M * N,
            1,
            blk_n,
            1,
            TILE_2D=False,
            USE_FP64=True,
            num_warps=8,
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
