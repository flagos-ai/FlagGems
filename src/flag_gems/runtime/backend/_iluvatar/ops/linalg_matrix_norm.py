import logging
import math

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.ops.abs import abs as gems_abs
from flag_gems.ops.all import all as gems_all

# ---- Reuse shared Triton kernels (core kernel reuse) ------------------------
from flag_gems.ops.linalg_matrix_norm import (  # noqa: E402
    _RANK2_BLOCK_R_MAX,
    _abs_norm_kernel,
    _fro_kernel,
    _rank2_svals_kernel,
)
from flag_gems.ops.max import max as gems_max
from flag_gems.ops.max import max_dim
from flag_gems.ops.min import min as gems_min
from flag_gems.ops.sqrt import sqrt as gems_sqrt
from flag_gems.ops.sum import sum_dim
from flag_gems.ops.topk import topk as gems_topk
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_SUPPORTED_NUMERIC = {1, -1, 2, -2, float("inf"), -float("inf")}


def _svd_shape(A):
    """Return (batch, M, N) for an SVD-shaped tensor."""
    if A.ndim < 2:
        return 0, 0, 0
    batch = 1
    for d in A.shape[:-2]:
        batch *= d
    return batch, A.shape[-2], A.shape[-1]


# ===========================================================================
# SVD kernels (QR -> Jacobi -> DBDSQR) -- linear-domain, fp32 only.
# Reused from the pre-refactor generic implementation; avoids the
# Gram-square precision loss for the smallest singular value.
# ===========================================================================


def _select_dbdsqr_params(k):
    """Autotune DBDSQR parameters by k: larger k needs more iterations.

    Returns (MAX_ITERS, num_warps, BLOCK_SWEEPS).  BLOCK_SWEEPS controls the
    inner Golub-Kahan QR sweep count inside ``_fused_dbdsqr_kernel``.
    """
    if k <= 32:
        return 30, 1, 50
    elif k <= 64:
        return 50, 1, 50
    elif k <= 128:
        return 100, 4, 50
    else:
        return 200, 4, 50


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
    DTYPE = tl.float32

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
# gesvd kernels -- fp32 Householder bidiagonalization + fused DBDSQR.
# CoreX has no fp64 path; all work is fp32.
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

    All computation in fp32 (CoreX has no fp64).  The input R is the fp32
    QR output; D and E are fp32, matching the fp32 DBDSQR tolerance used
    by ``_fused_dbdsqr_kernel``.

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


@libentry()
@triton.jit
def _fused_dbdsqr_kernel(
    D,
    E,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EPS: tl.constexpr,
    MAX_ITERS,
    BLOCK_SWEEPS,
):
    """Fused DBDSQR: on-device Golub-Kahan QR iteration.

    Single kernel launch per batch element (grid=(batch,)).  All convergence
    checking, block-finding, and zero-shift sweeps happen inside the kernel
    -- zero CPU sync, zero sub-launches.  Matches LAPACK DBDSQR convention.
    Compute dtype is fp32 (CoreX has no fp64)."""
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

    # MAX_ITERS is a runtime bound (not constexpr) so Triton treats this as a
    # runtime-trip-count loop and does not emit the "loop with constant trip
    # count not unrolled" warning.
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

                # BLOCK_SWEEPS is a runtime bound (not constexpr) — same
                # rationale as the outer loop.
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
    """Hybrid SVD: Jacobi on triangular R, DBDSQR fallback (iluvatar/CoreX).

    CoreX has no fp64 compute path, so everything runs in fp32.

      k ≥ 4:  fp32 QR + fp32 Jacobi (15–20 sweeps).
              Falls back to fp32 DBDSQR if Jacobi doesn't converge.
      k = 3:  fp32 DBDSQR directly.
    """
    batch, m, n = _svd_shape(input)
    k = min(m, n)
    tall = m >= n
    a = input.contiguous().reshape(batch, m, n)
    device = input.device

    # ---- 1. QR: A → triangular R (k×k), always fp32 on CoreX ----
    if tall:
        a_qr = a.float().clone()
        M_qr, N_qr = m, n
    else:
        a_qr = a.transpose(-2, -1).contiguous().float().clone()
        M_qr, N_qr = n, m
    block_m = triton.next_power_of_2(min(M_qr, 256))
    block_n = 32

    Rf = torch.zeros((batch, k, k), dtype=torch.float32, device=device)

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
            num_warps=4,
        )

    # Fence before Jacobi reads Rf: Triton kernel launches do not reliably
    # act as memory fences, so a subsequent read of Rf (via
    # Rf.transpose().contiguous() below) can observe stale data.  A full
    # device sync guarantees the QR output is visible.
    torch_device_fn.synchronize()

    # ---- 2. Jacobi SVD on R (k ≥ 4), fp32 ----
    if k >= 4:
        _JACOBI_SWEEPS = 15 if k <= 48 else 20
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
        gram = flag_gems.bmm(a_work, a_work.transpose(1, 2))
        k_idx = torch.arange(k, device=device)
        diag = gram[:, k_idx, k_idx]
        off_mask = ~torch.eye(k, dtype=torch.bool, device=device)
        max_off = max_dim(gems_abs(gram[:, off_mask]), dim=-1).values
        max_diag = max_dim(gems_abs(diag), dim=-1).values
        rel_tol = 5e-4
        jacobi_ok = bool(gems_all(max_off <= rel_tol * max_diag).item())

        if jacobi_ok:
            col_norms = gems_sqrt(
                flag_gems.clamp(sum_dim(a_work * a_work, (-1,)), mini=0.0)
            )
            s_sorted = gems_topk(col_norms, k, dim=-1, largest=True)[0]
            return s_sorted.reshape(*input.shape[:-2], k).to(input.dtype)
        # Non-converged → fall through to DBDSQR

    # ---- 3. DBDSQR: bidiagonalisation + Golub-Kahan QR (fp32) ----
    work_dtype = torch.float32
    block_k = triton.next_power_of_2(k)
    d_out = torch.zeros((batch, k), dtype=work_dtype, device=device)
    e_out = torch.zeros((batch, k - 1), dtype=work_dtype, device=device)

    with torch_device_fn.device(device):
        # Reuse Rf from step 1: both are fp32, and step 1's QR wrote its
        # Householder vectors to a_qr (a separate buffer), so Rf is clean.
        R_dbdsqr = Rf
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
            EPS=1.1920928955078125e-07,  # fp32 eps
            MAX_ITERS=max_iters,
            BLOCK_SWEEPS=block_sweeps,
            num_warps=num_w,
        )

    s_out = gems_abs(d_out).to(input.dtype)
    s_sorted = gems_topk(s_out, k, dim=-1, largest=True)[0]
    return s_sorted.reshape(*input.shape[:-2], k).to(input.dtype)


def _svdvals_for_norm(A):
    """SVD dispatch for ord=2/-2/nuc.  Returns (..., K) descending.

    ILUVATAR (CoreX) has no fp64 compute path: fp16/bf16 are upcast to fp32,
    and fp64 inputs raise (no fp64 hardware support).  All SVD runs in fp32.
    """
    in_dtype = A.dtype
    if in_dtype in (torch.float16, torch.bfloat16):
        A = A.float()
    if A.dtype == torch.float64:
        raise RuntimeError(
            f"linalg_matrix_norm: fp64 input is not supported on the "
            f"{flag_gems.vendor_name} backend"
        )
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
            USE_FP64=False,  # CoreX has no fp64
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


# ===========================================================================
# Host dispatch (mirrors the generic implementation; fp32 accumulation)
# ===========================================================================


def _fro_norm(A, dim, keepdim, dtype):
    d0, d1 = dim
    out_dtype = dtype if dtype is not None else A.dtype

    if A.ndim == 2 and d0 == 0 and d1 == 1:
        M, N = A.shape
        total = M * N
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
                USE_FP64=False,
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
            out = torch.zeros((), dtype=torch.float32, device=A.device)
            _fro_kernel[(grid_size,)](
                A,
                out,
                M,
                N,
                BM,
                BN,
                grid_n,
                TILE_2D=True,
                USE_FP64=False,
                num_warps=8,
            )
            result = gems_sqrt(out).to(out_dtype)
        if result.dtype != out_dtype:
            result = result.to(out_dtype)
        if keepdim:
            result = result.reshape(1, 1)
        return result

    ndim = A.ndim
    all_dims = list(range(ndim))
    remaining = [d for d in all_dims if d not in (d0, d1)]
    perm = remaining + [d0, d1]
    A_perm = A.permute(perm)
    batch = 1
    for i in range(A_perm.ndim - 2):
        batch *= A_perm.size(i)
    mat_size = A_perm.size(-2) * A_perm.size(-1)
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
        USE_FP64=False,
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
    d0, d1 = dim
    out_dtype = dtype if dtype is not None else A.dtype

    ndim = A.ndim
    all_dims = list(range(ndim))
    remaining = [d for d in all_dims if d not in (d0, d1)]
    perm = remaining + [d0, d1]
    A_perm = A.permute(perm) if perm != all_dims else A
    if dtype is not None:
        A_perm = A_perm.to(dtype)

    s = _svdvals_for_norm(A_perm)
    result = s[..., 0] if ord_val > 0 else s[..., -1]

    if result.dtype != out_dtype:
        result = result.to(out_dtype)
    if keepdim:
        out_shape = list(A.shape)
        out_shape[d0] = 1
        out_shape[d1] = 1
        result = result.reshape(out_shape)
    return result


def _choose_fast_tile(M, N):
    if M <= 1024 and N <= 1024:
        BM, BN = 32, 32
    elif N >= 8 * M or M >= 8 * N:
        BM, BN = min(M, 128), min(N, 128)
    else:
        BM, BN = 128, 32
    if BM > M:
        BM = triton.next_power_of_2(M)
    if BN > N:
        BN = triton.next_power_of_2(N)
    grid_m = triton.cdiv(M, BM)
    grid_n = triton.cdiv(N, BN)
    return BM, BN, int(grid_m), int(grid_n)


def _batched_kernel_dispatch(A, dim, ord_val, out_dtype, keepdim):
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
        tile_m = 16
        blk_dim = triton.next_power_of_2(min(mat_N, 256))
        # CoreX Triton codegen bug: a 512-element tile (16×32) with num_warps=8
        # produces wrong tl.sum results.  Bump tile_m to 32 (32×32=1024).
        if tile_m * blk_dim == 512:
            tile_m = 32
        grid_dim = triton.cdiv(mat_M, tile_m)
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
            USE_FP64=False,
            num_warps=8,
        )
    elif abs_ord == 1.0:
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
            USE_FP64=False,
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
    d0, d1 = dim
    out_dtype = dtype if dtype is not None else A.dtype
    is_min = ord_val < 0

    if A.ndim == 2 and d0 == 0 and d1 == 1:
        M, N = A.shape
        BM, BN, grid_m, grid_n = _choose_fast_tile(M, N)

        if grid_m * grid_n >= 128 or grid_m >= 16:
            partial = torch.zeros(N, dtype=torch.float32, device=A.device)
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
                USE_FP64=False,
                num_warps=8,
            )
            result = (gems_min(partial) if is_min else gems_max(partial)).view(())
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
                USE_FP64=False,
                num_warps=8,
            )
            result = out.to(out_dtype).view(())

        if keepdim:
            result = result.reshape(1, 1)
        if result.dtype != out_dtype:
            result = result.to(out_dtype)
        return result

    return _batched_kernel_dispatch(A, dim, ord_val, out_dtype, keepdim)


def _ordinf_norm(A, ord_val, dim, keepdim, dtype):
    d0, d1 = dim
    out_dtype = dtype if dtype is not None else A.dtype
    is_min = ord_val < 0

    if A.ndim == 2 and d0 == 0 and d1 == 1:
        M, N = A.shape
        BM, BN, grid_m, grid_n = _choose_fast_tile(M, N)

        if grid_m * grid_n >= 512 or grid_n >= 16:
            partial = torch.zeros(M, dtype=torch.float32, device=A.device)
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
                USE_FP64=False,
                num_warps=8,
            )
            result = (gems_min(partial) if is_min else gems_max(partial)).view(())
        else:
            # CoreX Triton bug: a 512-element tile (BM×BN==512) with
            # num_warps=8 produces wrong tl.sum results for SUM_AXIS=1.
            # Double BM so the tile becomes 1024 elements.
            if BM * BN == 512:
                BM = BM * 2
                grid_m = triton.cdiv(M, BM)
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
                USE_FP64=False,
                num_warps=8,
            )
            result = out.to(out_dtype).view(())

        if keepdim:
            result = result.reshape(1, 1)
        if result.dtype != out_dtype:
            result = result.to(out_dtype)
        return result

    return _batched_kernel_dispatch(A, dim, ord_val, out_dtype, keepdim)


def _nuc_norm(A, dim, keepdim=False, dtype=None):
    d0, d1 = dim

    ndim = A.ndim
    all_dims = list(range(ndim))
    remaining = [d for d in all_dims if d not in (d0, d1)]
    perm = remaining + [d0, d1]
    A_perm = A.permute(perm) if perm != all_dims else A
    if dtype is not None:
        A_perm = A_perm.to(dtype)

    s = _svdvals_for_norm(A_perm)
    result = sum_dim(s, dim=(-1,), keepdim=False)

    if keepdim:
        d0_sorted, d1_sorted = sorted([d0, d1])
        result = result.unsqueeze(d0_sorted).unsqueeze(d1_sorted)
    return result


def linalg_matrix_norm(A, ord="fro", dim=(-2, -1), keepdim=False, dtype=None):
    logger.debug("GEMS LINALG_MATRIX_NORM (ILUVATAR)")

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

    _svd_ord = (isinstance(ord, str) and ord == "nuc") or (
        not isinstance(ord, str) and abs(float(ord)) == 2
    )
    if _svd_ord and A.dtype in (torch.float16, torch.bfloat16):
        A = A.float()

    if isinstance(ord, str):
        if ord == "fro":
            return _fro_norm(A, dim, keepdim, dtype)
        if ord == "nuc":
            return _nuc_norm(A, dim=dim, keepdim=keepdim, dtype=dtype)
        raise RuntimeError(
            f"linalg_matrix_norm: Order '{ord}' not supported. Use 'fro' or 'nuc'."
        )

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
