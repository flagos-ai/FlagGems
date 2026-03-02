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

# Maximum matrix dimension supported by the Jacobi SVD kernel.
MAX_SVD_DIM = 64

# Threshold for Triton Jacobi SVD vs fallback to torch.linalg.svd.
# Jacobi SVD is faster for small matrices; for larger ones the cuSOLVER-based
# divide-and-conquer algorithm in torch.linalg.svd is asymptotically better.
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


def _svd_fallback(input, some, compute_uv):
    """Fallback to torch.linalg.svd for matrices exceeding Jacobi threshold.

    Uses cuSOLVER's divide-and-conquer SVD which is asymptotically faster
    for larger matrices (O(N^3) vs Jacobi's O(N^4)).
    """
    U, S, Vh = torch.linalg.svd(input, full_matrices=not some)
    V = Vh.mH  # conjugate transpose (= transpose for real matrices)
    if not compute_uv:
        U = torch.zeros_like(U)
        V = torch.zeros_like(V)
    return SVDResult(U, S, V)


def svd(input, some=True, compute_uv=True):
    """Compute the Singular Value Decomposition of a matrix.

    Implements SVD using the One-sided Jacobi algorithm in pure Triton for
    small matrices (min(m,n) <= threshold), falling back to torch.linalg.svd
    for larger matrices where cuSOLVER's divide-and-conquer is faster.

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

    # Fall back to torch.linalg.svd for large matrices where cuSOLVER's
    # divide-and-conquer algorithm is asymptotically better than Jacobi
    if k > _JACOBI_THRESHOLD:
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

    # Allocate column-major scratch buffers
    A_work = torch.empty(batch_size, n, m, device=input.device, dtype=torch.float32)
    V_work = torch.empty(batch_size, n, n, device=input.device, dtype=torch.float32)

    BLOCK_M = _next_power_of_2(m)
    BLOCK_N = _next_power_of_2(n)

    # Sweep count: 6 sweeps suffices for small matrices with early convergence
    num_sweeps = 6

    # num_warps: 1 is optimal for small BLOCK sizes (fewer scheduling overhead)
    num_warps_val = 1

    grid = (batch_size,)

    with torch_device_fn.device(input.device):
        jacobi_svd_kernel[grid](
            A,
            A_work,
            V_work,
            U_out,
            S_out,
            V_out,
            # Input A strides
            A.stride(0),
            A.stride(1),
            A.stride(2),
            # A_work strides (column-major)
            A_work.stride(0),
            A_work.stride(1),
            # V_work strides (column-major)
            V_work.stride(0),
            V_work.stride(1),
            # Output strides
            U_out.stride(0),
            U_out.stride(1),
            U_out.stride(2),
            S_out.stride(0),
            V_out.stride(0),
            V_out.stride(1),
            V_out.stride(2),
            # Runtime loop bounds
            n,
            num_sweeps,
            # Constexpr dimensions
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
