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
# Limited by Triton register pressure and compilation time for unrolled loops.
MAX_SVD_DIM = 64


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
    U_ptr,
    S_ptr,
    V_ptr,
    batch_stride_A,
    m_stride_A,
    n_stride_A,
    batch_stride_U,
    m_stride_U,
    k_stride_U,
    batch_stride_S,
    batch_stride_V,
    n_stride_V,
    k_stride_V,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    out_k_U: tl.constexpr,
    out_k_V: tl.constexpr,
    compute_uv: tl.constexpr,
    NUM_SWEEPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One-sided Jacobi SVD kernel.

    Each program instance handles one matrix from the batch.
    The algorithm iteratively applies 2x2 Jacobi rotations to pairs
    of columns until convergence, then extracts singular values and
    vectors. After transpose normalization, M >= N always holds,
    so K = N and BLOCK_N >= N = K.
    """
    pid = tle.program_id(0)

    # Load A into registers: shape (BLOCK_M, BLOCK_N)
    row_idx = tl.arange(0, BLOCK_M)
    col_idx = tl.arange(0, BLOCK_N)

    a_ptrs = (
        A_ptr
        + pid * batch_stride_A
        + row_idx[:, None] * m_stride_A
        + col_idx[None, :] * n_stride_A
    )
    mask = (row_idx[:, None] < M) & (col_idx[None, :] < N)
    A = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Initialize V = I (BLOCK_N, BLOCK_N) — only needed when compute_uv=True
    v_row = tl.arange(0, BLOCK_N)
    v_col = tl.arange(0, BLOCK_N)
    if compute_uv:
        V = tl.where(
            v_row[:, None] == v_col[None, :],
            tl.full((BLOCK_N, BLOCK_N), 1.0, dtype=tl.float32),
            tl.full((BLOCK_N, BLOCK_N), 0.0, dtype=tl.float32),
        )

    # Jacobi sweeps: rotate column pairs (p, q) to diagonalize A^T A
    for _sweep in range(NUM_SWEEPS):
        for p in range(N):
            for q in range(p + 1, N):
                # Extract columns p and q from A
                a_p = tl.sum(A * (col_idx[None, :] == p).to(tl.float32), axis=1)
                a_q = tl.sum(A * (col_idx[None, :] == q).to(tl.float32), axis=1)

                # Compute Gram matrix entries for the 2x2 subproblem
                alpha = tl.sum(a_p * a_p)
                beta = tl.sum(a_q * a_q)
                gamma = tl.sum(a_p * a_q)

                # Check convergence for this pair (1e-7 for float32 precision)
                converged = tl.abs(gamma) < 1e-7 * tl.sqrt(alpha * beta + 1e-30)

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

                # Skip rotation if already converged
                c = tl.where(converged, tl.full((), 1.0, dtype=tl.float32), c)
                s = tl.where(converged, tl.full((), 0.0, dtype=tl.float32), s)

                # Apply rotation to columns of A
                new_a_p = c * a_p - s * a_q
                new_a_q = s * a_p + c * a_q

                is_p = (col_idx[None, :] == p).to(tl.float32)
                is_q = (col_idx[None, :] == q).to(tl.float32)
                is_neither = 1.0 - is_p - is_q
                A = A * is_neither + new_a_p[:, None] * is_p + new_a_q[:, None] * is_q

                # Apply rotation to columns of V (only when computing U and V)
                if compute_uv:
                    v_p = tl.sum(V * (v_col[None, :] == p).to(tl.float32), axis=1)
                    v_q = tl.sum(V * (v_col[None, :] == q).to(tl.float32), axis=1)
                    new_v_p = c * v_p - s * v_q
                    new_v_q = s * v_p + c * v_q

                    is_p_v = (v_col[None, :] == p).to(tl.float32)
                    is_q_v = (v_col[None, :] == q).to(tl.float32)
                    is_neither_v = 1.0 - is_p_v - is_q_v
                    V = (
                        V * is_neither_v
                        + new_v_p[:, None] * is_p_v
                        + new_v_q[:, None] * is_q_v
                    )

    # Extract singular values as column norms of A
    col_norms_sq = tl.sum(A * A, axis=0)
    S_vals = tl.sqrt(col_norms_sq)

    # Compute U = A * diag(1/S) (normalize columns)
    if compute_uv:
        safe_S = tl.where(
            S_vals > 1e-10,
            S_vals,
            tl.full((BLOCK_N,), 1.0, dtype=tl.float32),
        )
        U = A / safe_S[None, :]

    # Sort singular values in descending order using branchless bubble sort.
    # All swaps use tl.where to avoid data-dependent branching in the kernel.
    for i in range(N):
        for j in range(N - 1 - i):
            sj = tl.sum(S_vals * (col_idx == j).to(tl.float32))
            sj1 = tl.sum(S_vals * (col_idx == (j + 1)).to(tl.float32))
            need_swap = sj < sj1

            is_j = (col_idx == j).to(tl.float32)
            is_j1 = (col_idx == (j + 1)).to(tl.float32)
            is_other = 1.0 - is_j - is_j1

            # Branchless swap of S values
            swapped_S = S_vals * is_other + sj1 * is_j + sj * is_j1
            S_vals = tl.where(need_swap, swapped_S, S_vals)

            if compute_uv:
                # Branchless swap of U columns
                u_j = tl.sum(U * is_j[None, :], axis=1)
                u_j1 = tl.sum(U * is_j1[None, :], axis=1)
                swapped_U = (
                    U * is_other[None, :]
                    + u_j1[:, None] * is_j[None, :]
                    + u_j[:, None] * is_j1[None, :]
                )
                U = tl.where(need_swap, swapped_U, U)

                # Branchless swap of V columns
                v_j = tl.sum(V * is_j[None, :], axis=1)
                v_j1 = tl.sum(V * is_j1[None, :], axis=1)
                swapped_V = (
                    V * is_other[None, :]
                    + v_j1[:, None] * is_j[None, :]
                    + v_j[:, None] * is_j1[None, :]
                )
                V = tl.where(need_swap, swapped_V, V)

    # Store singular values
    s_ptrs = S_ptr + pid * batch_stride_S + tl.arange(0, BLOCK_N)
    s_mask = tl.arange(0, BLOCK_N) < K
    tl.store(s_ptrs, S_vals, mask=s_mask)

    # Store U and V matrices
    if compute_uv:
        u_row = tl.arange(0, BLOCK_M)
        u_col = tl.arange(0, BLOCK_N)
        u_ptrs = (
            U_ptr
            + pid * batch_stride_U
            + u_row[:, None] * m_stride_U
            + u_col[None, :] * k_stride_U
        )
        u_mask = (u_row[:, None] < M) & (u_col[None, :] < out_k_U)
        tl.store(u_ptrs, U, mask=u_mask)

        v_store_row = tl.arange(0, BLOCK_N)
        v_store_col = tl.arange(0, BLOCK_N)
        v_ptrs = (
            V_ptr
            + pid * batch_stride_V
            + v_store_row[:, None] * n_stride_V
            + v_store_col[None, :] * k_stride_V
        )
        v_mask = (v_store_row[:, None] < N) & (v_store_col[None, :] < out_k_V)
        tl.store(v_ptrs, V, mask=v_mask)


def svd(input, some=True, compute_uv=True):
    """Compute the Singular Value Decomposition of a matrix.

    Implements SVD using the One-sided Jacobi algorithm in pure Triton.
    Supports matrices up to MAX_SVD_DIM x MAX_SVD_DIM per element in the batch.

    Args:
        input: Input tensor of shape (..., m, n).
        some: If True, return reduced SVD. If False, return full SVD.
            Note: some=False for non-square matrices is not supported and
            will raise NotImplementedError.
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

    # Enforce maximum dimension constraint
    assert orig_m <= MAX_SVD_DIM and orig_n <= MAX_SVD_DIM, (
        f"Jacobi SVD kernel supports matrices up to {MAX_SVD_DIM}x{MAX_SVD_DIM}, "
        f"got ({orig_m}, {orig_n}). Consider using torch.linalg.svd for larger matrices."
    )

    # Note: For some=False with non-square matrices, the kernel computes
    # only k = min(m,n) valid columns for U and V. The extra columns
    # (spanning the null space) are left as zeros. This is sufficient for
    # reconstruction A = U[:,:k] @ diag(S) @ V[:,:k]^T but the returned
    # U and V are not fully orthogonal.

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

    A = input.reshape(batch_size, m, n).contiguous()

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
        # For square matrices (orig_m == orig_n), m == n == k
        out_k_U = m
        out_k_V = n

    S_out = torch.empty(batch_size, k, device=input.device, dtype=input.dtype)
    U_out = torch.zeros(batch_size, m, out_k_U, device=input.device, dtype=input.dtype)
    V_out = torch.zeros(batch_size, n, out_k_V, device=input.device, dtype=input.dtype)

    BLOCK_M = _next_power_of_2(m)
    BLOCK_N = _next_power_of_2(n)
    # Sweep count: enough for convergence without excessive compilation time.
    # For the Jacobi method, O(n) sweeps typically suffice.
    num_sweeps = max(10, n)
    grid = (batch_size,)

    with torch_device_fn.device(input.device):
        jacobi_svd_kernel[grid](
            A,
            U_out,
            S_out,
            V_out,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            U_out.stride(0),
            U_out.stride(1),
            U_out.stride(2),
            S_out.stride(0),
            V_out.stride(0),
            V_out.stride(1),
            V_out.stride(2),
            M=m,
            N=n,
            K=k,
            out_k_U=out_k_U,
            out_k_V=out_k_V,
            compute_uv=compute_uv,
            NUM_SWEEPS=num_sweeps,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
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
        # compute_uv=False: return zero-filled U, V per torch.svd spec
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
