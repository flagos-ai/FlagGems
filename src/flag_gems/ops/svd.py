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
    # Dimensions
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
    """One-sided Jacobi SVD kernel with column-major global memory buffers.

    Each program instance handles one matrix from the batch.
    Uses column-major scratch buffers in global memory for O(M) column
    access instead of O(M*N) broadcast extraction from registers.
    After transpose normalization, M >= N always holds, so K = N.
    Singular values are output unsorted (sorting done in wrapper).
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

    # Jacobi sweeps with column-major load/store
    for _sweep in range(NUM_SWEEPS):
        for p in range(N):
            for q in range(p + 1, N):
                # Load columns p and q from A_work — O(M) direct load
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

                # Check convergence for this pair
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

                # Apply rotation to A columns — O(M) store
                new_a_p = c * a_p - s * a_q
                new_a_q = s * a_p + c * a_q
                tl.store(
                    aw_base + p * aw_col_stride + row_idx, new_a_p, mask=row_mask
                )
                tl.store(
                    aw_base + q * aw_col_stride + row_idx, new_a_q, mask=row_mask
                )

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
                    tl.store(
                        vw_base + p * vw_col_stride + v_idx, new_v_p, mask=v_m
                    )
                    tl.store(
                        vw_base + q * vw_col_stride + v_idx, new_v_q, mask=v_m
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

    # Store S (unsorted — sorting done in wrapper)
    s_ptrs = S_ptr + pid * batch_stride_S + s_idx
    tl.store(s_ptrs, S_vals, mask=s_mask)

    # Compute and store U = A_work_col / S_col (normalize columns)
    if compute_uv:
        for j in range(N):
            a_col_j = tl.load(
                aw_base + j * aw_col_stride + row_idx, mask=row_mask, other=0.0
            )
            s_j = tl.sum(S_vals * (s_idx == j).to(tl.float32))
            safe_s_j = tl.where(
                s_j > 1e-10, s_j, tl.full((), 1.0, dtype=tl.float32)
            )
            u_col_j = a_col_j / safe_s_j

            u_mask_j = row_mask & (j < out_k_U)
            u_ptrs = (
                U_ptr
                + pid * batch_stride_U
                + row_idx * m_stride_U
                + j * k_stride_U
            )
            tl.store(u_ptrs, u_col_j, mask=u_mask_j)

        # Store V from V_work
        v_out_idx = tl.arange(0, BLOCK_N)
        for j in range(N):
            v_col_j = tl.load(
                vw_base + j * vw_col_stride + v_out_idx,
                mask=v_out_idx < N,
                other=0.0,
            )
            v_mask_j = (v_out_idx < N) & (j < out_k_V)
            v_ptrs = (
                V_ptr
                + pid * batch_stride_V
                + v_out_idx * n_stride_V
                + j * k_stride_V
            )
            tl.store(v_ptrs, v_col_j, mask=v_mask_j)


def svd(input, some=True, compute_uv=True):
    """Compute the Singular Value Decomposition of a matrix.

    Implements SVD using the One-sided Jacobi algorithm in pure Triton.
    Uses column-major global memory scratch buffers for efficient column
    access. Supports matrices up to MAX_SVD_DIM x MAX_SVD_DIM per batch.

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

    # Enforce maximum dimension constraint
    assert orig_m <= MAX_SVD_DIM and orig_n <= MAX_SVD_DIM, (
        f"Jacobi SVD kernel supports matrices up to {MAX_SVD_DIM}x{MAX_SVD_DIM}, "
        f"got ({orig_m}, {orig_n}). Consider using torch.linalg.svd for larger matrices."
    )

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
        out_k_U = m
        out_k_V = n

    S_out = torch.empty(batch_size, k, device=input.device, dtype=input.dtype)
    U_out = torch.zeros(batch_size, m, out_k_U, device=input.device, dtype=input.dtype)
    V_out = torch.zeros(batch_size, n, out_k_V, device=input.device, dtype=input.dtype)

    # Allocate column-major scratch buffers
    # A_work shape: (batch_size, n, m) — column j of A is A_work[batch, j, :]
    A_work = torch.empty(
        batch_size, n, m, device=input.device, dtype=torch.float32
    )
    V_work = torch.empty(
        batch_size, n, n, device=input.device, dtype=torch.float32
    )

    BLOCK_M = _next_power_of_2(m)
    BLOCK_N = _next_power_of_2(n)
    num_sweeps = max(10, n)
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
            # Dimensions
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

    # Sort singular values in descending order (outside kernel)
    sorted_indices = torch.argsort(S_out, dim=-1, descending=True)
    S_out = torch.gather(S_out, -1, sorted_indices)
    if compute_uv:
        idx_U = sorted_indices.unsqueeze(-2).expand_as(U_out)
        U_out = torch.gather(U_out, -1, idx_U)
        idx_V = sorted_indices.unsqueeze(-2).expand_as(V_out)
        V_out = torch.gather(V_out, -1, idx_V)

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
