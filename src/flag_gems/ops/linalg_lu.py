import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_lu_factor import linalg_lu_factor
from flag_gems.ops.lu_unpack import (
    lu_unpack_l_kernel,
    lu_unpack_p_kernel_large,
    lu_unpack_p_kernel_small,
    lu_unpack_u_kernel,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

LinalgLUResult = namedtuple("LinalgLUResult", ["P", "L", "U"])

# Matrices up to 64x64 are factorized and unpacked in a single fused kernel
# (same threshold as the fast path of linalg_lu_factor).  Larger matrices go
# through the blocked path; fp64 additionally requires the padded work tile
# BLOCK_M*BLOCK_N <= 1024 to keep the in-register elimination loop from
# spilling (see _can_use_fused_kernel).
_LU_FUSED_BLOCK_MAX = 64


# ---------------------------------------------------------------------------
# Input validation — adapted from linalg_lu_factor.py / linalg_lu_factor_ex.py
# ---------------------------------------------------------------------------


def _linalg_lu_check(input, pivot):
    if input.dim() < 2:
        raise RuntimeError(
            "torch.linalg.lu: Expected input to have at least 2 dimensions, "
            f"got {input.dim()}"
        )
    if input.dtype not in (torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems linalg_lu currently supports float32 and float64 only, "
            f"got {input.dtype}"
        )
    m, n = input.shape[-2], input.shape[-1]
    if m == 0 or n == 0:
        raise NotImplementedError(
            "FlagGems linalg_lu currently does not support empty matrices"
        )
    if pivot not in (True, False):
        raise TypeError(f"pivot must be a bool, got {type(pivot)}")
    if not pivot and input.device.type != "cuda":
        raise NotImplementedError(
            "FlagGems linalg_lu: pivot=False is only supported on CUDA devices, "
            f"got device={input.device.type}"
        )


def _can_use_fused_kernel(input):
    m, n = input.shape[-2], input.shape[-1]
    if m > _LU_FUSED_BLOCK_MAX or n > _LU_FUSED_BLOCK_MAX:
        return False
    if input.dtype == torch.float64:
        # fp64 doubles the register pressure of the in-register elimination
        # loop: the work tile is BLOCK_M*BLOCK_N fp64 values.  Up to a 32x32
        # tile the fused kernel beats the blocked path (or ties); at 64x32
        # and larger it spills and was measured 1.5-5.6x slower, so those
        # shapes go through the blocked kernels.
        return triton.next_power_of_2(m) * triton.next_power_of_2(n) <= 1024
    return True


# ---------------------------------------------------------------------------
# Fused kernel: LU factorization + P/L/U extraction in a single launch.
#
# Based on the fast-path kernel of linalg_lu_factor, but instead of writing
# the packed LU matrix and pivots to global memory and unpacking them with
# three more kernels, the factorization result held in registers is written
# out directly as L, U and P:
#
#   - L (m, k): lower triangular with unit diagonal, from work[:, :k]
#   - U (k, n): upper triangular, from work[:k, :]
#   - P (m, m): the row permutation is tracked in registers during the
#     elimination loop (swap entries of perm whenever two rows of work are
#     swapped); at the end P[row, perm[row]] = 1 is written for every row,
#     which also zeroes P — no separate zeroing kernel is needed.
#
# For pivot=False, no permutation is tracked and P is untouched (the caller
# returns an empty (0,) tensor, matching torch semantics).
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def _linalg_lu_fused_kernel(
    A,
    P,
    L,
    U,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PIVOT: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    offsets = pid * M * N + rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    work = tl.load(A + offsets, mask=mask, other=0.0)

    if PIVOT:
        # perm[row] tracks where row ends up after all the row swaps.
        perm = rows

    for j_ind in tl.range(0, K):
        if PIVOT:
            # Extract column j_ind for pivot search.
            col_vals = tl.sum(tl.where(cols[None, :] == j_ind, work, 0.0), axis=1)
            abs_col = tl.abs(col_vals)
            abs_col = tl.where(rows < j_ind, -1.0, abs_col)
            abs_col = tl.where(rows < M, abs_col, -1.0)
            pivot_val = tl.max(abs_col, axis=0)
            pivot_row = tl.min(tl.where(abs_col == pivot_val, rows, BLOCK_M), axis=0)

            # Swap rows j_ind and pivot_row in work.
            row_j = tl.sum(tl.where(rows[:, None] == j_ind, work, 0.0), axis=0)
            row_p = tl.sum(tl.where(rows[:, None] == pivot_row, work, 0.0), axis=0)
            col_mask = cols[None, :] < N
            work = tl.where((rows[:, None] == j_ind) & col_mask, row_p, work)
            work = tl.where((rows[:, None] == pivot_row) & col_mask, row_j, work)

            # Track the same swap in the permutation vector, exactly like
            # lu_unpack_p_kernel_small: swap the VALUES j_ind and pivot_row
            # wherever they currently appear.  This must be a single nested
            # expression — a position-based swap (or two sequential value
            # swaps) tracks the inverse permutation and produces P with
            # P @ A = L @ U instead of the torch convention A = P @ L @ U.
            perm = tl.where(
                perm == j_ind,
                pivot_row,
                tl.where(perm == pivot_row, j_ind, perm),
            )

            # After swap, row j_ind == row_p (already extracted) — reuse.
            u_row = row_p

            # Update col_vals in-place: just swap elements at j_ind and
            # pivot_row, avoiding a full column re-extraction from work.
            old_j = tl.sum(tl.where(rows == j_ind, col_vals, 0.0), axis=0)
            old_p = tl.sum(tl.where(rows == pivot_row, col_vals, 0.0), axis=0)
            col_vals = tl.where(rows == j_ind, old_p, col_vals)
            col_vals = tl.where(rows == pivot_row, old_j, col_vals)
        else:
            col_vals = tl.sum(tl.where(cols[None, :] == j_ind, work, 0.0), axis=1)
            u_row = tl.sum(tl.where(rows[:, None] == j_ind, work, 0.0), axis=0)

        # Pivot is the diagonal element — index into the column vector.
        pivot = tl.sum(tl.where(rows == j_ind, col_vals, 0.0), axis=0)

        # Scale column below diagonal (L factors) and write back.
        scaled_col = tl.where(rows > j_ind, col_vals / pivot, col_vals)
        work = tl.where(
            (rows[:, None] > j_ind) & (cols[None, :] == j_ind),
            scaled_col[:, None],
            work,
        )

        # Rank-1 trailing update: work[j+1:, j+1:] -= scaled_col * u_row.
        update_mask = (rows[:, None] > j_ind) & (cols[None, :] > j_ind)
        work = tl.where(update_mask, work - scaled_col[:, None] * u_row[None, :], work)

    # ---- L: (m, k) lower triangular with unit diagonal ----
    l_mask = (rows[:, None] < M) & (cols[None, :] < K)
    l_vals = tl.where(
        rows[:, None] > cols[None, :],
        work,
        tl.where(rows[:, None] == cols[None, :], 1.0, 0.0),
    )
    tl.store(L + pid * M * K + rows[:, None] * K + cols[None, :], l_vals, mask=l_mask)

    # ---- U: (k, n) upper triangular ----
    u_mask = (rows[:, None] < K) & (cols[None, :] < N)
    u_vals = tl.where(rows[:, None] <= cols[None, :], work, 0.0)
    tl.store(U + pid * K * N + rows[:, None] * N + cols[None, :], u_vals, mask=u_mask)

    # ---- P: (m, m) permutation matrix, P[row, perm[row]] = 1 ----
    # Use `rows` for both dimensions: BLOCK_M = next_pow2(m) >= m always, so
    # every P column is covered even when n < m (cols would only span BLOCK_N).
    if PIVOT:
        p_vals = tl.where(rows[None, :] == perm[:, None], 1.0, 0.0)
        p_mask = (rows[:, None] < M) & (rows[None, :] < M)
        tl.store(
            P + pid * M * M + rows[:, None] * M + rows[None, :],
            p_vals,
            mask=p_mask,
        )


def _linalg_lu_fused_impl(input, *, pivot=True, P=None, L=None, U=None):
    """Fused single-kernel path: factorization + L/U/P extraction in one launch."""
    batch_shape = input.shape[:-2]
    m, n = input.shape[-2], input.shape[-1]
    k = min(m, n)
    batch = input.numel() // (m * n)

    if L is None:
        L = torch.empty((*batch_shape, m, k), device=input.device, dtype=input.dtype)
    else:
        L.resize_((*batch_shape, m, k))
    if U is None:
        U = torch.empty((*batch_shape, k, n), device=input.device, dtype=input.dtype)
    else:
        U.resize_((*batch_shape, k, n))
    if pivot:
        if P is None:
            P = torch.empty(
                (*batch_shape, m, m), device=input.device, dtype=input.dtype
            )
        else:
            P.resize_((*batch_shape, m, m))
    else:
        if P is None:
            P = torch.empty(0, device=input.device, dtype=input.dtype)
        else:
            P.resize_((0,))

    # The kernel never touches P when PIVOT is False (compile-time branch),
    # so a dummy pointer is passed for the empty P tensor.
    p_arg = P if pivot else L

    with torch_device_fn.device(input.device):
        _linalg_lu_fused_kernel[(batch,)](
            input,
            p_arg,
            L,
            U,
            m,
            n,
            k,
            triton.next_power_of_2(m),
            triton.next_power_of_2(n),
            pivot,
            num_warps=4,
        )
    return LinalgLUResult(P, L, U)


# ---------------------------------------------------------------------------
# Zero-copy unpack into pre-sized out tensors — used by the blocked path.
#
# Mirrors the kernel launches of flag_gems.ops.lu_unpack.lu_unpack, but
# writes directly into the provided P, L, U tensors, so the out variant
# skips the intermediate allocations and device-to-device copies.
# ---------------------------------------------------------------------------


def _lu_unpack_into(lu, pivots, P, L, U, pivot):
    batch_dims = lu.shape[:-2]
    m, n = lu.shape[-2], lu.shape[-1]
    k = min(m, n)

    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim

    pivots_shape = pivots.shape
    pivots_stride_b = pivots.stride(-2) if len(pivots_shape) > 1 else 0
    pivots_stride_k = pivots.stride(-1) if len(pivots_shape) > 0 else 0
    lu_stride_b = lu.stride(-3) if len(lu.shape) > 2 else 0
    l_stride_b = L.stride(-3) if len(batch_dims) > 0 else 0
    u_stride_b = U.stride(-3) if len(batch_dims) > 0 else 0

    with torch_device_fn.device(lu.device):
        if pivot:
            # P is pre-zeroed by the caller; only the 1s are scattered.
            p_stride_b = P.stride(-3) if len(batch_dims) > 0 else 0
            if m <= 512:
                lu_unpack_p_kernel_small[(batch_size,)](
                    pivots,
                    P,
                    m,
                    k,
                    pivots_stride_b,
                    pivots_stride_k,
                    p_stride_b,
                    P.stride(-2),
                    P.stride(-1),
                    triton.next_power_of_2(m),
                )
            else:
                lu_unpack_p_kernel_large[(batch_size * m,)](
                    pivots,
                    P,
                    m,
                    k,
                    pivots_stride_b,
                    pivots_stride_k,
                    p_stride_b,
                    P.stride(-2),
                    P.stride(-1),
                    1,
                )

        BLOCK_K = triton.next_power_of_2(k)
        if BLOCK_K > 1024:
            BLOCK_K = 1024
        lu_unpack_l_kernel[(batch_size * m,)](
            lu,
            L,
            m,
            n,
            k,
            lu_stride_b,
            lu.stride(-2),
            lu.stride(-1),
            l_stride_b,
            L.stride(-2),
            L.stride(-1),
            BLOCK_K,
        )

        BLOCK_N = triton.next_power_of_2(n)
        if BLOCK_N > 1024:
            BLOCK_N = 1024
        lu_unpack_u_kernel[(batch_size * k,)](
            lu,
            U,
            m,
            n,
            k,
            lu_stride_b,
            lu.stride(-2),
            lu.stride(-1),
            u_stride_b,
            U.stride(-2),
            U.stride(-1),
            BLOCK_N,
        )


# ---------------------------------------------------------------------------
# Internal implementation
#
# Two paths, both fully Triton (no torch compute, per the no-torch rule):
#
# 1) Fused path (m, n <= 64): one kernel factorizes A in registers and
#    writes L, U and P directly — no packed LU round-trip, no pivots
#    tensor, no separate unpack/zero kernels.
# 2) Blocked path (larger matrices / fp64): linalg_lu_factor for the
#    factorization followed by a zero-copy unpack into the outputs.
#
# For pivot=False, P is an empty (0,) tensor and A = L @ U, matching
# torch.linalg.lu semantics exactly.
# ---------------------------------------------------------------------------


def _linalg_lu_impl(input, *, pivot=True, P=None, L=None, U=None):
    input_contiguous = input.contiguous()
    batch_shape = input_contiguous.shape[:-2]
    m, n = input_contiguous.shape[-2], input_contiguous.shape[-1]
    k = min(m, n)

    if _can_use_fused_kernel(input_contiguous):
        return _linalg_lu_fused_impl(input_contiguous, pivot=pivot, P=P, L=L, U=U)

    # Blocked path: prepare the output tensors, then factorize and unpack.
    if pivot:
        if P is None:
            P = torch.zeros(
                (*batch_shape, m, m), device=input.device, dtype=input.dtype
            )
        else:
            P.resize_((*batch_shape, m, m))
            P.zero_()
    else:
        if P is None:
            P = torch.empty(0, device=input.device, dtype=input.dtype)
        else:
            P.resize_((0,))
    if L is None:
        L = torch.empty((*batch_shape, m, k), device=input.device, dtype=input.dtype)
    else:
        L.resize_((*batch_shape, m, k))
    if U is None:
        U = torch.empty((*batch_shape, k, n), device=input.device, dtype=input.dtype)
    else:
        U.resize_((*batch_shape, k, n))

    lu, pivots = linalg_lu_factor(input_contiguous, pivot=pivot)
    _lu_unpack_into(lu, pivots, P, L, U, pivot)

    return LinalgLUResult(P, L, U)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def linalg_lu(input, *, pivot=True):
    logger.debug("GEMS LINALG_LU")
    _linalg_lu_check(input, pivot)
    return _linalg_lu_impl(input, pivot=pivot)


def _resolve_linalg_lu_out_args(P, L, U, out):
    if out is not None:
        if P is not None or L is not None or U is not None:
            raise TypeError("linalg_lu(): out and P/L/U cannot both be set")
        if len(out) != 3:
            raise TypeError(
                "linalg_lu(): out must be a tuple of 3 tensors, " f"got {len(out)}"
            )
        return out
    if P is None or L is None or U is None:
        raise TypeError("linalg_lu(): P, L and U must all be provided for out variant")
    return P, L, U


def linalg_lu_out(input, *, pivot=True, P=None, L=None, U=None, out=None):
    logger.debug("GEMS LINALG_LU_OUT")
    _linalg_lu_check(input, pivot)
    p_out, l_out, u_out = _resolve_linalg_lu_out_args(P, L, U, out)
    return _linalg_lu_impl(input, pivot=pivot, P=p_out, L=l_out, U=u_out)
