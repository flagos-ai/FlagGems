import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_lu_factor import linalg_lu_factor, linalg_lu_factor_out
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

LinalgLUFactorExResult = namedtuple("LinalgLUFactorExResult", ["LU", "pivots", "info"])

_LU_FACTOR_BLOCK_MAX = 64


# ---------------------------------------------------------------------------
# Input validation — copied from linalg_lu_factor.py
# ---------------------------------------------------------------------------


def _linalg_lu_factor_check(input, pivot):
    if input.dim() < 2:
        raise RuntimeError(
            "torch.linalg.lu_factor: Expected input to have at least 2 dimensions, "
            f"got {input.dim()}"
        )
    if input.dtype not in (torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems linalg_lu_factor currently supports float32 and float64 only, "
            f"got {input.dtype}"
        )
    m, n = input.shape[-2], input.shape[-1]
    if m == 0 or n == 0:
        raise NotImplementedError(
            "FlagGems linalg_lu_factor currently does not support empty matrices"
        )
    if pivot not in (True, False):
        raise TypeError(f"pivot must be a bool, got {type(pivot)}")


def _can_use_fast_triton(input):
    m, n = input.shape[-2], input.shape[-1]
    # fp64 single-kernel path is slower than blocked for all sizes;
    # route fp64 through the optimized blocked path instead.
    if input.dtype == torch.float64:
        return False
    return m <= _LU_FACTOR_BLOCK_MAX and n <= _LU_FACTOR_BLOCK_MAX


# ---------------------------------------------------------------------------
# Fast-path kernel: LU factorization + info in a single kernel
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def _linalg_lu_factor_kernel_with_info(
    A,
    LU,
    PIVOTS,
    INFO,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PIVOT: tl.constexpr,
):
    """LU factorization that tracks the first zero/NaN pivot during the
    elimination loop.

    Optimizations over the baseline _linalg_lu_factor_kernel:
    - Column extraction uses direct axis=1 sum (no tl.trans).
    - Pivot value is indexed from the already-extracted column vector
      (O(K) reduction) instead of a nested O(M*N) reduction.
    - Column and row vectors are extracted once per iteration and
      reused for L scaling, pivot detection, and the trailing update.
    - The no-op U-row write-back is removed.
    """
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    offsets = pid * M * N + rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    work = tl.load(A + offsets, mask=mask, other=0.0).to(tl.float32)

    info_val = 0

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
            tl.store(PIVOTS + pid * K + j_ind, pivot_row + 1)

            # After swap, row j_ind == row_p (already extracted) — reuse.
            u_row = row_p

            # Update col_vals in-place: just swap elements at j_ind and pivot_row,
            # avoiding a full column re-extraction from work.
            old_j = tl.sum(tl.where(rows == j_ind, col_vals, 0.0), axis=0)
            old_p = tl.sum(tl.where(rows == pivot_row, col_vals, 0.0), axis=0)
            col_vals = tl.where(rows == j_ind, old_p, col_vals)
            col_vals = tl.where(rows == pivot_row, old_j, col_vals)
        else:
            tl.store(PIVOTS + pid * K + j_ind, j_ind + 1)
            col_vals = tl.sum(tl.where(cols[None, :] == j_ind, work, 0.0), axis=1)
            u_row = tl.sum(tl.where(rows[:, None] == j_ind, work, 0.0), axis=0)

        # Pivot is the diagonal element — index into the column vector.
        pivot = tl.sum(tl.where(rows == j_ind, col_vals, 0.0), axis=0)

        # Track first zero/NaN pivot.
        if info_val == 0:
            if pivot == 0.0 or pivot != pivot:
                info_val = j_ind + 1

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

    tl.store(LU + offsets, work, mask=mask)
    tl.store(INFO + pid, info_val)


# ---------------------------------------------------------------------------
# Diagonal scan kernel — for the blocked path, scans the diagonal of the
# LU factors to find the first zero or NaN.
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def _lu_factor_info_kernel(
    LU,
    INFO,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K
    diag = tl.load(LU + pid * M * N + offsets * (N + 1), mask=mask, other=1.0)

    sentinel = K + 1
    # Detect both exact zeros and NaN (NaN can arise from 0/0 division, e.g.
    # when the input matrix is all zeros).
    is_zero = (diag == 0) | (diag != diag)
    candidates = tl.where(is_zero & mask, offsets + 1, sentinel)
    first_zero = tl.min(candidates, axis=0)
    info = tl.where(first_zero == sentinel, 0, first_zero).to(tl.int32)
    tl.store(INFO + pid, info)


def _lu_factor_info(lu, out=None):
    """Compute the LAPACK-style info tensor by scanning the LU diagonal."""
    k = min(lu.shape[-2], lu.shape[-1])
    batch_shape = lu.shape[:-2]
    batch = lu.numel() // (lu.shape[-2] * lu.shape[-1])
    if out is None:
        info = torch.empty(batch_shape, device=lu.device, dtype=torch.int32)
    else:
        out.resize_(batch_shape)
        info = out

    with torch_device_fn.device(lu.device):
        _lu_factor_info_kernel[(batch,)](
            lu,
            info,
            lu.shape[-2],
            lu.shape[-1],
            k,
            triton.next_power_of_2(k),
            num_warps=4,
        )
    return info


# ---------------------------------------------------------------------------
# Internal implementation — accepts optional output tensors for the out
# variant to reuse memory, matching the pattern of _linalg_lu_factor_impl.
# ---------------------------------------------------------------------------


def _linalg_lu_factor_ex_impl(input, *, pivot=True, LU=None, pivots=None, info=None):
    _linalg_lu_factor_check(input, pivot)
    input_contiguous = input.contiguous()

    if _can_use_fast_triton(input_contiguous):
        # Fast path: kernel writes directly into provided (or newly allocated)
        # tensors — zero-copy for the out variant.
        batch_shape = input_contiguous.shape[:-2]
        m, n = input_contiguous.shape[-2], input_contiguous.shape[-1]
        k = min(m, n)
        batch = input_contiguous.numel() // (m * n)

        if LU is None:
            lu = torch.empty_like(input_contiguous)
        else:
            LU.resize_(input_contiguous.shape)
            lu = LU
        if pivots is None:
            pivots = torch.empty(
                (*batch_shape, k), device=input.device, dtype=torch.int32
            )
        else:
            pivots.resize_((*batch_shape, k))
        if info is None:
            _info = torch.empty(batch_shape, device=input.device, dtype=torch.int32)
        else:
            info.resize_(batch_shape)
            _info = info

        with torch_device_fn.device(input.device):
            _linalg_lu_factor_kernel_with_info[(batch,)](
                input_contiguous,
                lu,
                pivots,
                _info,
                m,
                n,
                k,
                triton.next_power_of_2(m),
                triton.next_power_of_2(n),
                pivot,
                num_warps=4,
            )
        return LinalgLUFactorExResult(lu, pivots, _info)
    else:
        # Blocked path: reuse out tensors via linalg_lu_factor_out when
        # available, falling back to linalg_lu_factor which allocates.
        if LU is not None and pivots is not None:
            lu, pivots = linalg_lu_factor_out(input, pivot=pivot, LU=LU, pivots=pivots)
        else:
            lu, pivots = linalg_lu_factor(input, pivot=pivot)
        _info = _lu_factor_info(lu, out=info)
        return LinalgLUFactorExResult(lu, pivots, _info)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _check_lu_factor_errors(info):
    failed = info != 0
    if not torch.any(failed).item():
        return

    info_cpu = info.detach().cpu().reshape(-1)
    first_info = int(info_cpu[info_cpu != 0][0].item())
    raise RuntimeError(
        "torch.linalg.lu_factor_ex: U[{},{}] is zero and using it on lu_solve "
        "would result in a division by zero. If you still want to perform the "
        "factorization, pass check_errors=False.".format(first_info, first_info)
    )


def _check_linalg_lu_factor_ex_args(pivot, check_errors):
    if pivot not in (True, False):
        raise TypeError(f"pivot must be a bool, got {type(pivot)}")
    if check_errors not in (True, False):
        raise TypeError(f"check_errors must be a bool, got {type(check_errors)}")


def linalg_lu_factor_ex(input, *, pivot=True, check_errors=False):
    logger.debug("GEMS LINALG_LU_FACTOR_EX")
    _check_linalg_lu_factor_ex_args(pivot, check_errors)

    res = _linalg_lu_factor_ex_impl(input, pivot=pivot)

    if check_errors:
        _check_lu_factor_errors(res.info)

    return res


def _resolve_linalg_lu_factor_ex_out_args(LU, pivots, info, out):
    if out is not None:
        if LU is not None or pivots is not None or info is not None:
            raise TypeError(
                "linalg_lu_factor_ex(): out and LU/pivots/info cannot both be set"
            )
        if len(out) != 3:
            raise TypeError(
                "linalg_lu_factor_ex(): out must be a tuple of 3 tensors, "
                f"got {len(out)}"
            )
        return out
    if LU is None or pivots is None or info is None:
        raise TypeError(
            "linalg_lu_factor_ex(): LU, pivots and info must all be provided "
            "for out variant"
        )
    return LU, pivots, info


def linalg_lu_factor_ex_out(
    input,
    *,
    pivot=True,
    check_errors=False,
    LU=None,
    pivots=None,
    info=None,
    out=None,
):
    logger.debug("GEMS LINALG_LU_FACTOR_EX.OUT")
    _check_linalg_lu_factor_ex_args(pivot, check_errors)
    lu_out, pivots_out, info_out = _resolve_linalg_lu_factor_ex_out_args(
        LU, pivots, info, out
    )

    res = _linalg_lu_factor_ex_impl(
        input, pivot=pivot, LU=lu_out, pivots=pivots_out, info=info_out
    )

    if check_errors:
        _check_lu_factor_errors(res.info)

    return res
