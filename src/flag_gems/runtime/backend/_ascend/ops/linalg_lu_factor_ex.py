import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .linalg_lu_factor import linalg_lu_factor, linalg_lu_factor_out

logger = logging.getLogger(__name__)

LinalgLUFactorExResult = namedtuple("LinalgLUFactorExResult", ["LU", "pivots", "info"])

_LU_FACTOR_BLOCK_MAX = (
    32  # simple fused kernel for m,n <= 32 (fast compile, tl.range loop)
)


# ---------------------------------------------------------------------------
# Input validation — adapted from linalg_lu_factor.py (Ascend-specific)
# ---------------------------------------------------------------------------


def _linalg_lu_factor_check(input, pivot):
    if input.dim() < 2:
        raise RuntimeError(
            "torch.linalg.lu_factor: Expected input to have at least 2 dimensions, "
            f"got {input.dim()}"
        )
    if input.dtype != torch.float32:
        raise NotImplementedError(
            "FlagGems linalg_lu_factor_ex currently supports float32 only, "
            f"got {input.dtype}"
        )
    m, n = input.shape[-2], input.shape[-1]
    if m == 0 or n == 0:
        raise NotImplementedError(
            "FlagGems linalg_lu_factor_ex currently does not support empty matrices"
        )
    if pivot not in (True, False):
        raise TypeError(f"pivot must be a bool, got {type(pivot)}")
    if not pivot and input.device.type != "cuda":
        raise NotImplementedError(
            "FlagGems linalg_lu_factor_ex: pivot=False is only supported on CUDA devices, "
            f"got device={input.device.type}"
        )


def _can_use_fast_triton(input):
    """Check if the fused single-kernel Triton path can be used.

    Requires 8 <= m,n <= _LU_FACTOR_BLOCK_MAX. Below 8, the Ascend
    compiler fails with 'strides must not be zero' for tiny block sizes.
    """
    m, n = input.shape[-2], input.shape[-1]
    return 8 <= m <= _LU_FACTOR_BLOCK_MAX and 8 <= n <= _LU_FACTOR_BLOCK_MAX


# ---------------------------------------------------------------------------
# Fused kernel: LU factorization + info in a single launch.
#
# Based on _linalg_lu_factor_kernel (already proven on Ascend), extended with
# a post-loop diagonal scan that computes the LAPACK-style info tensor.
#
# All conditionals on dynamic Triton tensors use tl.where / tl.min rather
# than Python "if", and all logical-OR uses Triton's | rather than Python
# "or", both of which are required for correct compilation on Ascend NPU.
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def _linalg_lu_factor_ex_kernel(
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
    """LU factorization + info tracking in one kernel launch.

    Performs the same factorization as _linalg_lu_factor_kernel, then
    extracts the diagonal of the LU matrix and scans for the first zero
    or NaN pivot using Triton-safe tl.where / tl.min patterns.
    """
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    offsets = pid * M * N + rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    work = tl.load(A + offsets, mask=mask, other=0.0).to(tl.float32)

    # ---- LU factorization loop (identical to _linalg_lu_factor_kernel) ----
    for j_ind in tl.range(0, K):
        if PIVOT:
            col_mask_j = (cols[None, :] == j_ind).to(tl.float32)
            col_j = tl.sum(work * col_mask_j, axis=1)
            abs_col = tl.abs(col_j)
            abs_col = tl.where(rows < j_ind, -1.0, abs_col)
            abs_col = tl.where(rows < M, abs_col, -1.0)
            pivot_val = tl.max(abs_col, axis=0)
            pivot_row = tl.min(tl.where(abs_col == pivot_val, rows, BLOCK_M), axis=0)

            row_mask_j = (rows[:, None] == j_ind).to(tl.float32)
            row_mask_p = (rows[:, None] == pivot_row).to(tl.float32)
            row_j = tl.sum(work * row_mask_j, axis=0)
            row_p = tl.sum(work * row_mask_p, axis=0)
            col_mask = cols[None, :] < N
            work = tl.where((rows[:, None] == j_ind) & col_mask, row_p, work)
            work = tl.where((rows[:, None] == pivot_row) & col_mask, row_j, work)
            tl.store(PIVOTS + pid * K + j_ind, pivot_row + 1)
        else:
            tl.store(PIVOTS + pid * K + j_ind, j_ind + 1)

        pivot_row_mask = (rows[:, None] == j_ind).to(tl.float32)
        pivot_col_mask = (cols[None, :] == j_ind).to(tl.float32)
        pivot_mask = pivot_row_mask * pivot_col_mask
        pivot = tl.sum(work * pivot_mask)

        pivot_row_vals = tl.sum(work * pivot_row_mask, axis=0)
        active_cols = cols > j_ind
        work = tl.where(
            (rows[:, None] == j_ind) & active_cols[None, :], pivot_row_vals, work
        )

        col_vals = tl.sum(work * pivot_col_mask, axis=1)
        multipliers = tl.where(rows > j_ind, col_vals / pivot, col_vals)
        work = tl.where(
            (rows[:, None] > j_ind) & (cols[None, :] == j_ind),
            multipliers[:, None],
            work,
        )

        l_col = tl.sum(work * pivot_col_mask, axis=1)
        u_row = tl.sum(work * pivot_row_mask, axis=0)
        update_mask = (rows[:, None] > j_ind) & (cols[None, :] > j_ind)
        work = tl.where(update_mask, work - l_col[:, None] * u_row[None, :], work)

    tl.store(LU + offsets, work, mask=mask)

    # ---- Post-loop: scan diagonal for first zero/NaN pivot (info tensor) ----
    # Build a 1-D vector of diagonal values work[i,i] for 0 <= i < K by
    # constructing a 2-D diagonal mask and reducing along axis=1.  This
    # matches the pivot-extraction pattern already proven in
    # _linalg_lu_factor_kernel (rows[:, None] == scalar / cols[None, :] ==
    # scalar → mask → tl.sum).  The 1-D vector is then processed with the
    # same tl.where / tl.min / sentinel pattern used in the verified
    # _lu_factor_info_kernel.  The entire scan is a constant number of
    # Triton ops — NO tl.range loop, NO conditional SSA assignment — so
    # Ascend's compiler sees purely static control flow.
    sentinel = K + 1

    # 1. Diagonal mask: (BLOCK_M, BLOCK_N) with 1.0 on the diagonal, 0 elsewhere.
    diag_full_mask = (rows[:, None] == cols[None, :]).to(tl.float32)
    # Restrict to rows within the K x K leading submatrix (in-kernel, K <=
    # BLOCK_M and K <= BLOCK_N because BLOCK_M/N are next_power_of_2).
    row_valid = (rows[:, None] < K).to(tl.float32)  # (BLOCK_M, 1)
    diag_full_mask = diag_full_mask * row_valid  # broadcast along cols

    # 2. Extract diagonal into a 1-D vector (BLOCK_M,).
    diag_vals = tl.sum(work * diag_full_mask, axis=1)

    # 3. Apply the same sentinel + tl.where + tl.min pattern as
    #    _lu_factor_info_kernel (proven to compile on Ascend).
    is_bad = (diag_vals == 0.0) | (diag_vals != diag_vals)  # (BLOCK_M,)
    candidates = tl.where(is_bad & (rows < K), rows + 1, sentinel)  # (BLOCK_M,)
    first_bad = tl.min(candidates, axis=0)  # scalar
    info_val = tl.where(first_bad == sentinel, 0, first_bad).to(tl.int32)
    tl.store(INFO + pid, info_val)


# ---------------------------------------------------------------------------
# Diagonal scan kernel — used by the blocked-path fallback.
# Scans the diagonal of the already-computed LU factors.
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
    """Scan the diagonal of LU to find the first zero or NaN pivot position.

    Returns 0 if all diagonal elements are non-zero and finite, otherwise
    returns the 1-indexed position of the first zero/NaN pivot.
    """
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K
    diag = tl.load(LU + pid * M * N + offsets * (N + 1), mask=mask, other=1.0)

    sentinel = K + 1
    # NOTE: use | (Triton element-wise OR), NOT Python "or" — Ascend compiler
    # rejects Python "or" on Triton tensors.
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
# Internal implementation
#
# Two paths:
# 1) Fast path (m,n <= 32): single fused kernel _linalg_lu_factor_ex_kernel
#    does LU factorization + info in one launch — no second kernel overhead.
# 2) Blocked path: delegates to ascend-optimized linalg_lu_factor, then
#    scans the diagonal with _lu_factor_info_kernel.
# ---------------------------------------------------------------------------


def _linalg_lu_factor_ex_impl(input, *, pivot=True, LU=None, pivots=None, info=None):
    _linalg_lu_factor_check(input, pivot)
    input_contiguous = input.contiguous()

    if _can_use_fast_triton(input_contiguous):
        # Fast path: single fused kernel.
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
            _linalg_lu_factor_ex_kernel[(batch,)](
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
            )
        return LinalgLUFactorExResult(lu, pivots, _info)
    else:
        # Blocked path: factorization + separate info scan.
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
    """Raise RuntimeError if any batch element has a zero/NaN pivot."""
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
    logger.debug("GEMS_ASCEND LINALG_LU_FACTOR_EX")
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
    logger.debug("GEMS_ASCEND LINALG_LU_FACTOR_EX_OUT")
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
