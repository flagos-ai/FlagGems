import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .linalg_lu_factor import linalg_lu_factor

logger = logging.getLogger(__name__)

LinalgLUResult = namedtuple("LinalgLUResult", ["P", "L", "U"])

# --- Local copies of the lu_unpack kernels from main ops (flag_gems.ops.lu_unpack) ---
# Defined here rather than imported to ensure fresh Ascend compilation
# (imported kernels may use cached GPU binaries that produce incorrect results).

# Tile sizes are kept small (32 x 64) so the 2D comparison / mask / data
# buffers of the extraction kernels stay well inside the Ascend UB limit
# (a 64 x 128 tile overflows the 192KB UB once the auto-multi-buffer pass
# duplicates the buffers).
_LU_UNPACK_TILE_M = 32  # row tile size for L/U extraction
_LU_UNPACK_TILE_K = 64  # col tile size for L extraction
_LU_UNPACK_TILE_N = 64  # col tile size for U extraction


# ---------------------------------------------------------------------------
# Input validation — adapted from linalg_lu_factor.py (Ascend-specific)
# ---------------------------------------------------------------------------


def _linalg_lu_check(input, pivot):
    if input.dim() < 2:
        raise RuntimeError(
            "torch.linalg.lu: Expected input to have at least 2 dimensions, "
            f"got {input.dim()}"
        )
    if input.dtype != torch.float32:
        raise NotImplementedError(
            "FlagGems linalg_lu currently supports float32 only, " f"got {input.dtype}"
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


# ---------------------------------------------------------------------------
# Unpack kernels: (LU, pivots) -> (P, L, U).
#
# All conditionals on dynamic Triton tensors use tl.where rather than Python
# "if", which is required for correct compilation on Ascend NPU.
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def _lu_unpack_p_kernel(
    PIVOTS,
    P,
    m,
    k,
    BLOCK_M: tl.constexpr,
):
    """Build the permutation matrix P from the pivot sequence.

    One program per batch element.  perm[j] tracks the final position of
    original row j (forward map).  Applying the pivot swaps in forward
    order (i = 0 .. k-1) to this position map yields exactly
    torch.linalg.lu's P when the 1s are scattered at P[j, perm[j]] — the
    same matrix as applying the row swaps to the identity in reverse order
    (verified against CPU torch.linalg.lu).

    Each swap is expressed as an arithmetic delta instead of a nested
    tl.where chain: the nested tl.where formulation miscompiles on Ascend
    (it produces duplicate positions, e.g. [0, 2, 3, 3] for pivots
    [1, 3, 4, 4], which is not a permutation), while the delta form
    perm[j] += d*(perm[j]==i) - d*(perm[j]==pivot_idx) with d=pivot_idx-i
    computes the swap correctly.
    P is pre-zeroed by the caller; only the 1.0 entries are scattered.
    """
    pid = tl.program_id(0)
    row_ids = tl.arange(0, BLOCK_M)
    perm = row_ids
    mask = row_ids < m

    for i in range(k):
        pivot_val = tl.load(PIVOTS + pid * k + i)
        pivot_idx = pivot_val - 1
        d = pivot_idx - i
        perm = (
            perm + d * (perm == i).to(tl.int32) - d * (perm == pivot_idx).to(tl.int32)
        )

    offsets = pid * m * m + row_ids * m + perm
    tl.store(P + offsets, tl.full([BLOCK_M], 1.0, tl.float32), mask=mask)


@libentry()
@triton.jit
def _lu_unpack_l_kernel(
    LU,
    L,
    m,
    n,
    k,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Extract L (unit lower triangular) from the packed LU factors.

    L[i, j] = LU[i, j] for j < i, 1.0 for j == i, 0.0 for j > i.
    Grid: (cdiv(m, BLOCK_M), cdiv(k, BLOCK_K), batch).
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_b = tl.program_id(2)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask = (rows[:, None] < m) & (cols[None, :] < k)
    vals = tl.load(
        LU + pid_b * m * n + rows[:, None] * n + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    out = tl.where(
        rows[:, None] > cols[None, :],
        vals,
        tl.where(rows[:, None] == cols[None, :], 1.0, 0.0),
    )
    tl.store(L + pid_b * m * k + rows[:, None] * k + cols[None, :], out, mask=mask)


@libentry()
@triton.jit
def _lu_unpack_u_kernel(
    LU,
    U,
    m,
    n,
    k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Extract U (upper triangular) from the packed LU factors.

    U[i, j] = LU[i, j] for j >= i, 0.0 for j < i.
    Grid: (cdiv(k, BLOCK_M), cdiv(n, BLOCK_N), batch).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (rows[:, None] < k) & (cols[None, :] < n)
    vals = tl.load(
        LU + pid_b * m * n + rows[:, None] * n + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    out = tl.where(cols[None, :] >= rows[:, None], vals, 0.0)
    tl.store(U + pid_b * k * n + rows[:, None] * n + cols[None, :], out, mask=mask)


# ---------------------------------------------------------------------------
# Zero-copy unpack into pre-sized out tensors.
#
# Mirrors the kernel launches of flag_gems.ops.lu_unpack.lu_unpack, but
# writes directly into the provided P, L, U tensors, so the out variant
# skips the intermediate allocations and device-to-device copies.
# ---------------------------------------------------------------------------


def _lu_unpack_into(lu, pivots, P, L, U, pivot):
    batch_shape = lu.shape[:-2]
    m, n = lu.shape[-2], lu.shape[-1]
    k = min(m, n)

    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim

    with torch_device_fn.device(lu.device):
        if pivot:
            _lu_unpack_p_kernel[(batch_size,)](
                pivots,
                P,
                m,
                k,
                triton.next_power_of_2(m),
                num_warps=4,
            )

        _lu_unpack_l_kernel[
            (
                triton.cdiv(m, _LU_UNPACK_TILE_M),
                triton.cdiv(k, _LU_UNPACK_TILE_K),
                batch_size,
            )
        ](lu, L, m, n, k, _LU_UNPACK_TILE_M, _LU_UNPACK_TILE_K, num_warps=4)

        _lu_unpack_u_kernel[
            (
                triton.cdiv(k, _LU_UNPACK_TILE_M),
                triton.cdiv(n, _LU_UNPACK_TILE_N),
                batch_size,
            )
        ](lu, U, m, n, k, _LU_UNPACK_TILE_M, _LU_UNPACK_TILE_N, num_warps=4)


# ---------------------------------------------------------------------------
# Internal implementation
#
# The whole computation runs through Ascend Triton kernels, per the
# no-torch rule: the factorization is done by the Ascend linalg_lu_factor
# (fast fused kernel for m,n <= 32, blocked path otherwise) and the packed
# LU factors are unpacked into (P, L, U) by the local Triton kernels above,
# reproducing torch.linalg.lu semantics: P @ L @ U = A with
# P of shape (..., m, m) for pivot=True.
# ---------------------------------------------------------------------------


def _linalg_lu_impl(input, *, pivot=True):
    lu, pivots = linalg_lu_factor(input, pivot=pivot)

    batch_shape = input.shape[:-2]
    m, n = input.shape[-2], input.shape[-1]
    k = min(m, n)

    if pivot:
        P = torch.zeros((*batch_shape, m, m), device=input.device, dtype=input.dtype)
    else:
        P = torch.empty(0, device=input.device, dtype=input.dtype)
    L = torch.empty((*batch_shape, m, k), device=input.device, dtype=input.dtype)
    U = torch.empty((*batch_shape, k, n), device=input.device, dtype=input.dtype)

    _lu_unpack_into(lu, pivots, P, L, U, pivot)
    return LinalgLUResult(P, L, U)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def linalg_lu(input, *, pivot=True):
    logger.debug("GEMS_ASCEND LINALG_LU")
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
    logger.debug("GEMS_ASCEND LINALG_LU_OUT")
    _linalg_lu_check(input, pivot)
    p_out, l_out, u_out = _resolve_linalg_lu_out_args(P, L, U, out)

    batch_shape = input.shape[:-2]
    m, n = input.shape[-2], input.shape[-1]
    k = min(m, n)

    # Resize the provided outputs to the expected shapes.  For pivot=False,
    # P is resized to the empty (0,) tensor, matching torch's out-variant
    # behavior.
    if pivot:
        p_out.resize_((*batch_shape, m, m))
        p_out.zero_()
    else:
        p_out.resize_((0,))
    l_out.resize_((*batch_shape, m, k))
    u_out.resize_((*batch_shape, k, n))

    lu, pivots = linalg_lu_factor(input, pivot=pivot)
    _lu_unpack_into(lu, pivots, p_out, l_out, u_out, pivot)

    return LinalgLUResult(p_out, l_out, u_out)
