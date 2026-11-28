import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_lu_factor import (
    linalg_lu_factor_out as _generic_lu_factor_out,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

LinalgLUFactorResult = namedtuple("LinalgLUFactorResult", ["LU", "pivots"])

# ---------------------------------------------------------------------------
# GCU400 (enflame) specialization of linalg_lu_factor.
#
# The generic implementation was copied here verbatim and fails the batched
# benchmark shapes with ``OutOfResources: out of resource: grid.y, Required:
# 1024, Hardware limit: 255``: its blocked path launches the swap/solve
# kernels with grid=(tiles, batch), and the GCU launcher caps grid.y/grid.z
# at 255 (grid.x at 65535), so batch > 255 is rejected at launch.
#
# This file replaces the dispatch with the scheme proven on the sibling op
# linalg_lu_factor_ex (job j20261117040833_6c6242f9: bench-core arithmean
# 1.666x vs torch native, pytest 46/46, bitwise-golden vs the swap kernel),
# specialized for the two-tensor result of lu_factor:
#
#   * fast path (fp32, m, n <= 128): one swap-free kernel per batch element.
#     The LAPACK row interchanges are tracked in a ``pos`` vector (logical
#     position of each original row) instead of physically swapping tile
#     rows.  The pivot search takes the lowest *logical position* among the
#     maxima (idamax tie-break), IPIV(k) is the logical position the pivot
#     row came from, and the final tile is scattered to output rows by the
#     final ``pos``, so torch.lu_unpack(LU, pivots) reconstructs P @ L @ U
#     exactly.  The pivot row and the row currently at logical position k
#     are extracted with one packed sum (16-bit lanes) instead of two
#     reduction trees; pivots accumulate in registers and store once.
#     ``tl.range`` (not ``static_range``): the fully unrolled variant blows
#     the GCU400 DSM budget at num_warps=4 (1.27MB vs 917KB) and is slower
#     even at num_warps=1.  num_warps: 1 below 16K-element tiles (the
#     cross-lane trees dominate, ~1.4x faster), 4 for 128x128-class tiles
#     (elementwise tile selects dominate).
#
#   * panel cascade (fp32, min(m, n) > 128, m <= 512): the swap-free panel
#     kernel, offset into the active submatrix, on flat 1-D grids
#     (tiles * batch <= 65535), which is what removes the grid.y limit.
#     On the current build (flagtree d8ca7a9e / Triton GCU 4.1.6) the
#     swap-free kernel silently corrupts when its row tile exceeds 256
#     (BLOCK_M=512: pivots and LU wrong with no error, num_warps- and
#     PANEL-width-independent), so tall panels (> 256 remaining rows) run
#     the generic physical-swap panel kernel instead -- verbatim, proven
#     correct at BLOCK_M=512 on this build -- at panel width 32, because
#     its 512x64 tile exceeds the DSM budget (OutOfResources 1286144 >
#     917504) while 512x32 is measured to fit.  m > 512 stays generic.
#     The same corruption makes the sibling linalg_lu_factor_ex cascade
#     fail its own (512,512) pytest today; that op is outside this file.
#
#   * everything else (fp64, m > 512, grids that do not fit): the generic
#     blocked path driven per <= 255-batch chunk, keeping every generic
#     launch inside the grid.y limit (P19 bypass); chunks are host-side
#     views only, so no extra kernels are introduced.
# ---------------------------------------------------------------------------

_LU_FACTOR_BLOCK_MAX = 128
_FAST_TILE_NW_AREA = 16384

# GCU400 hardware grid limits (triton.backends.enflame GcuLauncher):
# grid.x <= 65535, grid.y <= 255, grid.z <= 255.
_GCU_GRID_X_MAX = 65535
_GCU_GRID_Y_MAX = 255

_LU_FACTOR_PANEL = 64
_LU_FACTOR_TILE_M = 64
_LU_FACTOR_TILE_N = 128
_LU_FACTOR_PANEL_MAX_ROWS = 512

# Row budget of the swap-free panel kernel on the current build: 512-row
# pos-tracking tiles silently corrupt (see the note in the module header),
# 256-row tiles are proven.  Taller panels run the physical-swap kernel.
_SWAP_FREE_PANEL_MAX_ROWS = 256
# Panel width for the physical-swap branch: its 512x64 tile exceeds the DSM
# budget (OutOfResources 1286144 > 917504), 512x32 is measured to fit.
_SWAP_PANEL_WIDTH = 32


@libentry()
@triton.jit
def _lu_factor_panel_swap_kernel(
    LU,
    PIVOTS,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PANEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_B: tl.constexpr,
    LEFT_BLOCK_N: tl.constexpr,
    APPLY_LEFT: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    bcols = tl.arange(0, BLOCK_B)
    cols = K0 + bcols
    left_cols = tl.arange(0, LEFT_BLOCK_N)

    offsets = pid * M * N + rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (bcols[None, :] < PANEL)
    panel = tl.load(LU + offsets, mask=mask, other=0.0).to(tl.float32)

    for jj in tl.range(0, PANEL):
        j = K0 + jj

        # pivot search using DTYPE-aligned comparisons
        col_vals = tl.sum(tl.where(bcols[:, None] == jj, tl.trans(panel), 0.0), axis=0)
        abs_col = tl.abs(col_vals)
        abs_col = tl.where(rows < j, -1.0, abs_col)
        abs_col = tl.where(rows < M, abs_col, -1.0)
        pivot_val = tl.max(abs_col, axis=0)
        pivot_row = tl.min(tl.where(abs_col == pivot_val, rows, BLOCK_M), axis=0)

        # swap rows in panel
        row_j = tl.sum(tl.where(rows[:, None] == j, panel, 0.0), axis=0)
        row_p = tl.sum(tl.where(rows[:, None] == pivot_row, panel, 0.0), axis=0)
        panel = tl.where((rows[:, None] == j) & mask, row_p[None, :], panel)
        panel = tl.where((rows[:, None] == pivot_row) & mask, row_j[None, :], panel)
        tl.store(PIVOTS + pid * K + j, pivot_row + 1)

        if APPLY_LEFT:
            left_mask = left_cols < K0
            row_j_left_offsets = pid * M * N + j * N + left_cols
            row_p_left_offsets = pid * M * N + pivot_row * N + left_cols
            row_j_left = tl.load(LU + row_j_left_offsets, mask=left_mask, other=0.0).to(
                tl.float32
            )
            row_p_left = tl.load(LU + row_p_left_offsets, mask=left_mask, other=0.0).to(
                tl.float32
            )
            tl.store(LU + row_j_left_offsets, row_p_left, mask=left_mask)
            tl.store(LU + row_p_left_offsets, row_j_left, mask=left_mask)

        # pivot value after swap
        pivot = tl.sum(
            tl.sum(
                tl.where((rows[:, None] == j) & (bcols[None, :] == jj), panel, 0.0),
                axis=0,
            ),
            axis=0,
        )

        # scale column below diagonal
        col_vals = tl.sum(tl.where(bcols[:, None] == jj, tl.trans(panel), 0.0), axis=0)
        safe = tl.where(pivot != 0.0, pivot, 1.0)
        scaled = col_vals / safe
        scaled = tl.where(pivot != 0.0, scaled, 0.0)
        col_vals = tl.where(rows > j, scaled, col_vals)
        panel = tl.where(
            (rows[:, None] > j) & (bcols[None, :] == jj),
            col_vals[:, None],
            panel,
        )

        # rank-1 update on trailing sub-panel
        l_col = tl.sum(tl.where(bcols[:, None] == jj, tl.trans(panel), 0.0), axis=0)
        u_row = tl.sum(tl.where(rows[:, None] == j, panel, 0.0), axis=0)
        update_mask = (rows[:, None] > j) & (bcols[None, :] > jj)
        panel = tl.where(update_mask, panel - l_col[:, None] * u_row[None, :], panel)

    tl.store(LU + offsets, panel, mask=mask)


@libentry()
@triton.jit
def _lu_factor_panel_no_pivot_swap_kernel(
    LU,
    PIVOTS,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PANEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    # Physical-swap (identity-pivot) panel kernel, verbatim from the generic
    # implementation, for the pivot=False branch of tall cascades.
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    bcols = tl.arange(0, BLOCK_B)
    cols = K0 + bcols

    offsets = pid * M * N + rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (bcols[None, :] < PANEL)
    panel = tl.load(LU + offsets, mask=mask, other=0.0).to(tl.float32)

    for jj in tl.range(0, PANEL):
        j = K0 + jj
        pivot = tl.sum(
            tl.sum(
                tl.where((rows[:, None] == j) & (bcols[None, :] == jj), panel, 0.0),
                axis=0,
            ),
            axis=0,
        )
        col_vals = tl.sum(tl.where(bcols[:, None] == jj, tl.trans(panel), 0.0), axis=0)
        safe = tl.where(pivot != 0.0, pivot, 1.0)
        scaled = col_vals / safe
        scaled = tl.where(pivot != 0.0, scaled, 0.0)
        col_vals = tl.where(rows > j, scaled, col_vals)
        panel = tl.where(
            (rows[:, None] > j) & (bcols[None, :] == jj),
            col_vals[:, None],
            panel,
        )

        l_col = tl.sum(tl.where(bcols[:, None] == jj, tl.trans(panel), 0.0), axis=0)
        u_row = tl.sum(tl.where(rows[:, None] == j, panel, 0.0), axis=0)
        update_mask = (rows[:, None] > j) & (bcols[None, :] > jj)
        panel = tl.where(update_mask, panel - l_col[:, None] * u_row[None, :], panel)
        tl.store(PIVOTS + pid * K + j, j + 1)

    tl.store(LU + offsets, panel, mask=mask)


@libentry()
@triton.jit
def _linalg_lu_factor_kernel(
    A,
    LU,
    PIVOTS,
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
    r2 = rows[:, None]
    c2 = cols[None, :]
    mask = (r2 < M) & (c2 < N)
    offs = pid * M * N + r2 * N + c2
    work = tl.load(A + offs, mask=mask, other=0.0).to(tl.float32)

    pos = rows.to(tl.int32)
    active = rows < M
    piv_acc = tl.zeros((BLOCK_M,), tl.int32)

    for k in tl.range(0, K):
        col_vals = tl.sum(tl.where(c2 == k, work, 0.0), axis=1)
        if PIVOT:
            abs_col = tl.abs(col_vals)
            search = tl.where(active, abs_col, -1.0)
            search = tl.where(rows < M, search, -1.0)
            pivot_val = tl.max(search, axis=0)
            cand = search == pivot_val
            p_pos = tl.min(tl.where(cand, pos, BLOCK_M), axis=0)
            pk = tl.sum(
                tl.where(pos == p_pos, rows, 0) + 65536 * tl.where(pos == k, rows, 0),
                axis=0,
            )
            pivot_row = pk % 65536
            q_row = pk // 65536
            pos = tl.where(rows == pivot_row, k, pos)
            pos = tl.where(rows == q_row, p_pos, pos)
            active = tl.where(rows == pivot_row, False, active)
            piv_acc = tl.where(rows == k, p_pos + 1, piv_acc)
        else:
            pivot_row = k
            piv_acc = tl.where(rows == k, k + 1, piv_acc)
        pivot = tl.sum(tl.where(rows == pivot_row, col_vals, 0.0), axis=0)
        u_row = tl.sum(tl.where(r2 == pivot_row, work, 0.0), axis=0)
        # A zero pivot yields zero multipliers (guarded division), matching
        # ATen: the factor stays finite; a plain select would compute 0/0
        # and poison the trailing submatrix with NaN.
        safe = tl.where(pivot != 0.0, pivot, 1.0)
        scaled = col_vals / safe
        scaled = tl.where(pivot != 0.0, scaled, 0.0)
        if PIVOT:
            act2 = active & (rows != pivot_row)
            wb = act2[:, None] & (c2 == k)
            work = tl.where(wb, scaled[:, None], work)
            upd = act2[:, None] & (c2 > k)
        else:
            wb = (r2 > k) & (c2 == k)
            work = tl.where(wb, scaled[:, None], work)
            upd = (r2 > k) & (c2 > k)
        work = tl.where(upd, work - scaled[:, None] * u_row[None, :], work)

    out_offs = pid * M * N + pos[:, None] * N + c2
    tl.store(LU + out_offs, work, mask=mask)
    kk = tl.arange(0, BLOCK_M)
    tl.store(PIVOTS + pid * K + kk, piv_acc, mask=kk < K)


@libentry()
@triton.jit
def _lu_factor_panel_kernel(
    LU,
    PIVOTS,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PANEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_B: tl.constexpr,
    PIVOT: tl.constexpr,
):
    # Swap-free panel factorization on rows [K0, M), columns [K0, K0+PANEL):
    # the same pos-tracking scheme as the fast path, in panel-local
    # coordinates.  Pivots store the 1-based *global* row index of the
    # interchange partner (LAPACK IPIV convention), so the copied downstream
    # kernels replay them unchanged.
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_B)
    r2 = rows[:, None]
    c2 = cols[None, :]
    m_loc = M - K0
    mask = (r2 < m_loc) & (c2 < PANEL)
    offs = pid * M * N + (K0 + r2) * N + (K0 + c2)
    work = tl.load(LU + offs, mask=mask, other=0.0).to(tl.float32)

    pos = rows.to(tl.int32)
    active = rows < m_loc
    piv_acc = tl.zeros((BLOCK_M,), tl.int32)

    for k in tl.range(0, PANEL):
        col_vals = tl.sum(tl.where(c2 == k, work, 0.0), axis=1)
        if PIVOT:
            abs_col = tl.abs(col_vals)
            search = tl.where(active, abs_col, -1.0)
            search = tl.where(rows < m_loc, search, -1.0)
            pivot_val = tl.max(search, axis=0)
            cand = search == pivot_val
            p_pos = tl.min(tl.where(cand, pos, BLOCK_M), axis=0)
            pk = tl.sum(
                tl.where(pos == p_pos, rows, 0) + 65536 * tl.where(pos == k, rows, 0),
                axis=0,
            )
            pivot_row = pk % 65536
            q_row = pk // 65536
            pos = tl.where(rows == pivot_row, k, pos)
            pos = tl.where(rows == q_row, p_pos, pos)
            active = tl.where(rows == pivot_row, False, active)
            piv_acc = tl.where(rows == k, K0 + p_pos + 1, piv_acc)
        else:
            pivot_row = k
            piv_acc = tl.where(rows == k, K0 + k + 1, piv_acc)
        pivot = tl.sum(tl.where(rows == pivot_row, col_vals, 0.0), axis=0)
        u_row = tl.sum(tl.where(r2 == pivot_row, work, 0.0), axis=0)
        safe = tl.where(pivot != 0.0, pivot, 1.0)
        scaled = col_vals / safe
        scaled = tl.where(pivot != 0.0, scaled, 0.0)
        if PIVOT:
            act2 = active & (rows != pivot_row)
            wb = act2[:, None] & (c2 == k)
            work = tl.where(wb, scaled[:, None], work)
            upd = act2[:, None] & (c2 > k)
        else:
            wb = (r2 > k) & (c2 == k)
            work = tl.where(wb, scaled[:, None], work)
            upd = (r2 > k) & (c2 > k)
        work = tl.where(upd, work - scaled[:, None] * u_row[None, :], work)

    out_offs = pid * M * N + (K0 + pos[:, None]) * N + (K0 + c2)
    tl.store(LU + out_offs, work, mask=mask)
    kk = tl.arange(0, BLOCK_M)
    tl.store(PIVOTS + pid * K + K0 + kk, piv_acc, mask=kk < PANEL)


@libentry()
@triton.jit
def _lu_factor_swap_right_and_solve_kernel(
    LU,
    PIVOTS,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PANEL: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES_N: tl.constexpr,
):
    """Flat-grid variant of the generic swap-right-and-solve kernel."""
    pid = tl.program_id(0)
    pid_n = pid % TILES_N
    pid_b = pid // TILES_N
    brows = tl.arange(0, BLOCK_B)
    cols = K0 + PANEL + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = K0 + brows

    offsets = pid_b * M * N + rows[:, None] * N + cols[None, :]
    mask = (brows[:, None] < PANEL) & (cols[None, :] < N)
    vals = tl.load(LU + offsets, mask=mask, other=0.0).to(tl.float32)

    col_mask = cols[None, :] < N
    for jj in tl.range(0, PANEL):
        j = K0 + jj
        pivot_row = tl.load(PIVOTS + pid_b * K + j) - 1
        row_j = tl.sum(tl.where(brows[:, None] == jj, vals, 0.0), axis=0)

        row_p_offsets = pid_b * M * N + pivot_row * N + cols
        row_p = tl.load(LU + row_p_offsets, mask=cols < N, other=0.0).to(tl.float32)

        vals = tl.where((brows[:, None] == jj) & col_mask, row_p[None, :], vals)

        rel_pivot = pivot_row - K0
        vals = tl.where((brows[:, None] == rel_pivot) & col_mask, row_j[None, :], vals)

        tl.store(LU + row_p_offsets, row_j, mask=cols < N)

    for jj in tl.range(0, PANEL):
        row_j = tl.sum(tl.where(brows[:, None] == jj, vals, 0.0), axis=0)

        l_col_offsets = pid_b * M * N + (K0 + brows) * N + (K0 + jj)
        l_col = tl.load(LU + l_col_offsets, mask=brows < PANEL, other=0.0).to(
            tl.float32
        )
        l_col = tl.where(brows <= jj, 0.0, l_col)

        vals = tl.where(
            brows[:, None] > jj,
            vals - l_col[:, None] * row_j[None, :],
            vals,
        )

    tl.store(LU + offsets, vals, mask=mask)


@libentry()
@triton.jit
def _lu_factor_trailing_update_kernel(
    LU,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    PANEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_B: tl.constexpr,
    TILES_N,
    BATCH,
):
    # Flat 1-D grid: the GCU launcher caps grid.y and grid.z at 255, so a
    # 3-D (tiles_m, tiles_n, batch) launch would reject batch > 255.
    pid = tl.program_id(0)
    pid_m = pid // (TILES_N * BATCH)
    rem = pid % (TILES_N * BATCH)
    pid_n = rem // BATCH
    pid_b = rem % BATCH

    rows = K0 + PANEL + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = K0 + PANEL + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bidx = tl.arange(0, BLOCK_B)

    tile_offsets = pid_b * M * N + rows[:, None] * N + cols[None, :]
    tile_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tile = tl.load(LU + tile_offsets, mask=tile_mask, other=0.0).to(tl.float32)

    l_offsets = pid_b * M * N + rows[:, None] * N + (K0 + bidx[None, :])
    u_offsets = pid_b * M * N + (K0 + bidx[:, None]) * N + cols[None, :]
    l_mask = (rows[:, None] < M) & (bidx[None, :] < PANEL)
    u_mask = (bidx[:, None] < PANEL) & (cols[None, :] < N)
    # TLE load with .cg cache modifier — panel data shared across trailing tiles
    l_vals = tl.load(LU + l_offsets, mask=l_mask, other=0.0).to(tl.float32)
    u_vals = tl.load(LU + u_offsets, mask=u_mask, other=0.0).to(tl.float32)
    update = tl.dot(l_vals, u_vals, input_precision="ieee")

    tl.store(LU + tile_offsets, tile - update, mask=tile_mask)


@libentry()
@triton.jit
def _lu_factor_apply_panel_pivots_kernel(
    LU,
    PIVOTS,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PANEL: tl.constexpr,
    COL_START: tl.constexpr,
    NUM_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES_N: tl.constexpr,
):
    """Flat-grid variant of the generic apply-panel-pivots kernel."""
    pid = tl.program_id(0)
    pid_n = pid % TILES_N
    pid_b = pid // TILES_N
    cols = COL_START + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < COL_START + NUM_COLS

    for jj in tl.range(0, PANEL):
        j = K0 + jj
        pivot_row = tl.load(PIVOTS + pid_b * K + j) - 1
        row_j_offsets = pid_b * M * N + j * N + cols
        row_p_offsets = pid_b * M * N + pivot_row * N + cols
        row_j = tl.load(LU + row_j_offsets, mask=col_mask, other=0.0).to(tl.float32)
        row_p = tl.load(LU + row_p_offsets, mask=col_mask, other=0.0).to(tl.float32)
        tl.store(LU + row_j_offsets, row_p, mask=col_mask)
        tl.store(LU + row_p_offsets, row_j, mask=col_mask)


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
    # fp64 stays on the blocked path (generic behaviour); the swap-free tile
    # kernel is the fp32 win.  grid.x = batch must also fit the 65535 cap —
    # oversized batches fall through to the chunked blocked fallback.
    if input.dtype != torch.float32:
        return False
    m, n = input.shape[-2], input.shape[-1]
    if m > _LU_FACTOR_BLOCK_MAX or n > _LU_FACTOR_BLOCK_MAX:
        return False
    batch = input.numel() // (m * n)
    return batch <= _GCU_GRID_X_MAX


def _can_use_panel_cascade(input):
    # fp32, deep matrices (min(m, n) > 128), panels that fit the DSM budget
    # (m <= 512), and flat 1-D grids that fit grid.x (tiles * batch).
    if input.dtype != torch.float32:
        return False
    m, n = input.shape[-2], input.shape[-1]
    if m > _LU_FACTOR_PANEL_MAX_ROWS:
        return False
    if min(m, n) <= _LU_FACTOR_BLOCK_MAX:
        return False
    batch = input.numel() // (m * n)
    tiles_n = triton.cdiv(max(n - 1, 1), _LU_FACTOR_TILE_N)
    tiles_m = triton.cdiv(max(m - 1, 1), _LU_FACTOR_TILE_M)
    if tiles_n * batch > _GCU_GRID_X_MAX or tiles_m * tiles_n * batch > _GCU_GRID_X_MAX:
        return False
    if batch > _GCU_GRID_X_MAX:
        return False
    return True


def _lu_factor_cascade(input_contiguous, pivot, LU, pivots):
    """Panel-cascade factorization; honours LU/pivots in place."""
    m, n = input_contiguous.shape[-2], input_contiguous.shape[-1]
    k = min(m, n)
    batch = input_contiguous.numel() // (m * n)
    batch_shape = input_contiguous.shape[:-2]

    if LU is not None:
        LU.resize_(input_contiguous.shape)
        LU.copy_(input_contiguous)
        lu = LU
    else:
        lu = input_contiguous.clone()
    if pivots is not None:
        pivots.resize_((*batch_shape, k))
        piv = pivots
    else:
        piv = torch.empty(
            (*batch_shape, k), device=input_contiguous.device, dtype=torch.int32
        )

    with torch_device_fn.device(input_contiguous.device):
        # Panel width: 64 keeps the swap-free kernel inside its proven
        # <=256-row tile; taller inputs run the physical-swap panel, whose
        # 512x64 tile exceeds the DSM budget, so they run at width 32.
        panel_size = (
            _LU_FACTOR_PANEL if m <= _SWAP_FREE_PANEL_MAX_ROWS else _SWAP_PANEL_WIDTH
        )
        for k0 in range(0, k, panel_size):
            panel = min(panel_size, k - k0)
            trailing_n = n - k0 - panel
            trailing_m = m - k0 - panel
            block_b = triton.next_power_of_2(panel)
            panel_rows = m - k0
            if panel_rows > _SWAP_FREE_PANEL_MAX_ROWS:
                # 512-row swap-free panels silently corrupt on this build;
                # the physical-swap panel is measured correct at this size.
                if pivot:
                    _lu_factor_panel_swap_kernel[(batch,)](
                        lu,
                        piv,
                        k0,
                        m,
                        n,
                        k,
                        panel,
                        triton.next_power_of_2(m),
                        block_b,
                        1,
                        False,
                        num_warps=4,
                    )
                else:
                    _lu_factor_panel_no_pivot_swap_kernel[(batch,)](
                        lu,
                        piv,
                        k0,
                        m,
                        n,
                        k,
                        panel,
                        triton.next_power_of_2(m),
                        block_b,
                        num_warps=4,
                    )
            else:
                _lu_factor_panel_kernel[(batch,)](
                    lu,
                    piv,
                    k0,
                    m,
                    n,
                    k,
                    panel,
                    triton.next_power_of_2(panel_rows),
                    block_b,
                    pivot,
                    num_warps=1,
                )
            if trailing_n > 0:
                tiles_n = triton.cdiv(trailing_n, _LU_FACTOR_TILE_N)
                _lu_factor_swap_right_and_solve_kernel[(tiles_n * batch,)](
                    lu,
                    piv,
                    k0,
                    m,
                    n,
                    k,
                    panel,
                    block_b,
                    _LU_FACTOR_TILE_N,
                    tiles_n,
                    num_warps=4,
                )
            if trailing_m > 0 and trailing_n > 0:
                tiles_m = triton.cdiv(trailing_m, _LU_FACTOR_TILE_M)
                tiles_n = triton.cdiv(trailing_n, _LU_FACTOR_TILE_N)
                _lu_factor_trailing_update_kernel[(tiles_m * tiles_n * batch,)](
                    lu,
                    k0,
                    m,
                    n,
                    panel,
                    _LU_FACTOR_TILE_M,
                    _LU_FACTOR_TILE_N,
                    block_b,
                    tiles_n,
                    batch,
                    num_warps=4,
                )

        # Final pass: replay the panel pivots on the left columns (L factors).
        for k0 in range(panel_size, k, panel_size):
            panel = min(panel_size, k - k0)
            tiles_n = triton.cdiv(k0, _LU_FACTOR_TILE_N)
            _lu_factor_apply_panel_pivots_kernel[(tiles_n * batch,)](
                lu,
                piv,
                k0,
                m,
                n,
                k,
                panel,
                0,
                k0,
                _LU_FACTOR_TILE_N,
                tiles_n,
                num_warps=4,
            )
    return lu, piv


def _blocked_lu_factor_batch_safe(input_contiguous, pivot, LU, pivots):
    """Generic blocked path with a GCU grid.y-safe host split for large batches.

    The generic blocked LU launches its swap/solve/update kernels with
    grid=(tiles, batch); on GCU400 grid.y/grid.z are capped at 255, so a
    batched input with batch > 255 raises OutOfResources (the benchmark
    failure this file fixes).  Batch elements are independent, so split the
    batch on the host into <= 255 chunks and run the generic FlagGems
    ``linalg_lu_factor_out`` op per chunk (host-side views only; no new
    kernels).
    """
    m, n = input_contiguous.shape[-2], input_contiguous.shape[-1]
    batch = input_contiguous.numel() // (m * n)
    batch_shape = input_contiguous.shape[:-2]
    k = min(m, n)

    # Honour each out tensor independently (the generic pairing rule): the
    # contiguous ones are written in place, the rest via copy-back below.
    # Missing ones are allocated here, and every batch element goes through
    # the chunked generic out-variant (<= 255 per chunk keeps grid.y legal).
    if LU is not None and LU.is_contiguous():
        LU.resize_(input_contiguous.shape)
        lu = LU
    else:
        lu = torch.empty_like(input_contiguous)
    if pivots is not None and pivots.is_contiguous():
        pivots.resize_((*batch_shape, k))
        piv = pivots
    else:
        piv = torch.empty(
            (*batch_shape, k), device=input_contiguous.device, dtype=torch.int32
        )

    input_flat = input_contiguous.reshape(batch, m, n)
    lu_flat = lu.reshape(batch, m, n)
    piv_flat = piv.reshape(batch, k)
    for start in range(0, batch, _GCU_GRID_Y_MAX):
        stop = min(start + _GCU_GRID_Y_MAX, batch)
        _generic_lu_factor_out(
            input_flat[start:stop],
            pivot=pivot,
            LU=lu_flat[start:stop],
            pivots=piv_flat[start:stop],
        )

    # Non-contiguous out tensors are honoured by copy-back.
    if LU is not None and lu is not LU:
        LU.resize_(input_contiguous.shape)
        LU.copy_(lu)
    if pivots is not None and piv is not pivots:
        pivots.resize_((*batch_shape, k))
        pivots.copy_(piv)
    return lu, piv


def _linalg_lu_factor_impl(input, *, pivot=True, LU=None, pivots=None):
    _linalg_lu_factor_check(input, pivot)

    if (LU is not None and not LU.is_contiguous()) or (
        pivots is not None and not pivots.is_contiguous()
    ):
        lu, piv = _linalg_lu_factor_impl(input, pivot=pivot)
        LU.resize_(lu.shape)
        pivots.resize_(piv.shape)
        LU.copy_(lu)
        pivots.copy_(piv)
        return LinalgLUFactorResult(LU, pivots)

    input_contiguous = input.contiguous()

    if _can_use_fast_triton(input_contiguous):
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
            piv = torch.empty((*batch_shape, k), device=input.device, dtype=torch.int32)
        else:
            pivots.resize_((*batch_shape, k))
            piv = pivots

        with torch_device_fn.device(input.device):
            block_m = triton.next_power_of_2(m)
            block_n = triton.next_power_of_2(n)
            _linalg_lu_factor_kernel[(batch,)](
                input_contiguous,
                lu,
                piv,
                m,
                n,
                k,
                block_m,
                block_n,
                pivot,
                num_warps=4 if block_m * block_n >= _FAST_TILE_NW_AREA else 1,
            )
        return LinalgLUFactorResult(lu, piv)
    elif _can_use_panel_cascade(input_contiguous):
        lu, piv = _lu_factor_cascade(input_contiguous, pivot, LU, pivots)
        return LinalgLUFactorResult(lu, piv)
    else:
        lu, piv = _blocked_lu_factor_batch_safe(input_contiguous, pivot, LU, pivots)
        return LinalgLUFactorResult(lu, piv)


def linalg_lu_factor(input, *, pivot=True):
    logger.debug("GEMS_ENFLAME LINALG_LU_FACTOR")
    return _linalg_lu_factor_impl(input, pivot=pivot)


def _resolve_linalg_lu_factor_out_args(LU, pivots, out):
    # Two spellings reach here, and they must not be mixed:
    #   * ``out=(LU, pivots)``, the one ``torch.linalg`` documents and the only
    #     one a *direct* call needs -- the dispatcher never passes ``out`` (it
    #     binds the schema's ``LU``/``pivots`` arguments instead), so this form
    #     is invisible to the ``linalg_lu_factor.out`` registration;
    #   * explicit ``LU=``/``pivots=``, which is what that registration uses.
    # Mirrors ``_resolve_linalg_lu_factor_ex_out_args`` and the ascend backend.
    if out is not None:
        if LU is not None or pivots is not None:
            raise TypeError("linalg_lu_factor(): out and LU/pivots cannot both be set")
        if len(out) != 2:
            raise TypeError(
                "linalg_lu_factor(): out must be a tuple of 2 tensors, "
                f"got {len(out)}"
            )
        return out
    if LU is None or pivots is None:
        raise TypeError(
            "linalg_lu_factor(): LU and pivots must both be provided " "for out variant"
        )
    return LU, pivots


def linalg_lu_factor_out(input, *, pivot=True, LU=None, pivots=None, out=None):
    logger.debug("GEMS_ENFLAME LINALG_LU_FACTOR_OUT")
    lu_out, pivots_out = _resolve_linalg_lu_factor_out_args(LU, pivots, out)
    return _linalg_lu_factor_impl(input, pivot=pivot, LU=lu_out, pivots=pivots_out)
