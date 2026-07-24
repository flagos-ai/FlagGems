import logging
from collections import namedtuple

import torch
import triton

# import triton.experimental.tle.language as tle
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

LinalgLUFactorResult = namedtuple("LinalgLUFactorResult", ["LU", "pivots"])

_LU_FACTOR_BLOCK_MAX = 64
_LU_FACTOR_PANEL = 16
_LU_FACTOR_TILE_M = 64
_LU_FACTOR_TILE_N = 128
_LU_FACTOR_FUSED_TILE_N = 16
_LU_FACTOR_ENABLE_FUSED_PIVOT = False


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

    offsets = pid * M * N + rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    work = tl.load(A + offsets, mask=mask, other=0.0).to(tl.float32)

    for j_ind in tl.range(0, K):
        if PIVOT:
            col_j = tl.sum(
                tl.where(cols[:, None] == j_ind, tl.trans(work), 0.0), axis=0
            )
            abs_col = tl.abs(col_j)
            abs_col = tl.where(rows < j_ind, -1.0, abs_col)
            abs_col = tl.where(rows < M, abs_col, -1.0)
            pivot_val = tl.max(abs_col, axis=0)
            pivot_row = tl.min(tl.where(abs_col == pivot_val, rows, BLOCK_M), axis=0)

            row_j = tl.sum(tl.where(rows[:, None] == j_ind, work, 0.0), axis=0)
            row_p = tl.sum(tl.where(rows[:, None] == pivot_row, work, 0.0), axis=0)
            col_mask = cols[None, :] < N
            work = tl.where((rows[:, None] == j_ind) & col_mask, row_p, work)
            work = tl.where((rows[:, None] == pivot_row) & col_mask, row_j, work)
            tl.store(PIVOTS + pid * K + j_ind, pivot_row + 1)
        else:
            tl.store(PIVOTS + pid * K + j_ind, j_ind + 1)

        pivot = tl.sum(
            tl.sum(
                tl.where(
                    (rows[:, None] == j_ind) & (cols[None, :] == j_ind), work, 0.0
                ),
                axis=0,
            ),
            axis=0,
        )

        pivot_row_vals = tl.sum(tl.where(rows[:, None] == j_ind, work, 0.0), axis=0)
        active_cols = cols > j_ind
        work = tl.where(
            (rows[:, None] == j_ind) & active_cols[None, :], pivot_row_vals, work
        )

        col_vals = tl.sum(tl.where(cols[:, None] == j_ind, tl.trans(work), 0.0), axis=0)
        multipliers = tl.where(rows > j_ind, col_vals / pivot, col_vals)
        work = tl.where(
            (rows[:, None] > j_ind) & (cols[None, :] == j_ind),
            multipliers[:, None],
            work,
        )

        l_col = tl.sum(tl.where(cols[:, None] == j_ind, tl.trans(work), 0.0), axis=0)
        u_row = tl.sum(tl.where(rows[:, None] == j_ind, work, 0.0), axis=0)
        update_mask = (rows[:, None] > j_ind) & (cols[None, :] > j_ind)
        work = tl.where(update_mask, work - l_col[:, None] * u_row[None, :], work)

    tl.store(LU + offsets, work, mask=mask)


@libentry()
@triton.jit
def _lu_factor_panel_no_pivot_kernel(
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
        col_vals = tl.where(rows > j, col_vals / pivot, col_vals)
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
        col_vals = tl.where(rows > j, col_vals / pivot, col_vals)
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
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
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


@libentry()
@triton.jit
def _lu_factor_solve_block_row_no_pivot_kernel(
    LU,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    PANEL: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    brows = tl.arange(0, BLOCK_B)
    cols = K0 + PANEL + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = K0 + brows

    offsets = pid_b * M * N + rows[:, None] * N + cols[None, :]
    mask = (brows[:, None] < PANEL) & (cols[None, :] < N)
    vals = tl.load(LU + offsets, mask=mask, other=0.0).to(tl.float32)

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
):
    """Apply panel pivots to trailing columns and solve for U rows in one pass."""
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
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
def _lu_factor_panel_swap_right_and_solve_kernel(
    LU,
    PIVOTS,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PANEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_B: tl.constexpr,
    RIGHT_BLOCK_N: tl.constexpr,
    LEFT_BLOCK_N: tl.constexpr,
    APPLY_LEFT: tl.constexpr,
):
    """Factor a pivoted panel and solve its right block in one program."""
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    bcols = tl.arange(0, BLOCK_B)
    panel_cols = K0 + bcols
    left_cols = tl.arange(0, LEFT_BLOCK_N)

    panel_offsets = pid * M * N + rows[:, None] * N + panel_cols[None, :]
    panel_mask = (rows[:, None] < M) & (bcols[None, :] < PANEL)
    panel_vals = tl.load(LU + panel_offsets, mask=panel_mask, other=0.0).to(tl.float32)

    brows = tl.arange(0, BLOCK_B)
    right_cols = K0 + PANEL + tl.arange(0, RIGHT_BLOCK_N)
    right_rows = K0 + brows
    right_offsets = pid * M * N + right_rows[:, None] * N + right_cols[None, :]
    right_mask = (brows[:, None] < PANEL) & (right_cols[None, :] < N)
    right_vals = tl.load(LU + right_offsets, mask=right_mask, other=0.0).to(tl.float32)

    for jj in tl.range(0, PANEL):
        j = K0 + jj

        col_vals = tl.sum(
            tl.where(bcols[:, None] == jj, tl.trans(panel_vals), 0.0), axis=0
        )
        abs_col = tl.abs(col_vals)
        abs_col = tl.where(rows < j, -1.0, abs_col)
        abs_col = tl.where(rows < M, abs_col, -1.0)
        pivot_val = tl.max(abs_col, axis=0)
        pivot_row = tl.min(tl.where(abs_col == pivot_val, rows, BLOCK_M), axis=0)

        row_j_panel = tl.sum(tl.where(rows[:, None] == j, panel_vals, 0.0), axis=0)
        row_p_panel = tl.sum(
            tl.where(rows[:, None] == pivot_row, panel_vals, 0.0), axis=0
        )
        panel_vals = tl.where(
            (rows[:, None] == j) & panel_mask, row_p_panel[None, :], panel_vals
        )
        panel_vals = tl.where(
            (rows[:, None] == pivot_row) & panel_mask,
            row_j_panel[None, :],
            panel_vals,
        )
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

        row_j_right = tl.sum(tl.where(brows[:, None] == jj, right_vals, 0.0), axis=0)
        row_p_right_offsets = pid * M * N + pivot_row * N + right_cols
        row_p_right = tl.load(
            LU + row_p_right_offsets, mask=right_cols < N, other=0.0
        ).to(tl.float32)
        right_vals = tl.where(
            (brows[:, None] == jj) & right_mask,
            row_p_right[None, :],
            right_vals,
        )
        rel_pivot = pivot_row - K0
        right_vals = tl.where(
            (brows[:, None] == rel_pivot) & right_mask,
            row_j_right[None, :],
            right_vals,
        )
        tl.store(LU + row_p_right_offsets, row_j_right, mask=right_cols < N)

        pivot = tl.sum(
            tl.sum(
                tl.where(
                    (rows[:, None] == j) & (bcols[None, :] == jj),
                    panel_vals,
                    0.0,
                ),
                axis=0,
            ),
            axis=0,
        )

        col_vals = tl.sum(
            tl.where(bcols[:, None] == jj, tl.trans(panel_vals), 0.0), axis=0
        )
        col_vals = tl.where(rows > j, col_vals / pivot, col_vals)
        panel_vals = tl.where(
            (rows[:, None] > j) & (bcols[None, :] == jj),
            col_vals[:, None],
            panel_vals,
        )

        l_col = tl.sum(
            tl.where(bcols[:, None] == jj, tl.trans(panel_vals), 0.0), axis=0
        )
        u_row = tl.sum(tl.where(rows[:, None] == j, panel_vals, 0.0), axis=0)
        update_mask = (rows[:, None] > j) & (bcols[None, :] > jj)
        panel_vals = tl.where(
            update_mask, panel_vals - l_col[:, None] * u_row[None, :], panel_vals
        )

    for jj in tl.range(0, PANEL):
        row_j = tl.sum(tl.where(brows[:, None] == jj, right_vals, 0.0), axis=0)

        l_col_all = tl.sum(
            tl.where(bcols[:, None] == jj, tl.trans(panel_vals), 0.0), axis=0
        )
        l_col = tl.sum(
            tl.where(rows[:, None] == right_rows[None, :], l_col_all[:, None], 0.0),
            axis=0,
        )
        l_col = tl.where(brows <= jj, 0.0, l_col)
        right_vals = tl.where(
            brows[:, None] > jj,
            right_vals - l_col[:, None] * row_j[None, :],
            right_vals,
        )

    tl.store(LU + panel_offsets, panel_vals, mask=panel_mask)
    tl.store(LU + right_offsets, right_vals, mask=right_mask)


@libentry()
@triton.jit
def _lu_factor_panel_solve_no_pivot_kernel(
    LU,
    PIVOTS,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PANEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_B: tl.constexpr,
    RIGHT_BLOCK_N: tl.constexpr,
):
    """Factor a no-pivot panel and solve its right block in one program."""
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    bcols = tl.arange(0, BLOCK_B)
    panel_cols = K0 + bcols

    panel_offsets = pid * M * N + rows[:, None] * N + panel_cols[None, :]
    panel_mask = (rows[:, None] < M) & (bcols[None, :] < PANEL)
    panel_vals = tl.load(LU + panel_offsets, mask=panel_mask, other=0.0).to(tl.float32)

    brows = tl.arange(0, BLOCK_B)
    right_cols = K0 + PANEL + tl.arange(0, RIGHT_BLOCK_N)
    right_rows = K0 + brows
    right_offsets = pid * M * N + right_rows[:, None] * N + right_cols[None, :]
    right_mask = (brows[:, None] < PANEL) & (right_cols[None, :] < N)
    right_vals = tl.load(LU + right_offsets, mask=right_mask, other=0.0).to(tl.float32)

    for jj in tl.range(0, PANEL):
        j = K0 + jj

        pivot = tl.sum(
            tl.sum(
                tl.where(
                    (rows[:, None] == j) & (bcols[None, :] == jj), panel_vals, 0.0
                ),
                axis=0,
            ),
            axis=0,
        )

        col_vals = tl.sum(
            tl.where(bcols[:, None] == jj, tl.trans(panel_vals), 0.0), axis=0
        )
        col_vals = tl.where(rows > j, col_vals / pivot, col_vals)
        panel_vals = tl.where(
            (rows[:, None] > j) & (bcols[None, :] == jj),
            col_vals[:, None],
            panel_vals,
        )

        l_col = tl.sum(
            tl.where(bcols[:, None] == jj, tl.trans(panel_vals), 0.0), axis=0
        )
        u_row = tl.sum(tl.where(rows[:, None] == j, panel_vals, 0.0), axis=0)
        update_mask = (rows[:, None] > j) & (bcols[None, :] > jj)
        panel_vals = tl.where(
            update_mask, panel_vals - l_col[:, None] * u_row[None, :], panel_vals
        )
        tl.store(PIVOTS + pid * K + j, j + 1)

    for jj in tl.range(0, PANEL):
        row_j = tl.sum(tl.where(brows[:, None] == jj, right_vals, 0.0), axis=0)

        l_col_all = tl.sum(
            tl.where(bcols[:, None] == jj, tl.trans(panel_vals), 0.0), axis=0
        )
        l_col = tl.sum(
            tl.where(rows[:, None] == right_rows[None, :], l_col_all[:, None], 0.0),
            axis=0,
        )
        l_col = tl.where(brows <= jj, 0.0, l_col)
        right_vals = tl.where(
            brows[:, None] > jj,
            right_vals - l_col[:, None] * row_j[None, :],
            right_vals,
        )

    tl.store(LU + panel_offsets, panel_vals, mask=panel_mask)
    tl.store(LU + right_offsets, right_vals, mask=right_mask)


@triton.jit
def _lu_factor_fused_iter_no_pivot_kernel(
    LU,
    PIVOTS,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PANEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_B: tl.constexpr,
    RIGHT_BLOCK_N: tl.constexpr,
    TRAIL_BLOCK_M: tl.constexpr,
    TRAIL_BLOCK_N: tl.constexpr,
    FUSE_NEXT_PANEL: tl.constexpr,
    USE_GATHER: tl.constexpr,
):
    """Fused panel factor + right-block solve + trailing update for one iteration.

    For small matrices (k <= 128) where the trailing update fits in a single
    thread block, this eliminates two kernel launches per panel iteration.
    """
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    bcols = tl.arange(0, BLOCK_B)
    panel_cols = K0 + bcols

    # ---- Load panel ----
    panel_offsets = pid * M * N + rows[:, None] * N + panel_cols[None, :]
    panel_mask = (rows[:, None] < M) & (bcols[None, :] < PANEL)
    panel_vals = tl.load(LU + panel_offsets, mask=panel_mask, other=0.0).to(tl.float32)

    # ---- Load right block ----
    brows = tl.arange(0, BLOCK_B)
    right_cols = K0 + PANEL + tl.arange(0, RIGHT_BLOCK_N)
    right_rows = K0 + brows
    right_offsets = pid * M * N + right_rows[:, None] * N + right_cols[None, :]
    right_mask = (brows[:, None] < PANEL) & (right_cols[None, :] < N)
    right_vals = tl.load(LU + right_offsets, mask=right_mask, other=0.0).to(tl.float32)

    # ---- Factor panel + solve right block (interleaved) ----
    # Interleaving eliminates ~128 redundant column extractions by reusing
    # the just-computed L column for both panel rank-1 update and right-block
    # triangular solve in a single pass.
    for jj in tl.range(0, PANEL):
        j = K0 + jj
        j_local = jj  # local column index within panel

        # Extract column jj from panel
        if USE_GATHER:
            col_idx = tl.full((BLOCK_M, 1), j_local, dtype=tl.int32)
            col_vals = tl.ravel(tl.gather(panel_vals, col_idx, axis=1))
        else:
            col_vals = tl.sum(
                tl.where(bcols[None, :] == j_local, panel_vals, 0.0), axis=1
            )

        # Extract row jj from panel
        if USE_GATHER:
            row_idx = tl.full((1, BLOCK_B), j_local, dtype=tl.int32)
            u_row = tl.ravel(tl.gather(panel_vals, row_idx, axis=0))
        else:
            u_row = tl.sum(tl.where(rows[:, None] == j_local, panel_vals, 0.0), axis=0)

        # Extract row jj from right block
        if USE_GATHER:
            right_row_idx = tl.full((1, RIGHT_BLOCK_N), j_local, dtype=tl.int32)
            row_j = tl.ravel(tl.gather(right_vals, right_row_idx, axis=0))
        else:
            row_j = tl.sum(tl.where(brows[:, None] == j_local, right_vals, 0.0), axis=0)

        # Pivot (diagonal element for no-pivot)
        pivot = tl.sum(tl.where(rows == j_local, col_vals, 0.0), axis=0)

        # Scale column below diagonal
        scaled_col = tl.where(rows > j_local, col_vals / pivot, col_vals)

        # Write scaled column back to panel
        panel_vals = tl.where(
            (rows[:, None] > j_local) & (bcols[None, :] == j_local),
            scaled_col[:, None],
            panel_vals,
        )

        # Rank-1 update on trailing panel (cols > jj)
        update_mask = (rows[:, None] > j_local) & (bcols[None, :] > j_local)
        panel_vals = tl.where(
            update_mask, panel_vals - scaled_col[:, None] * u_row[None, :], panel_vals
        )

        # Right-block solve using the same L column (scaled_col).
        # Gather L factors for right-block rows (O(1) gather vs reduction).
        l_col_right = tl.gather(scaled_col, right_rows, axis=0)
        l_col_right = tl.where(brows <= j_local, 0.0, l_col_right)
        right_vals = tl.where(
            brows[:, None] > j_local,
            right_vals - l_col_right[:, None] * row_j[None, :],
            right_vals,
        )

        tl.store(PIVOTS + pid * K + j, j + 1)

    # Store panel (needed for L21 load below)
    tl.store(LU + panel_offsets, panel_vals, mask=panel_mask)
    # Right block will be stored after factoring to avoid writing unfactored U12
    # before the trailing update consumes it. But the trailing update loads U12
    # from global memory after it's stored, so we store right_vals first.
    tl.store(LU + right_offsets, right_vals, mask=right_mask)

    # ---- Trailing update: trailing -= L21 @ U12 ----
    trail_rows = K0 + PANEL + tl.arange(0, TRAIL_BLOCK_M)
    trail_cols = K0 + PANEL + tl.arange(0, TRAIL_BLOCK_N)

    # Load L21 from global memory (lower part of stored panel)
    l_offsets = pid * M * N + trail_rows[:, None] * N + (K0 + bcols[None, :])
    l_mask = (trail_rows[:, None] < M) & (bcols[None, :] < PANEL)
    l21 = tl.load(
        LU + l_offsets,
        mask=l_mask,
        other=0.0,
    )

    # Load U12 from global memory (stored right block)
    u_offsets = pid * M * N + (K0 + brows[:, None]) * N + trail_cols[None, :]
    u_mask = (brows[:, None] < PANEL) & (trail_cols[None, :] < N)
    u12 = tl.load(
        LU + u_offsets,
        mask=u_mask,
        other=0.0,
    )

    # Load trailing submatrix
    t_offsets = pid * M * N + trail_rows[:, None] * N + trail_cols[None, :]
    t_mask = (trail_rows[:, None] < M) & (trail_cols[None, :] < N)
    trail = tl.load(LU + t_offsets, mask=t_mask, other=0.0).to(tl.float32)

    update = tl.dot(l21, u12, input_precision="ieee")
    trail = trail - update

    # ---- Factor next panel in-place (fuse second panel to eliminate kernel launch) ----
    HAS_NEXT: tl.constexpr = FUSE_NEXT_PANEL and ((K0 + PANEL) < K)
    if HAS_NEXT:
        PANEL2: tl.constexpr = K - K0 - PANEL
        trail_local_rows = tl.arange(0, TRAIL_BLOCK_M)
        trail_local_cols = tl.arange(0, TRAIL_BLOCK_N)
        for jj in tl.range(0, PANEL2):
            j = K0 + PANEL + jj

            # Extract column jj from trail
            if USE_GATHER:
                col_idx = tl.full((TRAIL_BLOCK_M, 1), jj, dtype=tl.int32)
                col_vals = tl.ravel(tl.gather(trail, col_idx, axis=1))
            else:
                col_vals = tl.sum(
                    tl.where(trail_local_cols[None, :] == jj, trail, 0.0), axis=1
                )

            # Extract row jj from trail
            if USE_GATHER:
                row_idx = tl.full((1, TRAIL_BLOCK_N), jj, dtype=tl.int32)
                u_row = tl.ravel(tl.gather(trail, row_idx, axis=0))
            else:
                u_row = tl.sum(
                    tl.where(trail_local_rows[:, None] == jj, trail, 0.0), axis=0
                )

            # Pivot (diagonal element)
            pivot = tl.sum(tl.where(trail_local_rows == jj, col_vals, 0.0), axis=0)

            # Scale column below diagonal
            scaled_col = tl.where(trail_local_rows > jj, col_vals / pivot, col_vals)

            # Write L column
            trail = tl.where(
                (trail_local_rows[:, None] > jj) & (trail_local_cols[None, :] == jj),
                scaled_col[:, None],
                trail,
            )

            # Rank-1 update on trailing submatrix
            update_mask = (trail_local_rows[:, None] > jj) & (
                trail_local_cols[None, :] > jj
            )
            trail = tl.where(
                update_mask, trail - scaled_col[:, None] * u_row[None, :], trail
            )

            tl.store(PIVOTS + pid * K + j, j + 1)

    tl.store(LU + t_offsets, trail, mask=t_mask)


@triton.jit
def _lu_factor_fused_iter_pivot_kernel(
    LU,
    PIVOTS,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PANEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_B: tl.constexpr,
    RIGHT_BLOCK_N: tl.constexpr,
    TRAIL_BLOCK_M: tl.constexpr,
    TRAIL_BLOCK_N: tl.constexpr,
    FUSE_NEXT_PANEL: tl.constexpr,
    USE_GATHER: tl.constexpr,
):
    """Fused pivot panel factor + right swap-and-solve + trailing update.

    For small matrices (k <= 128) where the trailing update fits in a single
    thread block, this eliminates two kernel launches per panel iteration.
    Only used when k0 == 0 (APPLY_LEFT is False).
    """
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    bcols = tl.arange(0, BLOCK_B)
    panel_cols = K0 + bcols

    # ---- Load panel ----
    panel_offsets = pid * M * N + rows[:, None] * N + panel_cols[None, :]
    panel_mask = (rows[:, None] < M) & (bcols[None, :] < PANEL)
    panel_vals = tl.load(LU + panel_offsets, mask=panel_mask, other=0.0).to(tl.float32)

    # ---- Load right block ----
    brows = tl.arange(0, BLOCK_B)
    right_cols = K0 + PANEL + tl.arange(0, RIGHT_BLOCK_N)
    right_rows = K0 + brows
    right_offsets = pid * M * N + right_rows[:, None] * N + right_cols[None, :]
    right_mask = (brows[:, None] < PANEL) & (right_cols[None, :] < N)
    right_vals = tl.load(LU + right_offsets, mask=right_mask, other=0.0).to(tl.float32)

    # ---- Factor panel + solve right block (interleaved) ----
    # Interleaving eliminates ~64 redundant column extractions and the separate
    # right-block solve loop, reusing the just-computed L column for both the
    # panel rank-1 update and the right-block triangular solve.
    for jj in tl.range(0, PANEL):
        j = K0 + jj

        # Pivot search — extract column jj
        if USE_GATHER:
            col_idx = tl.full((BLOCK_M, 1), jj, dtype=tl.int32)
            row_j_idx = tl.full((1, BLOCK_B), jj, dtype=tl.int32)
            right_row_idx = tl.full((1, RIGHT_BLOCK_N), jj, dtype=tl.int32)
            col_vals = tl.ravel(tl.gather(panel_vals, col_idx, axis=1))
        else:
            col_vals = tl.sum(tl.where(bcols[None, :] == jj, panel_vals, 0.0), axis=1)
        abs_col = tl.abs(col_vals)
        abs_col = tl.where(rows < j, -1.0, abs_col)
        abs_col = tl.where(rows < M, abs_col, -1.0)
        pivot_val = tl.max(abs_col, axis=0)
        pivot_row = tl.min(tl.where(abs_col == pivot_val, rows, BLOCK_M), axis=0)

        # Swap rows in panel — extract rows
        if USE_GATHER:
            row_j_panel = tl.ravel(tl.gather(panel_vals, row_j_idx, axis=0))
        else:
            row_j_panel = tl.sum(tl.where(rows[:, None] == jj, panel_vals, 0.0), axis=0)
        if USE_GATHER:
            row_p_idx = tl.full((1, BLOCK_B), pivot_row, dtype=tl.int32)
            row_p_panel = tl.ravel(tl.gather(panel_vals, row_p_idx, axis=0))
        else:
            row_p_panel = tl.sum(
                tl.where(rows[:, None] == pivot_row, panel_vals, 0.0), axis=0
            )
        panel_vals = tl.where(
            (rows[:, None] == j) & panel_mask, row_p_panel[None, :], panel_vals
        )
        panel_vals = tl.where(
            (rows[:, None] == pivot_row) & panel_mask,
            row_j_panel[None, :],
            panel_vals,
        )
        tl.store(PIVOTS + pid * K + j, pivot_row + 1)

        # Swap rows in right block — extract row jj
        if USE_GATHER:
            row_j_right = tl.ravel(tl.gather(right_vals, right_row_idx, axis=0))
        else:
            row_j_right = tl.sum(
                tl.where(brows[:, None] == jj, right_vals, 0.0), axis=0
            )
        row_p_right_offsets = pid * M * N + pivot_row * N + right_cols
        row_p_right = tl.load(
            LU + row_p_right_offsets, mask=right_cols < N, other=0.0
        ).to(tl.float32)
        right_vals = tl.where(
            (brows[:, None] == jj) & right_mask,
            row_p_right[None, :],
            right_vals,
        )
        rel_pivot = pivot_row - K0
        right_vals = tl.where(
            (brows[:, None] == rel_pivot) & right_mask,
            row_j_right[None, :],
            right_vals,
        )
        tl.store(LU + row_p_right_offsets, row_j_right, mask=right_cols < N)

        # Pivot value after swap
        pivot = tl.sum(
            tl.sum(
                tl.where(
                    (rows[:, None] == j) & (bcols[None, :] == jj),
                    panel_vals,
                    0.0,
                ),
                axis=0,
            ),
            axis=0,
        )

        # Scale column below diagonal
        if USE_GATHER:
            col_vals = tl.ravel(tl.gather(panel_vals, col_idx, axis=1))
        else:
            col_vals = tl.sum(tl.where(bcols[None, :] == jj, panel_vals, 0.0), axis=1)
        col_vals = tl.where(rows > j, col_vals / pivot, col_vals)
        panel_vals = tl.where(
            (rows[:, None] > j) & (bcols[None, :] == jj),
            col_vals[:, None],
            panel_vals,
        )

        # Rank-1 update on trailing sub-panel
        # Re-extract L column after scaling (column jj is unchanged by rank-1 update)
        if USE_GATHER:
            l_col = tl.ravel(tl.gather(panel_vals, col_idx, axis=1))
        else:
            l_col = tl.sum(tl.where(bcols[None, :] == jj, panel_vals, 0.0), axis=1)
        if USE_GATHER:
            u_row = tl.ravel(tl.gather(panel_vals, row_j_idx, axis=0))
        else:
            u_row = tl.sum(tl.where(rows[:, None] == jj, panel_vals, 0.0), axis=0)
        update_mask = (rows[:, None] > j) & (bcols[None, :] > jj)
        panel_vals = tl.where(
            update_mask, panel_vals - l_col[:, None] * u_row[None, :], panel_vals
        )

        # ---- Right-block solve for this column (interleaved, reuses l_col) ----
        if USE_GATHER:
            row_j = tl.ravel(tl.gather(right_vals, right_row_idx, axis=0))
        else:
            row_j = tl.sum(tl.where(brows[:, None] == jj, right_vals, 0.0), axis=0)

        # Gather L factors for right-block rows (O(1) vs reduction)
        l_col_right = tl.gather(l_col, right_rows, axis=0)
        l_col_right = tl.where(brows <= jj, 0.0, l_col_right)
        right_vals = tl.where(
            brows[:, None] > jj,
            right_vals - l_col_right[:, None] * row_j[None, :],
            right_vals,
        )

    # Store panel and right block to global memory
    tl.store(LU + panel_offsets, panel_vals, mask=panel_mask)
    tl.store(LU + right_offsets, right_vals, mask=right_mask)

    # ---- Trailing update: trailing -= L21 @ U12 ----
    trail_rows = K0 + PANEL + tl.arange(0, TRAIL_BLOCK_M)
    trail_cols = K0 + PANEL + tl.arange(0, TRAIL_BLOCK_N)

    # Load L21 from global memory
    l_offsets = pid * M * N + trail_rows[:, None] * N + (K0 + bcols[None, :])
    l_mask = (trail_rows[:, None] < M) & (bcols[None, :] < PANEL)
    l21 = tl.load(
        LU + l_offsets,
        mask=l_mask,
        other=0.0,
    )

    # Load U12 from global memory
    u_offsets = pid * M * N + (K0 + brows[:, None]) * N + trail_cols[None, :]
    u_mask = (brows[:, None] < PANEL) & (trail_cols[None, :] < N)
    u12 = tl.load(
        LU + u_offsets,
        mask=u_mask,
        other=0.0,
    )

    # Load trailing submatrix
    t_offsets = pid * M * N + trail_rows[:, None] * N + trail_cols[None, :]
    t_mask = (trail_rows[:, None] < M) & (trail_cols[None, :] < N)
    trail = tl.load(LU + t_offsets, mask=t_mask, other=0.0).to(tl.float32)

    update = tl.dot(l21, u12, input_precision="ieee")
    trail = trail - update

    # ---- Factor next panel in-place with pivoting (fuse second panel) ----
    HAS_NEXT: tl.constexpr = FUSE_NEXT_PANEL and ((K0 + PANEL) < K)
    if HAS_NEXT:
        PANEL2: tl.constexpr = K - K0 - PANEL
        trail_local_rows = tl.arange(0, TRAIL_BLOCK_M)
        trail_local_cols = tl.arange(0, TRAIL_BLOCK_N)
        for jj2 in tl.range(0, PANEL2):
            j = K0 + PANEL + jj2

            # Pivot search on column jj2 of trail
            if USE_GATHER:
                col_idx2 = tl.full((TRAIL_BLOCK_M, 1), jj2, dtype=tl.int32)
                row_jj_idx = tl.full((1, TRAIL_BLOCK_N), jj2, dtype=tl.int32)
                col_vals = tl.ravel(tl.gather(trail, col_idx2, axis=1))
            else:
                col_vals = tl.sum(
                    tl.where(trail_local_cols[None, :] == jj2, trail, 0.0), axis=1
                )
            abs_col = tl.abs(col_vals)
            abs_col = tl.where(trail_local_rows < jj2, -1.0, abs_col)
            trail_global_rows = K0 + PANEL + trail_local_rows
            abs_col = tl.where(trail_global_rows < M, abs_col, -1.0)
            pivot_val = tl.max(abs_col, axis=0)
            pivot_row_local = tl.min(
                tl.where(abs_col == pivot_val, trail_local_rows, TRAIL_BLOCK_M), axis=0
            )

            # Swap rows in trail
            if USE_GATHER:
                row_jj = tl.ravel(tl.gather(trail, row_jj_idx, axis=0))
            else:
                row_jj = tl.sum(
                    tl.where(trail_local_rows[:, None] == jj2, trail, 0.0), axis=0
                )
            if USE_GATHER:
                row_p_idx = tl.full((1, TRAIL_BLOCK_N), pivot_row_local, dtype=tl.int32)
                row_p = tl.ravel(tl.gather(trail, row_p_idx, axis=0))
            else:
                row_p = tl.sum(
                    tl.where(trail_local_rows[:, None] == pivot_row_local, trail, 0.0),
                    axis=0,
                )
            trail = tl.where((trail_local_rows[:, None] == jj2), row_p[None, :], trail)
            trail = tl.where(
                (trail_local_rows[:, None] == pivot_row_local), row_jj[None, :], trail
            )

            pivot_row_global = K0 + PANEL + pivot_row_local
            tl.store(PIVOTS + pid * K + j, pivot_row_global + 1)

            # Extract column after swap for pivot value
            if USE_GATHER:
                col_vals = tl.ravel(tl.gather(trail, col_idx2, axis=1))
            else:
                col_vals = tl.sum(
                    tl.where(trail_local_cols[None, :] == jj2, trail, 0.0), axis=1
                )

            # Pivot value (diagonal element)
            pivot = tl.sum(tl.where(trail_local_rows == jj2, col_vals, 0.0), axis=0)

            # Scale column below diagonal
            scaled_col = tl.where(trail_local_rows > jj2, col_vals / pivot, col_vals)

            # Write L column
            trail = tl.where(
                (trail_local_rows[:, None] > jj2) & (trail_local_cols[None, :] == jj2),
                scaled_col[:, None],
                trail,
            )

            # Rank-1 update on trailing submatrix
            if USE_GATHER:
                u_row = tl.ravel(tl.gather(trail, row_jj_idx, axis=0))
            else:
                u_row = tl.sum(
                    tl.where(trail_local_rows[:, None] == jj2, trail, 0.0), axis=0
                )
            update_mask = (trail_local_rows[:, None] > jj2) & (
                trail_local_cols[None, :] > jj2
            )
            trail = tl.where(
                update_mask, trail - scaled_col[:, None] * u_row[None, :], trail
            )

    tl.store(LU + t_offsets, trail, mask=t_mask)


@libentry()
@triton.jit
def _lu_factor_trailing_update_no_pivot_kernel(
    LU,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    PANEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

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


def _blocked_lu_factor(input_contiguous, pivot):
    batch_shape = input_contiguous.shape[:-2]
    m, n = input_contiguous.shape[-2], input_contiguous.shape[-1]
    k = min(m, n)
    batch = input_contiguous.numel() // (m * n)

    lu = input_contiguous.clone()
    pivots = torch.empty(
        (*batch_shape, k), device=input_contiguous.device, dtype=torch.int32
    )

    block_m = triton.next_power_of_2(m)
    nw = 8 if input_contiguous.dtype == torch.float64 else 4

    with torch_device_fn.device(input_contiguous.device):
        if pivot:
            # Dynamic panel sizing: larger panels for small matrices reduce
            # kernel launch count (the dominant overhead for small sizes).
            if k <= 128:
                panel_size = 64
            elif k <= 256:
                panel_size = 32
            else:
                panel_size = _LU_FACTOR_PANEL
            panel_block = triton.next_power_of_2(panel_size)
            apply_left_in_panel = k <= _LU_FACTOR_TILE_N

            for k0 in range(0, k, panel_size):
                panel = min(panel_size, k - k0)
                trailing_n = n - k0 - panel
                trailing_m = m - k0 - panel
                use_fused_panel_solve = (
                    trailing_n > 0 and trailing_n <= _LU_FACTOR_FUSED_TILE_N
                )
                apply_left_this_panel = apply_left_in_panel and k0 > 0
                left_block = triton.next_power_of_2(k0) if apply_left_this_panel else 1

                # For small matrices (k <= 128) at the first panel, fuse
                # panel+swap+solve+trailing_update into a single kernel.
                if (
                    _LU_FACTOR_ENABLE_FUSED_PIVOT
                    and k0 == 0
                    and k <= 128
                    and trailing_m > 0
                    and trailing_n > 0
                    and trailing_n <= panel_size
                ):
                    trail_block_m = triton.next_power_of_2(trailing_m)
                    trail_block_n = triton.next_power_of_2(trailing_n)
                    # fp32: fuse next panel into the same kernel to save a launch.
                    # fp64: keep next panel as a separate kernel to avoid register
                    # pressure from extra fp64 iterations (8 warps → fewer regs/thread).
                    fuse_next = False
                    _lu_factor_fused_iter_pivot_kernel[(batch,)](
                        lu,
                        pivots,
                        k0,
                        m,
                        n,
                        k,
                        panel,
                        block_m,
                        panel_block,
                        trailing_n,
                        trail_block_m,
                        trail_block_n,
                        FUSE_NEXT_PANEL=fuse_next,
                        USE_GATHER=(nw == 8),
                        num_warps=nw,
                    )
                    if fuse_next:
                        break
                    else:
                        continue

                if use_fused_panel_solve:
                    _lu_factor_panel_swap_right_and_solve_kernel[(batch,)](
                        lu,
                        pivots,
                        k0,
                        m,
                        n,
                        k,
                        panel,
                        block_m,
                        panel_block,
                        _LU_FACTOR_FUSED_TILE_N,
                        left_block,
                        apply_left_this_panel,
                        num_warps=4,
                    )
                else:
                    _lu_factor_panel_kernel[(batch,)](
                        lu,
                        pivots,
                        k0,
                        m,
                        n,
                        k,
                        panel,
                        block_m,
                        panel_block,
                        left_block,
                        apply_left_this_panel,
                    )

                if trailing_n > 0 and not use_fused_panel_solve:
                    grid_combined = (
                        triton.cdiv(trailing_n, _LU_FACTOR_TILE_N),
                        batch,
                    )
                    _lu_factor_swap_right_and_solve_kernel[grid_combined](
                        lu,
                        pivots,
                        k0,
                        m,
                        n,
                        k,
                        panel,
                        panel_block,
                        _LU_FACTOR_TILE_N,
                        num_warps=4,
                    )

                if trailing_m > 0 and trailing_n > 0:
                    grid_update = (
                        triton.cdiv(trailing_m, _LU_FACTOR_TILE_M),
                        triton.cdiv(trailing_n, _LU_FACTOR_TILE_N),
                        batch,
                    )
                    _lu_factor_trailing_update_no_pivot_kernel[grid_update](
                        lu,
                        k0,
                        m,
                        n,
                        panel,
                        _LU_FACTOR_TILE_M,
                        _LU_FACTOR_TILE_N,
                        panel_block,
                        num_warps=4,
                    )

            # Final pass: apply all pivots to the left columns (L factors)
            if not apply_left_in_panel:
                for k0 in range(panel_size, k, panel_size):
                    panel = min(panel_size, k - k0)
                    grid_swap_left = (triton.cdiv(k0, _LU_FACTOR_TILE_N), batch)
                    _lu_factor_apply_panel_pivots_kernel[grid_swap_left](
                        lu,
                        pivots,
                        k0,
                        m,
                        n,
                        k,
                        panel,
                        0,
                        k0,
                        _LU_FACTOR_TILE_N,
                        num_warps=4,
                    )
        else:
            # No-pivot path: dynamic panel sizing
            # Small matrices benefit from larger panels (fewer kernel launches);
            # large matrices benefit from smaller panels (less O(PANEL²) work).
            if k <= 128:
                panel_size = 64
            elif k <= 256:
                panel_size = 32
            else:
                panel_size = _LU_FACTOR_PANEL
            panel_block = triton.next_power_of_2(panel_size)

            for k0 in range(0, k, panel_size):
                panel = min(panel_size, k - k0)
                trailing_n = n - k0 - panel
                trailing_m = m - k0 - panel

                use_fused_panel_solve = trailing_n > 0 and trailing_n <= panel_size

                # For small matrices (k <= 128), fuse panel+solve+trailing_update
                # into a single kernel to reduce kernel-launch overhead.
                if (
                    k <= 128
                    and trailing_m > 0
                    and trailing_n > 0
                    and trailing_n <= panel_size
                ):
                    trail_block_m = triton.next_power_of_2(trailing_m)
                    trail_block_n = triton.next_power_of_2(trailing_n)
                    # fp32: fuse next panel into the same kernel to save a launch.
                    # fp64: keep next panel as a separate kernel to avoid register
                    # pressure from extra fp64 iterations (8 warps → fewer regs/thread).
                    fuse_next = nw == 4
                    _lu_factor_fused_iter_no_pivot_kernel[(batch,)](
                        lu,
                        pivots,
                        k0,
                        m,
                        n,
                        k,
                        panel,
                        block_m,
                        panel_block,
                        trailing_n,
                        trail_block_m,
                        trail_block_n,
                        FUSE_NEXT_PANEL=fuse_next,
                        USE_GATHER=(nw == 8),
                        num_warps=nw,
                    )
                    if fuse_next:
                        break
                    else:
                        continue

                if use_fused_panel_solve:
                    _lu_factor_panel_solve_no_pivot_kernel[(batch,)](
                        lu,
                        pivots,
                        k0,
                        m,
                        n,
                        k,
                        panel,
                        block_m,
                        panel_block,
                        trailing_n,
                        num_warps=4,
                    )
                else:
                    _lu_factor_panel_no_pivot_kernel[(batch,)](
                        lu,
                        pivots,
                        k0,
                        m,
                        n,
                        k,
                        panel,
                        block_m,
                        panel_block,
                    )

                if trailing_n > 0 and not use_fused_panel_solve:
                    grid_solve = (triton.cdiv(trailing_n, _LU_FACTOR_TILE_N), batch)
                    _lu_factor_solve_block_row_no_pivot_kernel[grid_solve](
                        lu,
                        k0,
                        m,
                        n,
                        panel,
                        panel_block,
                        _LU_FACTOR_TILE_N,
                        num_warps=4,
                    )

                if trailing_m > 0 and trailing_n > 0:
                    grid_update = (
                        triton.cdiv(trailing_m, _LU_FACTOR_TILE_M),
                        triton.cdiv(trailing_n, _LU_FACTOR_TILE_N),
                        batch,
                    )
                    _lu_factor_trailing_update_no_pivot_kernel[grid_update](
                        lu,
                        k0,
                        m,
                        n,
                        panel,
                        _LU_FACTOR_TILE_M,
                        _LU_FACTOR_TILE_N,
                        panel_block,
                        num_warps=4,
                    )

    return LinalgLUFactorResult(lu, pivots)


def linalg_lu_factor(input, *, pivot=True):
    logger.debug("GEMS LINALG_LU_FACTOR")
    _linalg_lu_factor_check(input, pivot)

    input_contiguous = input.contiguous()

    if not _can_use_fast_triton(input_contiguous):
        return _blocked_lu_factor(input_contiguous, pivot)

    batch_shape = input_contiguous.shape[:-2]
    m, n = input_contiguous.shape[-2], input_contiguous.shape[-1]
    k = min(m, n)
    batch = input_contiguous.numel() // (m * n)

    lu = torch.empty_like(input_contiguous)
    pivots = torch.empty((*batch_shape, k), device=input.device, dtype=torch.int32)

    with torch_device_fn.device(input.device):
        _linalg_lu_factor_kernel[(batch,)](
            input_contiguous,
            lu,
            pivots,
            m,
            n,
            k,
            triton.next_power_of_2(m),
            triton.next_power_of_2(n),
            pivot,
            num_warps=4,
        )
    return LinalgLUFactorResult(lu, pivots)
