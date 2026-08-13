import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

LinalgLUFactorResult = namedtuple("LinalgLUFactorResult", ["LU", "pivots"])

_LU_FACTOR_BLOCK_MAX = (
    32  # simple Triton kernel for m,n <= 32 (fast compile, tl.range loop)
)
_LU_FACTOR_PANEL = 16  # panel width for blocked factorization
_LU_FACTOR_BLOCKED_M_MAX = 4096
_LU_FACTOR_BLOCKED_N_MAX = 4096
_LU_FACTOR_PANEL_BLOCK_M_MAX = (
    512  # max rows in a single Triton panel block (safe UB limit)
)
_LU_FACTOR_TILE_M = 64  # row tile size for trailing update (from main ops)
_LU_FACTOR_TILE_N = 128  # col tile size for solve / left-swap (from main ops)


# --- Local copies of kernels from main ops (flag_gems.ops.linalg_lu_factor) ---
# Defined here rather than imported to ensure fresh Ascend compilation
# (imported kernels may use cached GPU binaries that produce incorrect results).


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


# Ascend NPU can only handle grids with ≤ 40 blocks per launch.
# When the trailing update grid exceeds this, we split launches by
# column offset.  This constant is used by the launch helper only.
_ASCEND_MAX_GRID_BLOCKS = 40


@libentry()
@triton.jit
def _lu_factor_trailing_update_no_pivot_kernel(
    LU,
    k0,  # runtime: panel start column (avoids per-panel recompilation)
    m,  # runtime: total rows
    n,  # runtime: total cols
    row_offset,  # runtime: row offset within trailing submatrix (for grid splitting)
    col_offset,  # runtime: column offset within trailing submatrix (for grid splitting)
    PANEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Trailing matrix update: trailing -= L21 @ U12 via column-by-column rank-1 updates.

    Avoids tl.dot (buggy on Ascend) in favor of explicit outer-product
    accumulation over panel columns, matching the pattern used in
    _blocked_panel_kernel.

    k0, m, n, row_offset, and col_offset are runtime parameters so this
    kernel compiles once and is reused for all panels.  Grid splitting
    (when total blocks exceeds the Ascend 40-block limit) is handled by
    the caller via row_offset / col_offset.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    rows = k0 + PANEL + row_offset + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = k0 + PANEL + col_offset + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    tile_offsets = pid_b * m * n + rows[:, None] * n + cols[None, :]
    row_mask = rows[:, None] < m
    col_mask = cols[None, :] < n
    tile_mask = row_mask & col_mask
    tile = tl.load(LU + tile_offsets, mask=tile_mask, other=0.0).to(tl.float32)

    for jj in tl.range(0, PANEL):
        l_col_offsets = pid_b * m * n + rows * n + (k0 + jj)
        l_col = tl.load(LU + l_col_offsets, mask=rows < m, other=0.0).to(tl.float32)

        u_row_offsets = pid_b * m * n + (k0 + jj) * n + cols
        u_row = tl.load(LU + u_row_offsets, mask=cols < n, other=0.0).to(tl.float32)

        tile = tile - l_col[:, None] * u_row[None, :]

    tl.store(LU + tile_offsets, tile, mask=tile_mask)


@libentry()
@triton.jit
def _blocked_panel_kernel(
    LU_ptr,
    PIVOTS_ptr,
    k0,  # runtime: panel start column
    m,  # runtime: total rows
    n,  # runtime: total cols
    K,  # runtime: min(m, n) for PIVOTS offset
    panel: tl.constexpr,
    PIVOT: tl.constexpr,
    MAX_CHUNKS: tl.constexpr,  # triton.cdiv(m, TILE_M) for pivot column chunks
    TILE_M: tl.constexpr,  # tile size for both rows and column chunks
    MAX_M_TILES: tl.constexpr,  # triton.cdiv(m, TILE_M) for scale/update
    TILE_N: tl.constexpr,  # col tile size (panel rounded to pow2)
):
    """Unified single-block panel kernel supporting any remaining_m.

    Uses a single arange block for all tiled operations to minimize UB.
    One kernel launch per panel.  Only modifies panel columns [k0, k0+panel).
    """
    pid_b = tl.program_id(0)

    # Single arange tensors reused throughout the kernel
    idx_m = tl.arange(0, TILE_M)  # row index template
    idx_n = tl.arange(0, TILE_N)  # col index template

    panel_start = k0
    panel_last = k0 + panel

    for j_ind in tl.range(0, panel):
        j = k0 + j_ind

        # ================================================================
        # Step 1: Pivot search — load column j in TILE_M-element chunks
        # ================================================================
        if PIVOT:
            rows = j + idx_m
            row_mask = rows < m
            offs = pid_b * m * n + rows * n + j
            buf = tl.load(LU_ptr + offs, mask=row_mask, other=0.0).to(tl.float32)
            buf = tl.abs(buf)
            buf = tl.where(rows < j, -1.0, buf)
            buf = tl.where(rows < m, buf, -1.0)
            best_max = tl.max(buf, axis=0)
            best_row = tl.min(tl.where(buf == best_max, rows, m), axis=0)
            for chunk in tl.range(1, MAX_CHUNKS):
                rows = j + chunk * TILE_M + idx_m
                row_mask = rows < m
                offs = pid_b * m * n + rows * n + j
                buf = tl.load(LU_ptr + offs, mask=row_mask, other=0.0).to(tl.float32)
                buf = tl.abs(buf)
                buf = tl.where(rows < j, -1.0, buf)
                buf = tl.where(rows < m, buf, -1.0)
                cmax = tl.max(buf, axis=0)
                crow = tl.min(tl.where(buf == cmax, rows, m), axis=0)
                upd = cmax > best_max
                best_max = tl.where(upd, cmax, best_max)
                best_row = tl.where(upd, crow, best_row)
            pivot_row = best_row
            tl.store(PIVOTS_ptr + pid_b * K + j, pivot_row + 1)
        else:
            pivot_row = j
            tl.store(PIVOTS_ptr + pid_b * K + j, j + 1)

        # ================================================================
        # Step 2: Swap rows j and pivot_row within panel columns
        # ================================================================
        pcols = k0 + idx_n
        pcol_mask = (pcols >= panel_start) & (pcols < panel_last) & (pcols < n)

        off_j = pid_b * m * n + j * n + pcols
        off_p = pid_b * m * n + pivot_row * n + pcols
        buf_a = tl.load(LU_ptr + off_j, mask=pcol_mask, other=0.0).to(tl.float32)
        buf_b = tl.load(LU_ptr + off_p, mask=pcol_mask, other=0.0).to(tl.float32)
        tl.store(LU_ptr + off_j, tl.where(pivot_row != j, buf_b, buf_a), mask=pcol_mask)
        tl.store(LU_ptr + off_p, tl.where(pivot_row != j, buf_a, buf_b), mask=pcol_mask)

        # ================================================================
        # Step 3: Pivot value + per-tile scale & rank-1 update
        # ================================================================
        pivot_val = tl.load(LU_ptr + pid_b * m * n + j * n + j)

        # Trailing panel columns (j+1 .. panel_last-1)
        tcols = j + 1 + idx_n
        tn_mask = (tcols < panel_last) & (tcols < n) & (tcols > j)

        for mtile in tl.range(0, MAX_M_TILES):
            l_rows = j + 1 + mtile * TILE_M + idx_m
            l_mask = (l_rows < m) & (l_rows > j)

            # --- Scale: load L column slice, scale, store ---
            l_vals = tl.load(
                LU_ptr + pid_b * m * n + l_rows * n + j,
                mask=l_mask,
                other=0.0,
            ).to(tl.float32)
            l_vals = tl.where(l_mask, l_vals / pivot_val, l_vals)
            tl.store(
                LU_ptr + pid_b * m * n + l_rows * n + j,
                l_vals,
                mask=l_mask,
            )

            # --- Rank-1 update: tile -= L * U ---
            u_vals = tl.load(
                LU_ptr + pid_b * m * n + j * n + tcols,
                mask=tn_mask,
                other=0.0,
            ).to(tl.float32)

            tile_offs = pid_b * m * n + l_rows[:, None] * n + tcols[None, :]
            tile_mask = l_mask[:, None] & tn_mask[None, :]
            tile = tl.load(
                LU_ptr + tile_offs,
                mask=tile_mask,
                other=0.0,
            ).to(tl.float32)

            tile = tile - l_vals[:, None] * u_vals[None, :]
            tl.store(LU_ptr + tile_offs, tile, mask=tile_mask)


# --- General panel factorization kernels (handle arbitrary row counts) ---
# These replace _panel_lu_factor_pytorch with Triton kernels for edge cases
# where _blocked_panel_kernel cannot be used (remaining_m < 8 or > 512).


@libentry()
@triton.jit
def _panel_pivot_and_swap_kernel(
    LU,
    PIVOTS,
    j,  # runtime: absolute column index (avoids per-J recompilation)
    K: tl.constexpr,
    COL_START: tl.constexpr,
    NUM_COLS: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Fused pivot search + row swap in panel columns for one column.

    j is a runtime argument (not constexpr) so the kernel compiles once
    and is reused for all columns — avoids hundreds of recompilations.

    Step 1: Load column j in NUM_CHUNKS chunks, find pivot row via running max.
    Step 2: Swap row j with pivot_row in columns [COL_START, COL_START+NUM_COLS).
    """
    pid_b = tl.program_id(0)
    local_rows = tl.arange(0, BLOCK_M)

    # --- Step 1: Pivot search (load column j in chunks) ---
    rows = j + local_rows
    row_mask = rows < M
    offsets = pid_b * M * N + rows * N + j
    col = tl.load(LU + offsets, mask=row_mask, other=0.0).to(tl.float32)

    abs_col = tl.abs(col)
    abs_col = tl.where(rows < j, -1.0, abs_col)
    abs_col = tl.where(rows < M, abs_col, -1.0)

    best_max = tl.max(abs_col, axis=0)
    best_row = tl.min(tl.where(abs_col == best_max, rows, M), axis=0)

    for chunk in tl.range(1, NUM_CHUNKS):
        rows = j + chunk * BLOCK_M + local_rows
        row_mask = rows < M
        offsets = pid_b * M * N + rows * N + j
        col = tl.load(LU + offsets, mask=row_mask, other=0.0).to(tl.float32)

        abs_col = tl.abs(col)
        abs_col = tl.where(rows < j, -1.0, abs_col)
        abs_col = tl.where(rows < M, abs_col, -1.0)

        chunk_max = tl.max(abs_col, axis=0)
        chunk_row = tl.min(tl.where(abs_col == chunk_max, rows, M), axis=0)

        update = chunk_max > best_max
        best_max = tl.where(update, chunk_max, best_max)
        best_row = tl.where(update, chunk_row, best_row)

    pivot_row = best_row
    tl.store(PIVOTS + pid_b * K + j, pivot_row + 1)

    # --- Step 2: Swap rows j and pivot_row in panel columns ---
    cols = COL_START + tl.arange(0, NUM_COLS)
    col_mask = cols < COL_START + NUM_COLS

    off_j = pid_b * M * N + j * N + cols
    off_p = pid_b * M * N + pivot_row * N + cols

    row_j = tl.load(LU + off_j, mask=col_mask, other=0.0).to(tl.float32)
    row_p = tl.load(LU + off_p, mask=col_mask, other=0.0).to(tl.float32)

    tl.store(LU + off_j, row_p, mask=col_mask)
    tl.store(LU + off_p, row_j, mask=col_mask)


@libentry()
@triton.jit
def _panel_column_factor_kernel(
    LU,
    j,  # runtime: absolute column index (avoids per-J recompilation)
    panel_end,  # runtime: k0 + panel (avoids per-panel recompilation)
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Scale column j below diagonal and rank-1 update on trailing columns.

    Tiled over rows (pid_m) and trailing panel columns (pid_n).
    j and panel_end are runtime args to avoid recompilation.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    # Rows below diagonal j
    rows = j + 1 + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    # Trailing panel columns: j+1 .. panel_end-1
    cols = j + 1 + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < panel_end

    # --- Scale L column ---
    l_col_offsets = pid_b * M * N + rows * N + j
    l_col = tl.load(LU + l_col_offsets, mask=row_mask, other=0.0).to(tl.float32)

    pivot = tl.load(LU + pid_b * M * N + j * N + j)

    l_col = tl.where(rows > j, l_col / pivot, l_col)
    tl.store(LU + l_col_offsets, l_col, mask=row_mask)

    # --- Rank-1 update on trailing submatrix ---
    if j + 1 < panel_end:
        l_col_update = tl.where(rows > j, l_col, 0.0)

        u_row_offsets = pid_b * M * N + j * N + cols
        u_row = tl.load(LU + u_row_offsets, mask=col_mask, other=0.0).to(tl.float32)

        tile_offsets = pid_b * M * N + rows[:, None] * N + cols[None, :]
        tile_mask = row_mask[:, None] & col_mask[None, :]
        tile = tl.load(LU + tile_offsets, mask=tile_mask, other=0.0).to(tl.float32)

        tile = tile - l_col_update[:, None] * u_row[None, :]

        tl.store(LU + tile_offsets, tile, mask=tile_mask)


def _panel_lu_factor_general(lu, pivots, k0, m, n, k, panel):
    """General panel factorization using per-column Triton kernels.

    Handles cases that _blocked_panel_kernel cannot:
      - remaining_m < 8: Ascend compiler fails with tiny block sizes
      - remaining_m > 512: single-block UB overflow

    Uses column-by-column processing with tiled kernels.
    Row swaps within panel columns and left columns are done explicitly
    (unlike the fused kernel which defers them).
    """
    for jj in range(panel):
        j = k0 + jj

        # --- Fused pivot search + row swap in panel columns ---
        # Single-block kernel: finds pivot in column J (chunked load)
        # then swaps rows within the panel.  Left-column swap is
        # deferred to the caller's _lu_factor_apply_panel_pivots_kernel.
        remaining = m - j
        if remaining <= _LU_FACTOR_PANEL_BLOCK_M_MAX:
            nchunks = 1
        else:
            nchunks = triton.cdiv(remaining, _LU_FACTOR_PANEL_BLOCK_M_MAX)
        block_m = max(
            min(triton.next_power_of_2(remaining), _LU_FACTOR_PANEL_BLOCK_M_MAX), 8
        )

        with torch_device_fn.device(lu.device):
            _panel_pivot_and_swap_kernel[(1,)](
                lu,
                pivots,
                j=j,
                K=k,
                COL_START=k0,
                NUM_COLS=panel,
                M=m,
                N=n,
                NUM_CHUNKS=nchunks,
                BLOCK_M=block_m,
            )

        # --- Scale column j and rank-1 update ---
        if j + 1 < m:
            panel_end = min(k0 + panel, n)
            if j + 1 < panel_end:
                grid_scale = (
                    triton.cdiv(m - j - 1, _LU_FACTOR_TILE_M),
                    triton.cdiv(panel_end - j - 1, _LU_FACTOR_TILE_N),
                    1,
                )
            else:
                grid_scale = (triton.cdiv(m - j - 1, _LU_FACTOR_TILE_M), 1, 1)

            with torch_device_fn.device(lu.device):
                _panel_column_factor_kernel[grid_scale](
                    lu,
                    j=j,
                    panel_end=panel_end,
                    M=m,
                    N=n,
                    BLOCK_M=_LU_FACTOR_TILE_M,
                    BLOCK_N=_LU_FACTOR_TILE_N,
                )

    return lu, pivots


def _panel_lu_factor_triton(lu, pivots, k0, m, n, k, panel):
    """Panel factorization using the unified single-block Triton kernel.

    Uses chunked pivot search + tiled scale/update to handle ANY
    remaining_m in one kernel launch, without UB overflow.

    Only falls back to the per-column general path for tiny matrices
    (remaining_m < 8) that trigger Ascend compiler issues.
    """
    trailing_m = m - k0

    if trailing_m >= 8:
        tile_m = _LU_FACTOR_TILE_M
        tile_n = min(triton.next_power_of_2(panel), 32)
        max_chunks = triton.cdiv(m, tile_m)
        max_m_tiles = triton.cdiv(m, tile_m)

        with torch_device_fn.device(lu.device):
            _blocked_panel_kernel[(1,)](
                lu,
                pivots,
                k0=k0,
                m=m,
                n=n,
                K=k,
                panel=panel,
                PIVOT=True,
                MAX_CHUNKS=max_chunks,
                TILE_M=tile_m,
                MAX_M_TILES=max_m_tiles,
                TILE_N=tile_n,
            )
    else:
        _panel_lu_factor_general(lu, pivots, k0, m, n, k, panel)


def _linalg_lu_factor_check(input, pivot):
    if input.dim() < 2:
        raise RuntimeError(
            "torch.linalg.lu_factor: Expected input to have at least 2 dimensions, "
            f"got {input.dim()}"
        )
    if input.dtype != torch.float32:
        raise NotImplementedError(
            "FlagGems linalg_lu_factor currently supports float32 only, "
            f"got {input.dtype}"
        )
    m, n = input.shape[-2], input.shape[-1]
    if m == 0 or n == 0:
        raise NotImplementedError(
            "FlagGems linalg_lu_factor currently does not support empty matrices"
        )
    if pivot not in (True, False):
        raise TypeError(f"pivot must be a bool, got {type(pivot)}")
    if not pivot and input.device.type != "cuda":
        raise NotImplementedError(
            "FlagGems linalg_lu_factor: pivot=False is only supported on CUDA devices, "
            f"got device={input.device.type}"
        )


def _can_use_triton(input):
    """Check if the simple Triton kernel can be used.

    Requires 8 <= m,n <= _LU_FACTOR_BLOCK_MAX. Below 8, the Ascend
    compiler fails with 'strides must not be zero' for tiny block sizes.
    """
    m, n = input.shape[-2], input.shape[-1]
    return 8 <= m <= _LU_FACTOR_BLOCK_MAX and 8 <= n <= _LU_FACTOR_BLOCK_MAX


def _can_use_blocked_triton(input):
    m, n = input.shape[-2], input.shape[-1]
    return m <= _LU_FACTOR_BLOCKED_M_MAX and n <= _LU_FACTOR_BLOCKED_N_MAX


def _blocked_lu_factor(input_contiguous):
    """Blocked LU factorization using only Triton kernels (pivot=True always).

    - Panel factorization: ascend-specific single-block Triton kernel
      (_blocked_panel_kernel) for fast path (8..512 rows), or per-column
      Triton kernels (_panel_lu_factor_general) for edge cases.
    - Pivot application + forward substitution: Triton kernel
      (_lu_factor_swap_right_and_solve_kernel).
    - Trailing matrix update: Triton kernel using rank-1 updates
      (_lu_factor_trailing_update_no_pivot_kernel).
    - Deferred left-column pivot application: Triton kernel
      (_lu_factor_apply_panel_pivots_kernel).
    """
    batch_shape = input_contiguous.shape[:-2]
    m, n = input_contiguous.shape[-2], input_contiguous.shape[-1]
    k = min(m, n)
    batch = input_contiguous.numel() // (m * n)

    lu = input_contiguous.clone()
    pivots = torch.empty(
        (*batch_shape, k), device=input_contiguous.device, dtype=torch.int32
    )

    # Flat views for batch slicing — slice along dim 0 to create sub-batches
    # that stay within the Ascend 40-block-per-launch limit.
    lu_flat = lu.reshape(-1, m, n)
    pivots_flat = pivots.reshape(-1, k)

    with torch_device_fn.device(input_contiguous.device):
        for k0 in range(0, k, _LU_FACTOR_PANEL):
            panel = min(_LU_FACTOR_PANEL, k - k0)
            trailing_n = n - k0 - panel
            trailing_m = m - k0 - panel

            # Phase 1: Panel factorization with grid-based batch.
            # Splits batch into chunks of ≤ _ASCEND_MAX_GRID_BLOCKS elements
            # and launches the panel kernel once per chunk (instead of once
            # per batch element), giving orders-of-magnitude speedup.
            trailing_m_k = m - k0
            for b_start in range(0, batch, _ASCEND_MAX_GRID_BLOCKS):
                b_end = min(b_start + _ASCEND_MAX_GRID_BLOCKS, batch)
                b_chunk = b_end - b_start
                if trailing_m_k >= 8:
                    tile_m = _LU_FACTOR_TILE_M
                    tile_n = min(triton.next_power_of_2(panel), 32)
                    max_chunks = triton.cdiv(m, tile_m)
                    max_m_tiles = triton.cdiv(m, tile_m)
                    _blocked_panel_kernel[(b_chunk,)](
                        lu_flat[b_start:b_end],
                        pivots_flat[b_start:b_end],
                        k0=k0,
                        m=m,
                        n=n,
                        K=k,
                        panel=panel,
                        PIVOT=True,
                        MAX_CHUNKS=max_chunks,
                        TILE_M=tile_m,
                        MAX_M_TILES=max_m_tiles,
                        TILE_N=tile_n,
                    )
                else:
                    for b in range(b_start, b_end):
                        _panel_lu_factor_general(
                            lu_flat[b],
                            pivots_flat[b],
                            k0,
                            m,
                            n,
                            k,
                            panel,
                        )

            # Phase 2: Right-block swap + solve.
            # Split batch so grid_n × batch_chunk ≤ _ASCEND_MAX_GRID_BLOCKS.
            if trailing_n > 0:
                gn = triton.cdiv(trailing_n, _LU_FACTOR_TILE_N)
                max_batch_per_launch = max(1, _ASCEND_MAX_GRID_BLOCKS // gn)
                for b_start in range(0, batch, max_batch_per_launch):
                    b_end = min(b_start + max_batch_per_launch, batch)
                    b_chunk = b_end - b_start
                    _lu_factor_swap_right_and_solve_kernel[(gn, b_chunk)](
                        lu_flat[b_start:b_end],
                        pivots_flat[b_start:b_end],
                        k0,
                        m,
                        n,
                        k,
                        panel,
                        BLOCK_B=triton.next_power_of_2(panel),
                        BLOCK_N=_LU_FACTOR_TILE_N,
                        num_warps=4,
                    )

            # Phase 3: Trailing matrix update via rank-1 updates (avoids tl.dot).
            # Split on spatial dimensions (gm/gn) AND batch to stay ≤ 40 blocks.
            if trailing_m > 0 and trailing_n > 0:
                gm = triton.cdiv(trailing_m, _LU_FACTOR_TILE_M)
                gn = triton.cdiv(trailing_n, _LU_FACTOR_TILE_N)
                # Split the larger spatial dimension first, then split batch
                if gm >= gn:
                    # gm is the larger dimension; cap gm per launch, split batch separately
                    max_gm = max(1, _ASCEND_MAX_GRID_BLOCKS // gn)
                    for gm_start in range(0, gm, max_gm):
                        gm_chunk = min(max_gm, gm - gm_start)
                        row_off = gm_start * _LU_FACTOR_TILE_M
                        spat_blocks = gm_chunk * gn
                        max_b = max(1, _ASCEND_MAX_GRID_BLOCKS // spat_blocks)
                        for b_start in range(0, batch, max_b):
                            b_end = min(b_start + max_b, batch)
                            b_chunk = b_end - b_start
                            _lu_factor_trailing_update_no_pivot_kernel[
                                (gm_chunk, gn, b_chunk)
                            ](
                                lu_flat[b_start:b_end],
                                k0=k0,
                                m=m,
                                n=n,
                                row_offset=row_off,
                                col_offset=0,
                                PANEL=panel,
                                BLOCK_M=_LU_FACTOR_TILE_M,
                                BLOCK_N=_LU_FACTOR_TILE_N,
                                num_warps=4,
                            )
                else:
                    # gn is the larger dimension; cap gn per launch, split batch separately
                    max_gn = max(1, _ASCEND_MAX_GRID_BLOCKS // gm)
                    for gn_start in range(0, gn, max_gn):
                        gn_chunk = min(max_gn, gn - gn_start)
                        col_off = gn_start * _LU_FACTOR_TILE_N
                        spat_blocks = gm * gn_chunk
                        max_b = max(1, _ASCEND_MAX_GRID_BLOCKS // spat_blocks)
                        for b_start in range(0, batch, max_b):
                            b_end = min(b_start + max_b, batch)
                            b_chunk = b_end - b_start
                            _lu_factor_trailing_update_no_pivot_kernel[
                                (gm, gn_chunk, b_chunk)
                            ](
                                lu_flat[b_start:b_end],
                                k0=k0,
                                m=m,
                                n=n,
                                row_offset=0,
                                col_offset=col_off,
                                PANEL=panel,
                                BLOCK_M=_LU_FACTOR_TILE_M,
                                BLOCK_N=_LU_FACTOR_TILE_N,
                                num_warps=4,
                            )

        # Phase 4: Apply deferred panel pivots to left columns.
        # Split batch so grid_n × batch_chunk ≤ _ASCEND_MAX_GRID_BLOCKS.
        for k0 in range(_LU_FACTOR_PANEL, k, _LU_FACTOR_PANEL):
            panel = min(_LU_FACTOR_PANEL, k - k0)
            gn = triton.cdiv(k0, _LU_FACTOR_TILE_N)
            max_batch_per_launch = max(1, _ASCEND_MAX_GRID_BLOCKS // gn)
            for b_start in range(0, batch, max_batch_per_launch):
                b_end = min(b_start + max_batch_per_launch, batch)
                b_chunk = b_end - b_start
                _lu_factor_apply_panel_pivots_kernel[(gn, b_chunk)](
                    lu_flat[b_start:b_end],
                    pivots_flat[b_start:b_end],
                    k0,
                    m,
                    n,
                    k,
                    panel,
                    COL_START=0,
                    NUM_COLS=k0,
                    BLOCK_N=_LU_FACTOR_TILE_N,
                    num_warps=4,
                )

    return LinalgLUFactorResult(lu, pivots)


def linalg_lu_factor(input, *, pivot=True):
    logger.debug("GEMS_ASCEND LINALG_LU_FACTOR")
    _linalg_lu_factor_check(input, pivot)

    input_contiguous = input.contiguous()

    if not _can_use_triton(input_contiguous):
        if _can_use_blocked_triton(input_contiguous):
            logger.debug("GEMS_ASCEND LINALG_LU_FACTOR blocked Triton path")
            return _blocked_lu_factor(input_contiguous)
        raise NotImplementedError(
            "FlagGems linalg_lu_factor Triton large-shape path is not available "
            "for this input"
        )

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
            True,
        )
    return LinalgLUFactorResult(lu, pivots)


def _resolve_linalg_lu_factor_out_args(LU, pivots, out):
    if out is not None:
        if LU is not None or pivots is not None:
            raise TypeError("linalg_lu_factor(): out and LU/pivots cannot both be set")
        if len(out) != 2:
            raise TypeError(
                f"linalg_lu_factor(): out must be a tuple of 2 tensors, got {len(out)}"
            )
        return out
    if LU is None or pivots is None:
        raise TypeError(
            "linalg_lu_factor(): LU and pivots must both be provided for out variant"
        )
    return LU, pivots


def linalg_lu_factor_out(input, *, pivot=True, LU=None, pivots=None, out=None):
    logger.debug("GEMS_ASCEND LINALG_LU_FACTOR.OUT")
    lu_out, pivots_out = _resolve_linalg_lu_factor_out_args(LU, pivots, out)
    lu, piv = linalg_lu_factor(input, pivot=pivot)

    lu_out.resize_(lu.shape)
    pivots_out.resize_(piv.shape)
    lu_out.copy_(lu)
    pivots_out.copy_(piv)
    return LinalgLUFactorResult(lu_out, pivots_out)
