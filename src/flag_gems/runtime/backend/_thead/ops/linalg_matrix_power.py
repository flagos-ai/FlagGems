# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""linalg_matrix_power override for the thead (PPU) backend.

Two thead quirks (see the generic module's docstring for the background):
  * grid spin-barrier kernels hang here — the fused Tier-3 path and the
    barrier-based parallel LU are not used; 65 <= M <= 256 runs the host
    loop and large-M inverses use the private barrier-free LU below;
  * the fp64 tl.dot is nondeterministic once operand magnitudes grow, so
    negative powers run in df64 (pure fp32 arithmetic, M <= 64) or in an
    fp64 chain whose every tl.dot is a (16,16,16) tile (M > 64).

Everything else — validation, positive powers, the df64 kernels — is
imported from the generic module.
"""

import importlib

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.ops.linalg_matrix_power import (
    _LU_PAR_PANEL,
    _LU_PAR_TILE_M,
    _LU_PAR_TILE_N,
    SINGLE_TILE_MAX,
    TILED_MAX,
    TRITON_THRESHOLD,
    _df64_power_scale,
    _df64_recombine,
    _eye_like,
    _info_has_error,
    _inverse,
    _lu_apply_left_par,
    _lu_swap_right_solve_par,
    _matmul,
    _matrix_power_df64_pair,
    _pivots_to_perm_gpu,
    _single_tile_kernel,
    _split_fp64_pair,
    _trsm_solve_register,
    _trsm_update_register,
)
from flag_gems.utils import libentry

# NB: bind the *module* explicitly - ``import flag_gems.ops.linalg_matrix_power
# as _generic`` would resolve through the package attribute, which the
# re-exported ``linalg_matrix_power`` function shadows.
_generic = importlib.import_module("flag_gems.ops.linalg_matrix_power")


_DF64_MAX_M = 64  # in-register df64 chain limit (see the module docstring)


@triton.jit
def _trsm_solve_d16(
    A_ptr,
    B_ptr,
    INV_ptr,
    pid,
    stride_a_n,
    stride_b_k,
    blk_start,
    blk_end,
    blk_sz,
    a_cols,
    xr,
    col_offs,
    col_mask,
    BLOCK_SIZE: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
):
    """D16 solve variant of _trsm_kernel's diagonal-block phase (thead only,
    see the platform-policy table): each already-solved row is re-read from
    global memory and every step is fenced with a debug_barrier — the same
    fenced pattern as _thead_lu_panel_serial."""
    # thead: the register-chain variant below races on this platform
    # (its cross-thread register dependency relies on the tl.sum sync
    # only — empirically nondeterministic here), so the D16 mode
    # re-reads each already-solved row from global memory and fences
    # every row with a debug_barrier — the same fenced pattern as
    # _thead_lu_panel_serial.
    for r_idx in range(blk_sz):
        row = blk_end - 1 - r_idx if UPPER else blk_start + r_idx
        row_rel = row - blk_start

        # Row of A restricted to the block's triangle.
        a_row = tl.load(
            A_ptr + row * stride_a_n + blk_start + a_cols,
            mask=a_cols < blk_sz,
            other=0.0,
        )
        if UPPER:
            a_row = tl.where(a_cols > row_rel, a_row, 0.0)
        else:
            a_row = tl.where(a_cols < row_rel, a_row, 0.0)

        # Already-solved block rows from global memory (unsolved rows
        # are masked out by a_row).
        xs = tl.load(
            B_ptr + (blk_start + xr) * stride_b_k + col_offs[None, :],
            mask=(xr < blk_sz) & col_mask[None, :],
            other=0.0,
        ).to(A_ptr.dtype.element_ty)
        x_sum = tl.sum(a_row[:, None] * xs, axis=0)

        x_vals = (
            tl.load(
                B_ptr + row * stride_b_k + col_offs,
                mask=col_mask,
                other=0.0,
            )
            - x_sum
        )
        if not UNIT:
            inv_d = tl.load(INV_ptr + pid * BLOCK_SIZE + row_rel)
            x_vals *= inv_d
        tl.store(B_ptr + row * stride_b_k + col_offs, x_vals, mask=col_mask)
        # fence this row's store before the next row's xs reload.
        tl.debug_barrier()


@triton.jit
def _trsm_update_d16(
    A_ptr,
    B_ptr,
    stride_a_n,
    stride_b_k,
    blk_start,
    blk_end,
    blk_sz,
    M_REM,
    rem_s,
    bound,
    col_offs,
    col_mask,
):
    """D16 update variant of _trsm_kernel (thead only): every fp64 dot is
    capped at a (16,16,16) tile — wider fp64 dots nondeterministically return
    garbage on this platform (see the thead policy row)."""
    # thead: keep every fp64 dot at (16,16,16) — wider fp64 dots
    # nondeterministically return garbage on this platform (see
    # the thead policy row).  The solved block's X panel was stored to
    # B above and is re-read here in 16-row x 16-col chunks.
    rr16 = tl.arange(0, 16)
    # fence the serial chain's stores before the cross-thread
    # x16 reloads.
    tl.debug_barrier()
    for m_start in range(0, M_REM, 16):
        rm = rem_s + m_start + rr16
        mask_m = rm < bound
        b_base = B_ptr + rm[:, None] * stride_b_k + col_offs[None, :]
        b_curr = tl.load(
            b_base,
            mask=mask_m[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float64)
        for kc in range(0, blk_sz, 16):
            kr = blk_start + kc + rr16
            km = kr < blk_end
            a16 = tl.load(
                A_ptr + rm[:, None] * stride_a_n + kr[None, :],
                mask=mask_m[:, None] & km[None, :],
                other=0.0,
            ).to(tl.float64)
            x16 = tl.load(
                B_ptr + kr[:, None] * stride_b_k + col_offs[None, :],
                mask=km[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float64)
            b_curr -= tl.dot(a16, x16, allow_tf32=False)
        tl.store(
            b_base,
            b_curr.to(B_ptr.dtype.element_ty),
            mask=mask_m[:, None] & col_mask[None, :],
        )


@libentry()
@triton.jit
def _trsm_kernel(
    A_ptr,
    B_ptr,
    INV_ptr,
    N,
    K,
    stride_a_n,
    stride_b_k,
    BLOCK_SIZE: tl.constexpr,
    K_SLICE: tl.constexpr,
    BM: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
    USE_FP64: tl.constexpr,
    FIX_FP64: tl.constexpr = False,
    G_BM: tl.constexpr = 1,
    D16: tl.constexpr = False,
):
    """Blocked triangular solve A X = B in place (B <- X), one program per
    K_SLICE-column group of the RHS.

    Rows are processed in BLOCK_SIZE blocks.  The diagonal block is solved by
    serial forward/backward substitution (row-by-row, parallel across the
    K_SLICE columns), then the remaining rows are updated with a tl.dot gemm —
    every data dependency stays within a single program, so the kernel is
    barrier-free apart from the D16 (thead) variant's cross-thread fences in
    the per-block helpers above.  This replaces the (batch, n)-grid scalar
    substitution kernels, whose one-program-per-column serial O(n^2) loop was
    the dominant cost of the negative-power path (~77 ms per solve at n=1024
    vs ~1 ms here).
    """
    pid = tl.program_id(0)
    col_start = pid * K_SLICE
    if col_start >= K:
        return

    num_blocks = tl.cdiv(N, BLOCK_SIZE)

    a_cols = tl.arange(0, BLOCK_SIZE)
    x_rows = tl.arange(0, BLOCK_SIZE)
    x_kcols = tl.arange(0, K_SLICE)
    xr = tl.broadcast_to(x_rows[:, None], (BLOCK_SIZE, K_SLICE))
    col_offs = col_start + x_kcols
    col_mask = col_offs < K
    rr = tl.arange(0, BM)

    for block_idx in range(num_blocks):
        if D16:
            # Cross-thread dependency through global memory inside this CTA
            # (the previous block's update stores are read by this block's
            # solve loads); fence each block iteration (see
            # _thead_lu_panel_serial for the same race without the barrier).
            tl.debug_barrier()
        bk = block_idx if not UPPER else num_blocks - 1 - block_idx
        blk_start = bk * BLOCK_SIZE
        blk_end = tl.minimum(blk_start + BLOCK_SIZE, N)
        blk_sz = blk_end - blk_start

        # ═══ Diagonal block: serial substitution over rows ═══
        # Pre-compute diagonal reciprocals (division out of the serial chain).
        if not UNIT:
            diag_vals = tl.load(
                A_ptr + (blk_start + a_cols) * stride_a_n + (blk_start + a_cols),
                mask=a_cols < blk_sz,
                other=1.0,
            )
            tl.store(
                INV_ptr + pid * BLOCK_SIZE + a_cols,
                1.0 / diag_vals,
                mask=a_cols < blk_sz,
            )

        # Serial solve of the diagonal block — two variants of the same
        # substitution, selected by D16 (thead, see the policy table).
        if D16:
            _trsm_solve_d16(
                A_ptr,
                B_ptr,
                INV_ptr,
                pid,
                stride_a_n,
                stride_b_k,
                blk_start,
                blk_end,
                blk_sz,
                a_cols,
                xr,
                col_offs,
                col_mask,
                BLOCK_SIZE,
                UPPER,
                UNIT,
            )
        else:
            x_all = _trsm_solve_register(
                A_ptr,
                B_ptr,
                INV_ptr,
                pid,
                stride_a_n,
                stride_b_k,
                blk_start,
                blk_end,
                blk_sz,
                a_cols,
                xr,
                col_offs,
                col_mask,
                BLOCK_SIZE,
                UPPER,
                UNIT,
            )

        if D16:
            # fence the serial chain's stores before the D16 update re-reads
            # the solved rows from B across threads (see
            # _thead_lu_panel_serial for the same race without the barrier).
            tl.debug_barrier()

        # ═══ Update: B[rest, kslice] -= A[rest, blk] @ X[blk, kslice] ═══
        need_update = tl.where(UPPER, bk > 0, blk_end < N)
        if need_update:
            M_REM = tl.where(UPPER, blk_start, N - blk_end)
            rem_s = tl.where(UPPER, 0, blk_end)
            bound = tl.where(UPPER, blk_start, N)
            if D16:
                _trsm_update_d16(
                    A_ptr,
                    B_ptr,
                    stride_a_n,
                    stride_b_k,
                    blk_start,
                    blk_end,
                    blk_sz,
                    M_REM,
                    rem_s,
                    bound,
                    col_offs,
                    col_mask,
                )
            else:
                _trsm_update_register(
                    A_ptr,
                    B_ptr,
                    stride_a_n,
                    stride_b_k,
                    blk_start,
                    blk_end,
                    blk_sz,
                    M_REM,
                    rem_s,
                    bound,
                    x_all,
                    a_cols,
                    rr,
                    col_offs,
                    col_mask,
                    BM,
                    K_SLICE,
                    USE_FP64,
                    FIX_FP64,
                    G_BM,
                )


def _trsm_solve_2d(A_tri, B, upper: bool, unitriangular: bool, d16: bool = False):
    """Solve a 2-D triangular system A_tri X = B in place, with ``_trsm_kernel``.

    Self-contained in this operator (no dependency on the shared triangular-solve
    op, whose persistent N>512 path also reuses one barrier counter across its
    batch loop and races on batched inputs).

    ``d16`` (thead large-M path): every fp64 update dot is restricted to a
    (16,16,16) tile — the only fp64-dot shape that stays deterministic on
    thead once operand magnitudes grow — so the solve runs with 16-row update
    sub-tiles and a 16-column K_SLICE.
    """
    n = A_tri.shape[0]
    k = B.shape[1]
    # The update gemm runs in fp64 (USE_FP64 is always true off-iluvatar), so on
    # MetaX it always hits the fp64 dot path.  That backend cannot lower the
    # fp64 dot with N < 16 and scrambles its output rows, so it uses a wider
    # K_SLICE and un-scrambles the result — for fp32- and fp64-stored factors
    # alike.  Everywhere else the narrower slice keeps the iluvatar elementwise
    # fallback small and the fp64 dot legal (NVIDIA handles N = 8 fine).
    fix_fp64 = False
    K_SLICE = 16 if (fix_fp64 or d16) else 8
    BM = 16 if d16 else 128
    # Software-pipeline depth for the update gemm.  num_stages is a pipelining
    # hint only — it never changes the math — but it decides whether the kernel
    # fits the per-block shared memory: at the 3-stage default the fp64-stored
    # factor path needs ~73.7 KB, over the 64 KB limit on small-smem backends.
    if d16:
        num_stages = 2
    elif fix_fp64:
        num_stages = 1
    elif False:
        num_stages = 2
    else:
        num_stages = 3
    num_kslices = (k + K_SLICE - 1) // K_SLICE
    if unitriangular:
        # INV_ptr is only dereferenced when UNIT is false — pass B as a dummy.
        inv = B
        unit_flag = True
    else:
        inv = torch.zeros(num_kslices * 32, dtype=A_tri.dtype, device=A_tri.device)
        unit_flag = False
    _trsm_kernel[(num_kslices,)](
        A_tri,
        B,
        inv,
        n,
        k,
        A_tri.stride(0),
        B.stride(0),
        32,
        K_SLICE,
        BM,
        upper,
        unit_flag,
        True,
        fix_fp64,
        8,  # G_BM = BM // 16 = 128 // 16
        D16=d16,
        num_warps=4,
        # fp64 dot operands are staged through shared memory (8 bytes/element).
        # On 64 KB-shared-memory backends the pipeline depth must drop below
        # the 3-stage default so the 128x32 fp64 update tile fits (see the
        # num_stages computation above); NVIDIA keeps the full pipeline.
        num_stages=num_stages,
    )
    return B


# ===========================================================================
# Solve / inverse assembly (host): LU factors + row permutation -> solve
# (linalg_lu_solve), fp64-accumulation Newton refinement of f32 inverses
# (_newton_refine), and the inverse entry point (_inverse, below the
# large-M LU section it dispatches to).
# ===========================================================================


@triton.jit
def _thead_fp64_mm16_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
):
    """Tiled fp64 C = A @ B whose every tl.dot is capped at a (32,32,32) tile
    (4 warps) — thead's fp64 tl.dot with wider tiles or more warps
    sporadically returns garbage once operand magnitudes grow (see the thead
    policy row; 4-warp dots up to (64,64,64) were bitwise-deterministic in the
    envelope probe).  The thead M > 64 negative-power chain routes every fp64
    matmul through this kernel.  Grid: (output-tiles, batch)."""
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    grid_n = tl.cdiv(N, 32)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    rm = pid_m * 32 + tl.arange(0, 32)
    rn = pid_n * 32 + tl.arange(0, 32)
    rk = tl.arange(0, 32)

    a_base = A + pid_b * stride_ab
    b_base = B + pid_b * stride_bb
    c_base = C + pid_b * stride_cb

    acc = tl.zeros((32, 32), dtype=tl.float64)
    for k0 in tl.range(0, K, 32):
        kr = k0 + rk
        a = tl.load(
            a_base + rm[:, None] * stride_am + kr[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (kr[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_base + kr[:, None] * stride_bk + rn[None, :] * stride_bn,
            mask=(kr[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a.to(tl.float64), b.to(tl.float64), allow_tf32=False)
    cmask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(
        c_base + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc.to(C.dtype.element_ty),
        mask=cmask,
    )


def _thead_fp64_mm16(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """fp64 matmul for the thead large-M deterministic path — every tl.dot is
    a (16,16,16) fp64 tile (see ``_thead_fp64_mm16_kernel``)."""
    *_, M, K = A.shape
    *_, K2, N = B.shape
    assert K == K2
    A2 = A.reshape(-1, M, K)
    B2 = B.reshape(-1, K, N)
    batch = A2.shape[0]
    C = torch.empty(batch, M, N, dtype=A.dtype, device=A.device)
    grid = (triton.cdiv(M, 32) * triton.cdiv(N, 32), batch)
    _thead_fp64_mm16_kernel[grid](
        A2,
        B2,
        C,
        M,
        N,
        K,
        A2.stride(0),
        A2.stride(1),
        A2.stride(2),
        B2.stride(0),
        B2.stride(1),
        B2.stride(2),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        num_warps=4,
    )
    return C.reshape(A.shape[:-2] + (M, N))


@triton.jit
def _thead_lu_update16_par(
    LU_ptr,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    PANEL: tl.constexpr,
):
    """A[K0+PANEL:M, K0+PANEL:N] -= L @ U for the thead large-M LU — the
    trailing update of _lu_trailing_update_par with fp64 tl.dots capped at
    (32,32,32) tiles (thead's fp64 dots with wider tiles / more warps
    sporadically return garbage once magnitudes grow; 4-warp dots up to
    (64,64,64) were bitwise-deterministic in the envelope probe).  Grid:
    (row-tiles, col-tiles, batch)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    rows = K0 + PANEL + pid_m * 32 + tl.arange(0, 32)
    cols = K0 + PANEL + pid_n * 32 + tl.arange(0, 32)
    kr32 = tl.arange(0, 32)
    base = pid_b * M * N
    tile_offs = base + rows[:, None] * N + cols[None, :]
    tile_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tile = tl.load(LU_ptr + tile_offs, mask=tile_mask, other=0.0).to(
        LU_ptr.dtype.element_ty
    )
    acc = tl.zeros((32, 32), dtype=tl.float64)
    for kb in range(0, PANEL, 32):
        kr = K0 + kb + kr32
        k_mask = kr < K0 + PANEL
        l32 = tl.load(
            LU_ptr + base + rows[:, None] * N + kr[None, :],
            mask=(rows[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        u32 = tl.load(
            LU_ptr + base + kr[:, None] * N + cols[None, :],
            mask=k_mask[:, None] & (cols[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(l32.to(tl.float64), u32.to(tl.float64), allow_tf32=False)
    tl.store(
        LU_ptr + tile_offs,
        (tile - acc.to(LU_ptr.dtype.element_ty)),
        mask=tile_mask,
    )


@triton.jit
def _thead_lu_panel_serial(
    LU_ptr,
    pivots_ptr,
    info_ptr,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    PANEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    COL_BLOCK: tl.constexpr,
    ROW_TILE: tl.constexpr,
):
    """Barrier-free panel LU factorization for thead (one program per matrix;
    the R-group barrier kernel _lu_panel_par deadlocks there).  Serial
    right-looking elimination over the PANEL columns of [K0, M) x [K0, K0+PANEL):
    per column — pivot search over the column vector in registers, physical row
    swap over the panel columns, scale the below-diagonal entries, then the
    rank-1 update over ROW_TILE-row chunks (register-bounded).  All scalars /
    reductions stay inside one CTA, so no grid barrier is needed and the result
    is deterministic.  Stores 0-based IPIV pivots."""
    pid = tl.program_id(0)
    base = pid * M * N
    allr = tl.arange(0, BLOCK_M)
    allc = tl.arange(0, COL_BLOCK)
    rt = tl.arange(0, ROW_TILE)
    info_val = 0
    for jj in tl.range(0, PANEL):
        j = K0 + jj
        # 1. current column j (rows >= j) — pivot search in registers.
        col = tl.load(
            LU_ptr + base + (j + allr) * N + j, mask=(j + allr) < M, other=0.0
        )
        ac = tl.where((j + allr) < M, tl.abs(col), -1.0)
        pmax = tl.max(ac, axis=0)
        p = j + tl.min(tl.where(ac == pmax, allr, BLOCK_M), axis=0)
        pivot = tl.load(LU_ptr + base + p * N + j)
        if pmax != pmax or pmax == 0.0:
            if info_val == 0:
                info_val = j + 1
        tl.store(pivots_ptr + pid * M + j, p)
        # 2. swap rows j and p over the panel columns.
        if p != j:
            rj = tl.load(
                LU_ptr + base + j * N + K0 + allc, mask=allc < PANEL, other=0.0
            )
            rp = tl.load(
                LU_ptr + base + p * N + K0 + allc, mask=allc < PANEL, other=0.0
            )
            tl.store(LU_ptr + base + j * N + K0 + allc, rp, mask=allc < PANEL)
            tl.store(LU_ptr + base + p * N + K0 + allc, rj, mask=allc < PANEL)
        # Cross-thread dependency through global memory inside this CTA: rows
        # written by one thread are read by others in the next step, so each
        # store phase must be fenced before the dependent loads (same pattern
        # as the chained in-place swaps in _lu_apply_left_par /
        # _lu_swap_right_solve_par — without the barrier Triton may reorder
        # the loads ahead of the stores and the factorization races).
        tl.debug_barrier()
        # 3. scale the below-diagonal entries of column j.
        s = tl.load(
            LU_ptr + base + (j + 1 + allr) * N + j,
            mask=(j + 1 + allr) < M,
            other=0.0,
        )
        s = s / pivot
        tl.store(
            LU_ptr + base + (j + 1 + allr) * N + j,
            s,
            mask=(j + 1 + allr) < M,
        )
        tl.debug_barrier()
        # 4. rank-1 update rows (j, M) x cols (jj+1, PANEL), in row chunks.
        if jj + 1 < PANEL:
            ux = tl.load(
                LU_ptr + base + j * N + (K0 + jj + 1 + allc),
                mask=(jj + 1 + allc) < PANEL,
                other=0.0,
            )
            for r0 in tl.range(j + 1, M, ROW_TILE):
                rows_r = r0 + rt
                rmask = rows_r < M
                sc = tl.load(LU_ptr + base + rows_r * N + j, mask=rmask, other=0.0)
                xmask = (jj + 1 + allc) < PANEL
                blk = tl.load(
                    LU_ptr + base + rows_r[:, None] * N + (K0 + jj + 1 + allc)[None, :],
                    mask=rmask[:, None] & xmask[None, :],
                    other=0.0,
                )
                blk = blk - sc[:, None] * ux[None, :]
                tl.store(
                    LU_ptr + base + rows_r[:, None] * N + (K0 + jj + 1 + allc)[None, :],
                    blk,
                    mask=rmask[:, None] & xmask[None, :],
                )
        # fence this column's stores before the next column's pivot search
        tl.debug_barrier()
    tl.store(info_ptr + pid, info_val)


def _thead_lu_factor(A):
    """Blocked LU for thead (M > 64), fully barrier-free and deterministic:
    the panel factorization is a single CTA per matrix (_thead_lu_panel_serial —
    the R-group barrier kernel deadlocks on thead), the row swaps / solve and
    the trailing update are launch-ordered kernels whose fp64 tl.dots are
    capped at (32,32,32) 4-warp tiles (the shape envelope thead's fp64 dot
    stays deterministic in once magnitudes grow — see the probe evidence in
    the _thead_fp64_mm16_kernel note).  Returns (LU, pivots, info) with
    0-based IPIV pivots."""
    A = A.contiguous()
    M = A.shape[-1]
    N = A.shape[-1]
    batch = A.numel() // (M * N)
    LU = A.reshape(batch, M, N).clone()
    pivots = torch.empty(batch, M, dtype=torch.int32, device=A.device)
    info = torch.zeros(batch, dtype=torch.int32, device=A.device)

    for k0 in range(0, N, _LU_PAR_PANEL):
        p = min(_LU_PAR_PANEL, N - k0)
        col_block = triton.next_power_of_2(p)
        _thead_lu_panel_serial[(batch,)](
            LU,
            pivots,
            info,
            k0,
            M,
            N,
            p,
            triton.next_power_of_2(M),
            col_block,
            _LU_PAR_TILE_M // 2,
            num_warps=8,
        )
        if k0 > 0:
            _lu_apply_left_par[(triton.cdiv(k0, _LU_PAR_TILE_N), batch)](
                LU,
                pivots,
                k0,
                M,
                N,
                p,
                _LU_PAR_TILE_N,
                num_warps=4,
            )
        trailing_n = N - k0 - p
        if trailing_n > 0:
            _lu_swap_right_solve_par[(triton.cdiv(trailing_n, _LU_PAR_TILE_N), batch)](
                LU,
                pivots,
                k0,
                M,
                N,
                p,
                col_block,
                _LU_PAR_TILE_N,
                num_warps=4,
            )
            _thead_lu_update16_par[
                (
                    triton.cdiv(trailing_n, 32),
                    triton.cdiv(trailing_n, 32),
                    batch,
                )
            ](LU, k0, M, N, p, num_warps=4)
    return LU.reshape(A.shape), pivots, info


def _thead_inverse_large(A):
    """A^-1 for M > 64 on thead, deterministic: parallel LU whose only fp64
    tl.dots are (16,16,16) tiles (_thead_lu_factor), then the scalar
    forward/backward substitution solve (no tl.dot at all).  fp64 storage
    throughout — no fp64 dot ever exceeds the tile shape that is reliable on
    this platform."""
    m = A.shape[-1]
    A3 = A.reshape(-1, m, m)
    batch = A3.shape[0]
    LU, pivots, info = _thead_lu_factor(A3)
    if _info_has_error(info):
        raise RuntimeError(
            "linalg_matrix_power: the input matrix is singular (LU factorization "
            "encountered a zero pivot)"
        )
    # 0-based IPIV -> 1-based -> row-permutation index, then solve with the
    # row-permuted identity: A = P L U, so L U X = P solves A X = I.
    perm = _pivots_to_perm_gpu(pivots + 1, m)  # (batch, m), int64
    eye = torch.eye(m, dtype=A.dtype, device=A.device)
    if batch == 1:
        pb = eye[perm[0]].reshape(1, m, m)
    else:
        pb = torch.gather(
            eye.expand(batch, m, m).contiguous(),
            1,
            perm.unsqueeze(-1).expand(batch, m, m),
        )
    # Blocked triangular solve in place, mirroring the shared lu_solve flow
    # (unit-lower L forward, then upper U backward) with _trsm_kernel's D16
    # mode — every fp64 tl.dot stays a (16,16,16) tile, the only fp64-dot
    # shape that is deterministic on thead once magnitudes grow.  (The legacy
    # scalar substitution kernels ran one serial O(M^2) loop per RHS column,
    # ~50x slower at M=1024.)
    X = pb.clone()
    for b in range(batch):
        _trsm_solve_2d(LU[b], X[b], upper=False, unitriangular=True, d16=True)
        _trsm_solve_2d(LU[b], X[b], upper=True, unitriangular=False, d16=True)
    return X.reshape(A.shape)


def _thead_fp64_power_large(X, k, shape):
    """X^k for M > 64 on thead, deterministic: host binary exponentiation
    whose matmuls are the (16,16,16)-tile _thead_fp64_mm16."""
    m = X.shape[-1]
    X2 = X.reshape(-1, m, m)
    res = None
    while k > 0:
        if k & 1:
            res = X2 if res is None else _thead_fp64_mm16(res, X2)
        k >>= 1
        if k > 0:
            X2 = _thead_fp64_mm16(X2, X2)
    return res.reshape(shape)


def linalg_matrix_power(A, n, *, out=None):
    """thead dispatch: negative powers via df64 / 16-tile fp64 routes (see
    the module docstring); positive powers follow the generic NV dispatch
    minus the grid-sync Tier 3."""
    _generic.logger.debug("GEMS LINALG_MATRIX_POWER (thead)")

    # ---- validation (identical to the generic entry) ----
    shape = A.shape
    if len(shape) < 2:
        raise RuntimeError(
            f"linalg_matrix_power: A must be at least 2-D, got shape {shape}"
        )
    m, k = shape[-2], shape[-1]
    if m != k:
        raise RuntimeError(f"linalg_matrix_power: A must be square, got ({m}, {k})")
    if not isinstance(n, int):
        raise TypeError(f"linalg_matrix_power: n must be int, got {type(n).__name__}")
    if A.dtype not in (torch.float32, torch.float64):
        raise RuntimeError(
            f"linalg_matrix_power: flag_gems supports only float32 and float64, "
            f"got {A.dtype}"
        )

    # ---- n == 0 / n == 1 ----
    if n == 0:
        eye = _eye_like(A)
        if out is not None:
            out.copy_(eye)
            return out
        return eye
    if n == 1:
        if out is not None:
            out.copy_(A)
            return out
        return A.clone()

    if A.device.type != flag_gems.device:
        raise RuntimeError(
            f"linalg_matrix_power: flag_gems supports only {flag_gems.device}, "
            f"got {A.device}"
        )

    upcast = False
    if n < 0:
        # thead negative powers are fully deterministic by construction:
        # pure-fp32 df64 arithmetic for M <= 64 (in-register kernels), and an
        # fp64 chain whose every tl.dot is a (16,16,16) tile for larger M —
        # the platform's fp64 dot sporadically returns garbage with wider
        # tiles once operand magnitudes grow (see the thead policy row).
        if m > _DF64_MAX_M:
            # M > 64: the in-register df64 kernels stop at 64, so run the
            # whole chain in fp64 through _thead_inverse_large /
            # _thead_fp64_mm16 (16-tile dots only).  fp32 inputs ride the same
            # fp64 chain and the result is cast back — the cast reproduces the
            # fp32-cast reference exactly, including +/-inf past fp32 range.
            A64 = A if A.dtype == torch.float64 else A.double()
            X = _thead_inverse_large(A64)
            hi = _thead_fp64_power_large(X, -n, shape)
            if A.dtype == torch.float32:
                hi = hi.float()
            if out is not None:
                out.copy_(hi)
                return out
            return hi
        if A.dtype == torch.float64:
            # Split into an fp32 (hi, lo) pair (_split_fp64_pair keeps the
            # split error-free: h = fp32(a), l = a - fp64(h) is exact) and
            # invert the *pair* in df64: inverting the hi part alone leaves
            # the inverse wrong by ~cond(A)*2^-24 (~1e-6 on the cond-80
            # tests, over the fp64 rtol).
            Ah, Al = _split_fp64_pair(A)
        else:
            # fp32 inputs are exact as the df64 pair's hi part (low part 0).
            Ah = A.contiguous()
            Al = None
        Xh, Xl = _inverse(Ah, use_df64=True, A_l=Al)
        k = -n
        # A df64 pair overflows fp32 once (A^-1)^k leaves ~3.4e38 (the
        # cond-80 inputs do that past |n| ~ 19).  Pre-scale the power's input
        # pair by an exact power of two so every binary-exponentiation
        # intermediate stays in fp32 range — possible for any k whose result
        # fp64 itself can represent — then recombine in fp64 and scale back
        # (both steps exact: powers of two).  fp32 results are cast from the
        # fp64 recombine, which also keeps the correct +/-inf sign past fp32
        # range (the fp32-cast reference is +/-inf there too).
        s = _df64_power_scale(Xh, k)
        hi, lo = _matrix_power_df64_pair(Xh, Xl, k, m, shape, scale=2.0**-s)
        # fp64 recombine (hi + lo) with the exact power-of-two scale-back
        # 2**(k*s), cast to fp32 for fp32 inputs — one custom kernel instead
        # of torch add/mul/pow (see the host-function rules).
        res = _df64_recombine(hi, lo, shape, A.dtype, k * s)
        if out is not None:
            out.copy_(res)
            return out
        return res
    upcast = False
    # ---- n == 2, 3: fast paths for large M ----
    if n == 2 and m > TRITON_THRESHOLD:
        r = _matmul(A, A)
        if upcast:
            r = r.float()
        if out is not None:
            out.copy_(r)
            return out
        return r
    if n == 3 and m > TRITON_THRESHOLD:
        r = _matmul(_matmul(A, A), A)
        if upcast:
            r = r.float()
        if out is not None:
            out.copy_(r)
            return out
        return r

    # ---- flatten batch ----
    if len(shape) > 2:
        A_flat = A.reshape(-1, m, m)
    else:
        A_flat = A.unsqueeze(0)
    batch_size = A_flat.shape[0]
    batch_stride = m * m

    if out is not None:
        if upcast:
            # fp64 compute buffer (the kernels produce fp64); cast to fp32 out
            # at the end.
            out_flat = torch.empty(
                batch_size, m, m, dtype=torch.float64, device=A.device
            )
        else:
            out_flat = out.reshape(-1, m, m)
    else:
        out_flat = torch.empty(batch_size, m, m, dtype=A.dtype, device=A.device)

    # ---- dispatch ----
    if m <= SINGLE_TILE_MAX and A.device.type == flag_gems.device:
        # Tier 1: single-program fused (M <= 32).  tl.dot in sweet spot.
        BLOCK = max(triton.next_power_of_2(m), 16)
        _single_tile_kernel[(batch_size,)](
            A_flat,
            out_flat,
            m,
            n,
            batch_stride,
            BLOCK=BLOCK,
        )

    elif m <= TILED_MAX and A.device.type == flag_gems.device:
        # Tier 2: single-tile (33 <= M <= 64).
        # Grid-sync barrier overhead (~5 us/barrier × 3) exceeds the
        # single-SM tl.dot(64,64) time for 4-tile grids.  CUDA graph
        # memcpy overhead (~5 us × 3 copies) also dominates for M≤64.
        BLOCK = max(triton.next_power_of_2(m), 16)
        _single_tile_kernel[(batch_size,)](
            A_flat,
            out_flat,
            m,
            n,
            batch_stride,
            BLOCK=BLOCK,
        )

    else:
        # M > 256 (and, on thead, 65 <= M <= 256: the grid spin-barrier
        # Tier 3 hangs here, see the module docstring): host-side binary
        # exponentiation with the flag_gems Triton matmul kernels (mm for
        # 2D, bmm for batched), one launch per step.
        is_batched = batch_size > 1
        z = A_flat if is_batched else A_flat.squeeze(0)
        result = None
        n_remaining = n
        while n_remaining > 0:
            if n_remaining & 1:
                result = z if result is None else _matmul(result, z)
            n_remaining >>= 1
            if n_remaining > 0:
                z = _matmul(z, z)
        if is_batched:
            out_flat.copy_(result)
        else:
            out_flat.squeeze_(0).copy_(result)

    # ---- reshape back ----
    if upcast:
        out_flat = out_flat.float()
    if len(shape) > 2:
        out_flat = out_flat.reshape(shape)
    else:
        out_flat = out_flat.squeeze(0)

    if out is not None:
        if upcast:
            out.copy_(out_flat)
        return out
    return out_flat
