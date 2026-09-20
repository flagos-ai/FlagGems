import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.triton_version_utils import HAS_TLE

if HAS_TLE:
    import triton.experimental.tle.language as tle
else:
    tle = None

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _kslice_trsm_kernel(
    A_ptr,
    B_ptr,
    N,
    K,
    stride_a_n,
    stride_b_k,
    BLOCK_SIZE: tl.constexpr,
    K_SLICE: tl.constexpr,
    BM: tl.constexpr,
    IS_FP64: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
):
    """K-slice parallel, barrier-free TRSM (mirrors cuBLAS structure).

    Each CTA owns one K-slice (column range) and walks ALL diagonal blocks
    serially. Cross-block dependency (block k+1 diag needs block k update of
    B[blk_{k+1}]) is satisfied by serial ordering WITHIN a CTA — no cross-CTA
    barrier needed. K-slices are fully independent.
    """
    pid = tl.program_id(0)
    col_start = pid * K_SLICE
    if col_start >= K:
        return

    sm_dtype = tl.float64 if IS_FP64 else tl.float32
    num_blocks = tl.cdiv(N, BLOCK_SIZE)

    A_sm = tle.gpu.alloc(
        [BLOCK_SIZE, BLOCK_SIZE],
        dtype=sm_dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    X_sm = tle.gpu.alloc(
        [BLOCK_SIZE, K_SLICE],
        dtype=sm_dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    D_sm = tle.gpu.alloc(
        [BLOCK_SIZE],
        dtype=sm_dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )

    a_rows = tl.arange(0, BLOCK_SIZE)
    a_cols = tl.arange(0, BLOCK_SIZE)
    ar = tl.broadcast_to(a_rows[:, None], (BLOCK_SIZE, BLOCK_SIZE))
    ac = tl.broadcast_to(a_cols[None, :], (BLOCK_SIZE, BLOCK_SIZE))

    x_rows = tl.arange(0, BLOCK_SIZE)
    x_kcols = tl.arange(0, K_SLICE)
    xr = tl.broadcast_to(x_rows[:, None], (BLOCK_SIZE, K_SLICE))
    xc = tl.broadcast_to(x_kcols[None, :], (BLOCK_SIZE, K_SLICE))
    col_offs = col_start + x_kcols
    col_mask = col_offs < K

    rr = tl.arange(0, BM)

    for block_idx in range(num_blocks):
        bk = block_idx if not UPPER else num_blocks - 1 - block_idx
        blk_start = bk * BLOCK_SIZE
        blk_end = tl.minimum(blk_start + BLOCK_SIZE, N)
        blk_sz = blk_end - blk_start

        # ═══════ Diag: forward substitution for X[blk_k, kslice] ═══════
        am = (ar < blk_sz) & (ac < blk_sz)
        loc_a = tle.gpu.local_ptr(A_sm, (ar, ac))
        src_a = A_ptr + (blk_start + ar) * stride_a_n + (blk_start + ac)
        tl.store(loc_a, tl.load(src_a, mask=am, other=0.0))

        rm_bc = xr < blk_sz
        loc_x = tle.gpu.local_ptr(X_sm, (xr, xc))
        src_x = B_ptr + (blk_start + xr) * stride_b_k + col_offs[None, :]
        tl.store(loc_x, tl.load(src_x, mask=rm_bc & col_mask[None, :], other=0.0))

        # Pre-compute diagonal reciprocals in parallel: move division out of
        # the serial forward-substitution chain (multiply instead)
        if not UNIT:
            diag_loc = tle.gpu.local_ptr(A_sm, (a_rows, a_rows))
            inv_diag = 1.0 / tl.load(diag_loc, mask=a_rows < blk_sz, other=1.0)
            d_loc = tle.gpu.local_ptr(D_sm, (a_rows,))
            tl.store(d_loc, inv_diag, mask=a_rows < blk_sz)

        for r_idx in range(blk_sz):
            row = blk_end - 1 - r_idx if UPPER else blk_start + r_idx
            row_rel = row - blk_start
            rb = tl.broadcast_to(row_rel, (BLOCK_SIZE,))
            rbk = tl.broadcast_to(row_rel, (K_SLICE,))

            # parallel reduction over j (breaks serial FMA dependency chain)
            a_row_loc = tle.gpu.local_ptr(A_sm, (rb, a_cols))
            a_row = tl.load(a_row_loc, mask=a_cols < blk_sz, other=0.0)
            if UPPER:
                a_row = tl.where(a_cols > row_rel, a_row, 0.0)
            else:
                a_row = tl.where(a_cols < row_rel, a_row, 0.0)

            x_all_loc = tle.gpu.local_ptr(X_sm, (xr, xc))
            x_all = tl.load(x_all_loc, mask=rm_bc & col_mask[None, :], other=0.0)
            x_sum = tl.sum(a_row[:, None] * x_all, axis=0)

            # Updated row written back through a 2D row-masked store: a 1D
            # (K_SLICE,) store gets replicated across the CTA's lanes and
            # races (write-write) on smem.
            row_sel = (xr == rbk) & col_mask[None, :]
            x_new = x_all - tl.broadcast_to(x_sum[None, :], (BLOCK_SIZE, K_SLICE))
            if not UNIT:
                inv_d_loc = tle.gpu.local_ptr(D_sm, (rbk,))
                x_new *= tl.broadcast_to(
                    tl.load(inv_d_loc, mask=col_mask, other=1.0)[None, :],
                    (BLOCK_SIZE, K_SLICE),
                )
            tl.store(x_all_loc, x_new, mask=row_sel)

        # write X back to B[blk_k, kslice] (in-place output)
        dst_x = B_ptr + (blk_start + xr) * stride_b_k + col_offs[None, :]
        tl.store(
            dst_x,
            tl.load(loc_x, mask=rm_bc & col_mask[None, :], other=0.0),
            mask=rm_bc & col_mask[None, :],
        )

        # ═══════ Update: B[rest, kslice] -= A[rest, blk_k] @ X[blk_k, kslice] ═══════
        need_update = tl.where(UPPER, bk > 0, blk_end < N)
        if need_update:
            M_REM = tl.where(UPPER, blk_start, N - blk_end)
            rem_s = tl.where(UPPER, 0, blk_end)
            bound = tl.where(UPPER, blk_start, N)

            x_panel = tl.load(loc_x, mask=rm_bc & col_mask[None, :], other=0.0)
            for m_start in range(0, M_REM, BM):
                rm = rem_s + m_start + rr
                mask_m = rm < bound
                a_sub = tl.load(
                    A_ptr + rm[:, None] * stride_a_n + (blk_start + a_cols)[None, :],
                    mask=mask_m[:, None] & (a_cols[None, :] < blk_sz),
                    other=0.0,
                )
                if K_SLICE < 8:
                    # tl.dot requires N >= 8; use manual FMA reduction for small K_SLICE
                    acc = tl.sum(a_sub[:, :, None] * x_panel[None, :, :], axis=1)
                elif IS_FP64:
                    acc = tl.dot(a_sub, x_panel, allow_tf32=False)
                else:
                    acc = tl.dot(
                        a_sub.to(tl.float32), x_panel.to(tl.float32), allow_tf32=False
                    )
                b_base = B_ptr + rm[:, None] * stride_b_k + col_offs[None, :]
                b_curr = tl.load(
                    b_base, mask=mask_m[:, None] & col_mask[None, :], other=0.0
                )
                b_curr = b_curr.to(acc.dtype) - acc
                tl.store(b_base, b_curr, mask=mask_m[:, None] & col_mask[None, :])

        # Ensure this block's update is visible to the next block's diagonal load (prevent compiler reordering)
        tl.debug_barrier()


@libentry()
@triton.jit
def _small_diag_kernel(
    A_ptr,
    B_ptr,
    N,
    K,
    BLOCK_K: tl.constexpr,
    IS_UPPER: tl.constexpr,
    UNITRIANGULAR: tl.constexpr,
    IS_FP64: tl.constexpr,
    stride_a_n,
    stride_b_k,
):
    pid_batch = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_start = pid_k * BLOCK_K
    if k_start >= K:
        return

    A_batch = A_ptr + pid_batch * stride_a_n * N
    B_batch = B_ptr + pid_batch * stride_b_k * N
    k_offs = k_start + tl.arange(0, BLOCK_K)
    k_mask = k_offs < K

    # Pre-compute diagonal reciprocals: move division out of the serial forward-substitution chain (multiply instead)
    if not UNITRIANGULAR:
        sm_dtype = tl.float64 if IS_FP64 else tl.float32
        d_sm = tle.gpu.alloc(
            [16],
            dtype=sm_dtype,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        r16 = tl.arange(0, 16)
        diag_vals = tl.load(A_batch + r16 * stride_a_n + r16, mask=r16 < N, other=1.0)
        d_loc = tle.gpu.local_ptr(d_sm, (r16,))
        tl.store(d_loc, 1.0 / diag_vals, mask=r16 < N)

    for r in range(N):
        row = N - 1 - r if IS_UPPER else r
        b_row = tl.load(B_batch + row * stride_b_k + k_offs, mask=k_mask, other=0.0)
        if IS_UPPER:
            for j in range(row + 1, N):
                a_val = tl.load(A_batch + row * stride_a_n + j)
                x_j = tl.load(B_batch + j * stride_b_k + k_offs, mask=k_mask, other=0.0)
                b_row -= a_val * x_j
        else:
            for j in range(row):
                a_val = tl.load(A_batch + row * stride_a_n + j)
                x_j = tl.load(B_batch + j * stride_b_k + k_offs, mask=k_mask, other=0.0)
                b_row -= a_val * x_j
        if not UNITRIANGULAR:
            inv_d = tl.load(
                tle.gpu.local_ptr(d_sm, (tl.full((), row, dtype=tl.int32),))
            )
            b_row *= inv_d
        tl.store(B_batch + row * stride_b_k + k_offs, b_row, mask=k_mask)


# ══════════════════════════════════════════════════════════════════════════
# Non-TLE fallback implementations (HAS_TLE=False platforms, pure Triton, no tle.gpu dependency)
# Structure mirrors the TLE versions, replacing smem buffers with global memory access
# ══════════════════════════════════════════════════════════════════════════


@libentry()
@triton.jit
def _small_diag_kernel_notle(
    A_ptr,
    B_ptr,
    INV_ptr,
    N,
    K,
    NUM_K_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_UPPER: tl.constexpr,
    UNITRIANGULAR: tl.constexpr,
    stride_a_n,
    stride_b_k,
):
    pid_batch = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_start = pid_k * BLOCK_K
    if k_start >= K:
        return

    A_batch = A_ptr + pid_batch * stride_a_n * N
    B_batch = B_ptr + pid_batch * stride_b_k * N
    k_offs = k_start + tl.arange(0, BLOCK_K)
    k_mask = k_offs < K

    # Pre-compute diagonal reciprocals (global array, division moved out of the serial chain)
    if not UNITRIANGULAR:
        r16 = tl.arange(0, 16)
        diag_vals = tl.load(A_batch + r16 * stride_a_n + r16, mask=r16 < N, other=1.0)
        tl.store(
            INV_ptr + (pid_batch * NUM_K_TILES + pid_k) * 16 + r16,
            1.0 / diag_vals,
            mask=r16 < N,
        )
        # Make INV writeback visible to the row loop's cross-thread reads (no automatic sync on global)
        tl.debug_barrier()

    for r in range(N):
        row = N - 1 - r if IS_UPPER else r
        b_row = tl.load(B_batch + row * stride_b_k + k_offs, mask=k_mask, other=0.0)
        if IS_UPPER:
            for j in range(row + 1, N):
                a_val = tl.load(A_batch + row * stride_a_n + j)
                x_j = tl.load(B_batch + j * stride_b_k + k_offs, mask=k_mask, other=0.0)
                b_row -= a_val * x_j
        else:
            for j in range(row):
                a_val = tl.load(A_batch + row * stride_a_n + j)
                x_j = tl.load(B_batch + j * stride_b_k + k_offs, mask=k_mask, other=0.0)
                b_row -= a_val * x_j
        if not UNITRIANGULAR:
            inv_d = tl.load(INV_ptr + (pid_batch * NUM_K_TILES + pid_k) * 16 + row)
            b_row *= inv_d
        tl.store(B_batch + row * stride_b_k + k_offs, b_row, mask=k_mask)
        # Make this row's writeback visible to the next row's cross-thread
        # reads (row dependency, no automatic sync on global)
        tl.debug_barrier()


@libentry()
@triton.jit
def _kslice_trsm_kernel_notle(
    A_ptr,
    B_ptr,
    N,
    K,
    stride_a_n,
    stride_b_k,
    BLOCK_SIZE: tl.constexpr,
    K_SLICE: tl.constexpr,
    BM: tl.constexpr,
    IS_FP64: tl.constexpr,
    SMALL_N: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
):
    """Non-TLE K-slice TRSM: X buffer operates directly on global B, no smem."""
    pid = tl.program_id(0)
    col_start = pid * K_SLICE
    if col_start >= K:
        return

    num_blocks = tl.cdiv(N, BLOCK_SIZE)

    a_cols = tl.arange(0, BLOCK_SIZE)
    x_rows = tl.arange(0, BLOCK_SIZE)
    x_kcols = tl.arange(0, K_SLICE)
    xr = tl.broadcast_to(x_rows[:, None], (BLOCK_SIZE, K_SLICE))
    # 1D column offsets feed the 1xK_SLICE broadcasts used for B stores; all
    # stores go through 2D-masked versions so every element is written by
    # exactly one lane (a 1D store gets replicated across lanes and races).
    col_offs_1d = col_start + x_kcols
    col_mask_1d = col_offs_1d < K
    col_offs_1xs = col_offs_1d[None, :]
    col_mask_1xs = col_mask_1d[None, :]
    xc = tl.broadcast_to(x_kcols[None, :], (BLOCK_SIZE, K_SLICE))
    col_offs = col_start + xc
    col_mask = col_offs < K

    rr = tl.arange(0, BM)

    for block_idx in range(num_blocks):
        bk = block_idx if not UPPER else num_blocks - 1 - block_idx
        blk_start = bk * BLOCK_SIZE
        blk_end = tl.minimum(blk_start + BLOCK_SIZE, N)
        blk_sz = blk_end - blk_start

        # ═══════ Diag: block solve ═══════
        if IS_FP64 or SMALL_N:
            # Serial row loop for fp64 and small n: the parallel Neumann form
            # below costs 9 tl.dot's per block, which is 2.4-2.8x slower than
            # serial substitution on backends that emulate fp64 dot (NVIDIA)
            # and ~3x slower for tiny blocks on Hygon where dot is FMA-
            # emulated (n=32: 0.086ms Neumann vs ~0.030ms serial there).
            # The reciprocal is read inline per row (no INV scratch buffer).
            for r_idx in range(blk_sz):
                row = blk_end - 1 - r_idx if UPPER else blk_start + r_idx
                row_rel = row - blk_start

                # Row of A (global, triangular mask)
                a_row = tl.load(
                    A_ptr + row * stride_a_n + blk_start + a_cols,
                    mask=a_cols < blk_sz,
                    other=0.0,
                )
                if UPPER:
                    a_row = tl.where(a_cols > row_rel, a_row, 0.0)
                else:
                    a_row = tl.where(a_cols < row_rel, a_row, 0.0)

                x_all = tl.load(
                    B_ptr + (blk_start + xr) * stride_b_k + col_offs,
                    mask=(xr < blk_sz) & col_mask,
                    other=0.0,
                )
                x_sum = tl.sum(a_row[:, None] * x_all, axis=0)
                # Update the full tile, write back only row `row_rel` through
                # a 2D row mask (a 1D store gets replicated across the CTA's
                # lanes and races on global memory).
                x_new = x_all - tl.broadcast_to(x_sum[None, :], (BLOCK_SIZE, K_SLICE))
                if not UNIT:
                    x_new *= 1.0 / tl.load(A_ptr + row * stride_a_n + row)
                row_sel = (xr == row_rel) & col_mask
                tl.store(
                    B_ptr + (blk_start + xr) * stride_b_k + col_offs,
                    x_new,
                    mask=row_sel,
                )
                # Make this row's writeback visible to the next row's
                # cross-thread reads (row dependency, no sync on global)
                tl.debug_barrier()
        else:
            # Parallel block solve via Neumann series (fp32): remap rows so
            # the solve order is ascending (upper blocks process bottom-up,
            # relative index r maps to matrix row blk_end-1-r), write
            # T = D (I + M) where M is strictly lower in relative space and
            # nilpotent (M^BLOCK_SIZE = 0), then
            #     (I + M)^{-1} = (I - M)(I + M^2)(I + M^4)(I + M^8)(I + M^16)
            # exactly (the factors are polynomials in M and commute). The
            # whole block solve is 9 parallel tl.dot's -- no serial row loop,
            # no per-row barriers, which was the Hygon bottleneck (one
            # strided row load + 32-wide reduction + barrier per row).
            rows32 = tl.arange(0, BLOCK_SIZE)
            row1 = blk_end - 1 - rows32 if UPPER else blk_start + rows32
            r2 = row1[:, None]
            c2 = row1[None, :]
            # Validity is judged on RELATIVE indices: remapped row/col values
            # for upper blocks go negative on padding lanes (and exceed
            # blk_sz on lower blocks), so (r2 < blk_sz) would dereference
            # garbage or mask out the whole tile.
            in_blk = (rows32[:, None] < blk_sz) & (rows32[None, :] < blk_sz)
            a_blk = tl.load(
                A_ptr + r2 * stride_a_n + c2,
                mask=in_blk,
                other=0.0,
            )
            if UNIT:
                d_inv = tl.full((BLOCK_SIZE, 1), 1.0, dtype=a_blk.dtype)
            else:
                diag = tl.load(
                    A_ptr + row1 * stride_a_n + row1,
                    mask=rows32 < blk_sz,
                    other=1.0,
                )
                d_inv = (1.0 / diag)[:, None]
            # M must be strictly lower in RELATIVE space so the Neumann
            # series terminates. Lower blocks: pick matrix rows > cols
            # (r2 > c2). Upper blocks: the bottom-up remap flips the
            # structure, so pick matrix cols > rows (c2 > r2) -- these are
            # exactly the entries the relative rows consume, and
            # c2 > r2 <=> rel_col < rel_row. Padding rows/cols of a ragged
            # trailing block load as zero / diag 1, i.e. identity.
            if UPPER:
                M = tl.where(c2 > r2, a_blk, 0.0) * d_inv
            else:
                M = tl.where(c2 < r2, a_blk, 0.0) * d_inv
            x = tl.load(
                B_ptr + r2 * stride_b_k + col_offs,
                mask=(rows32[:, None] < blk_sz) & col_mask,
                other=0.0,
            )
            x = x * d_inv
            M2 = tl.dot(M, M, allow_tf32=False)
            M4 = tl.dot(M2, M2, allow_tf32=False)
            M8 = tl.dot(M4, M4, allow_tf32=False)
            M16 = tl.dot(M8, M8, allow_tf32=False)
            x = x - tl.dot(M, x, allow_tf32=False)
            x = x + tl.dot(M2, x, allow_tf32=False)
            x = x + tl.dot(M4, x, allow_tf32=False)
            x = x + tl.dot(M8, x, allow_tf32=False)
            x = x + tl.dot(M16, x, allow_tf32=False)
            if BLOCK_SIZE == 64:
                # M^64 = 0 needs one more factor; without it the series is
                # truncated at M^32 and wrong for ||M|| close to 1 (the test
                # matrices have ||M|| ~ 0.025, which hides the error).
                M32 = tl.dot(M16, M16, allow_tf32=False)
                x = x + tl.dot(M32, x, allow_tf32=False)
            tl.store(
                B_ptr + r2 * stride_b_k + col_offs,
                x,
                mask=(rows32[:, None] < blk_sz) & col_mask,
            )
            # Diag stores must be visible to the update phase's x_panel load
            # (no automatic sync on global)
            tl.debug_barrier()

        # ═══════ Update: B[rest, kslice] -= A[rest, blk_k] @ X[blk_k, kslice] ═══════
        need_update = tl.where(UPPER, bk > 0, blk_end < N)
        if need_update:
            M_REM = tl.where(UPPER, blk_start, N - blk_end)
            rem_s = tl.where(UPPER, 0, blk_end)
            bound = tl.where(UPPER, blk_start, N)

            # Solved X panel loaded once per block; each m-tile's update is a
            # dense matmul via tl.dot (allow_tf32=False, fp32 + fp64 — verified
            # on Hygon: exact precision, 1.5-2.6x faster than the serial
            # per-element FMA loop it replaces, which issued one strided 4B
            # load per element and wasted ~8x bandwidth on cache sectors).
            x_panel = tl.load(
                B_ptr + (blk_start + xr) * stride_b_k + col_offs,
                mask=(xr < blk_sz) & col_mask,
                other=0.0,
            )
            for m_start in range(0, M_REM, BM):
                rm = rem_s + m_start + rr
                mask_m = rm < bound
                b_base = B_ptr + rm[:, None] * stride_b_k + col_offs_1xs
                b_curr = tl.load(b_base, mask=mask_m[:, None] & col_mask_1xs, other=0.0)
                a_sub = tl.load(
                    A_ptr + rm[:, None] * stride_a_n + (blk_start + a_cols)[None, :],
                    mask=mask_m[:, None] & (a_cols[None, :] < blk_sz),
                    other=0.0,
                )
                acc = tl.dot(a_sub, x_panel, allow_tf32=False)
                tl.store(b_base, b_curr - acc, mask=mask_m[:, None] & col_mask_1xs)

        # Ensure this block's update is visible to the next block's diagonal load
        tl.debug_barrier()


def linalg_solve_triangular(A, B, *, upper, left=True, unitriangular=False, out=None):
    logger.debug("GEMS LINALG_SOLVE_TRIANGULAR")
    if A.dtype not in (torch.float32, torch.float64):
        raise ValueError("linalg_solve_triangular only supports float32 and float64")
    if B.dtype != A.dtype:
        raise ValueError("A and B must have the same dtype")

    if A.numel() == 0 or B.numel() == 0:
        if out is not None:
            out.copy_(B)
            return out
        return B.clone()

    if A.ndim < 2:
        raise ValueError("A must be at least 2D")
    if B.ndim < 2:
        raise ValueError("B must be at least 2D")
    if A.shape[-1] != A.shape[-2]:
        raise ValueError("A must be a square matrix")

    if not left:
        if A.shape[-1] != B.shape[-1]:
            raise ValueError("Shape mismatch for XA=B")
        result = linalg_solve_triangular(
            A.mT.contiguous(),
            B.mT.contiguous(),
            upper=not upper,
            left=True,
            unitriangular=unitriangular,
        )
        result = result.mT
        if out is not None:
            out.copy_(result)
            return out
        return result

    A = A.contiguous()
    B = B.contiguous()
    n, k = A.shape[-1], B.shape[-1]
    dtype = A.dtype
    is_fp64 = dtype == torch.float64
    orig_shape = B.shape

    if A.ndim > 2:
        batch = 1
        for d in A.shape[:-2]:
            batch *= d
        A_view = A.reshape(batch, n, n)
        B_view = B.clone().reshape(batch, n, k)
    else:
        batch = 1
        A_view = A.unsqueeze(0)
        B_view = B.clone().unsqueeze(0)

    stride_a_n = A_view.stride(1)
    stride_b_k = B_view.stride(1)

    # Fast path: N <= 16
    if n <= 16:
        BLOCK_K = 32 if k <= 32 else 64
        num_k_tiles = (k + BLOCK_K - 1) // BLOCK_K
        if HAS_TLE:
            _small_diag_kernel[(batch, num_k_tiles)](
                A_view,
                B_view,
                n,
                k,
                BLOCK_K,
                upper,
                unitriangular,
                is_fp64,
                stride_a_n,
                stride_b_k,
            )
        else:
            inv = torch.zeros(
                batch * num_k_tiles * 16, dtype=dtype, device=A_view.device
            )
            _small_diag_kernel_notle[(batch, num_k_tiles)](
                A_view,
                B_view,
                inv,
                n,
                k,
                num_k_tiles,
                BLOCK_K,
                upper,
                unitriangular,
                stride_a_n,
                stride_b_k,
            )
        B_view = B_view.reshape(orig_shape)
        if out is not None:
            out.copy_(B_view)
            return out
        return B_view

    # K-slice barrier-free kernel for N > 32: one CTA per column slice,
    # intra-CTA sync only (tl.debug_barrier), no cross-CTA barrier, fully
    # portable across backends.
    upd_bm = 128
    if HAS_TLE:
        # NVIDIA: keep 32/8 -- ks16+bs64 measured catastrophic there
        # (0.37-0.44x), ks8/bs32 is near-optimal for every (n, k).
        BLOCK_SIZE = 32
        K_SLICE = 8
    else:
        # Hygon sweep (2026-08-25): larger tiles win decisively at n>=512
        # (n=512 k=256: 0.88->1.19, n=512 k=8: 0.76->1.01, n=1024 k=8:
        # 0.44->0.58, n=2048 k=8: 0.24->0.33); all configs tie for n<512.
        BLOCK_SIZE = 64 if n >= 512 else 32
        K_SLICE = 16 if n >= 512 else 8
    num_kslices = (k + K_SLICE - 1) // K_SLICE
    # Serial diag for fp64 (Neumann costs 9 fp64 dots ~2.5x slower) and for
    # n<=64 fp32 (Hygon FMA-emulated dot makes tiny-block Neumann ~3x slower
    # than the serial row loop).
    small_diag = n <= 64
    if HAS_TLE:
        for b in range(batch):
            _kslice_trsm_kernel[(num_kslices,)](
                A_view[b],
                B_view[b],
                n,
                k,
                stride_a_n,
                stride_b_k,
                BLOCK_SIZE,
                K_SLICE,
                upd_bm,
                is_fp64,
                upper,
                unitriangular,
                num_warps=4,
            )
    else:
        for b in range(batch):
            _kslice_trsm_kernel_notle[(num_kslices,)](
                A_view[b],
                B_view[b],
                n,
                k,
                stride_a_n,
                stride_b_k,
                BLOCK_SIZE,
                K_SLICE,
                upd_bm,
                is_fp64,
                small_diag,
                upper,
                unitriangular,
                num_warps=4,
            )

    B_view = B_view.reshape(orig_shape)
    if out is not None:
        out.copy_(B_view)
        return out
    return B_view


def linalg_solve_triangular_out(
    A, B, *, upper, left=True, unitriangular=False, out=None
):
    return linalg_solve_triangular(
        A, B, upper=upper, left=left, unitriangular=unitriangular, out=out
    )
