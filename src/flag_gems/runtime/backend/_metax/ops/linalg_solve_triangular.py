"""Metax (MACA) backend for linalg_solve_triangular.

Same K-slice design as src/flag_gems/ops/linalg_solve_triangular.py (notle
path: one CTA per column slice, intra-CTA tl.debug_barrier only), with four
metax-specific constraints (MetaX C550, triton 3.6.0+metax):
  1. tl.dot needs both dims >= 16 (N=8 fails in ConvertTritonGPUToLLVM)
     -> K_SLICE is 16+ everywhere, never 8.
  2. fp64 tl.dot SILENTLY produces wrong results (probe rel ~1.2)
     -> fp64 stays on serial substitution + serial outer-product update.
  3. The tl.sum(3D-broadcast) pattern is auto-rewritten to a tf32 dot on
     triton >= 3.5 -> never used (the 2D x_sum reduction in the serial diag
     is NOT rewritten -- matmul-equiv M=1 < 16 -- verified on triton 3.6).
  4. CTA shared memory limit is 64 KB: a BLOCK_SIZE=64 kernel needs 78-86 KB
     for the 64x64 dot operand staging regardless of num_stages (OutOfResources
     at launch) -> BLOCK_SIZE is 32 everywhere, the bs64/M^32 Neumann branch
     is intentionally absent.

fp32: dot update (exact, allow_tf32=False honored, maca_mma v2) + Neumann
parallel diag (9 dots, M^32 = 0 for bs32, all dot dims >= 16); metax sweep
shows Neumann beats the serial diag even at n=32/64 (2-3x). fp64: fully
serial, never dot.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


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
    """K-slice TRSM: X buffer operates directly on global B, no smem."""
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
            # Serial row loop for fp64 (metax fp64 dot is silently wrong) and
            # for small n (keeps the already-good <=64 behavior; Neumann costs
            # 9+ dots per block). The reciprocal is read inline per row (no
            # INV scratch buffer). x_sum uses a 2D reduction, which is NOT
            # tf32-dot-rewritten on triton 3.6 (matmul-equiv M=1 < 16).
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
            # no per-row barriers.
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
            # series terminates. Lower blocks: matrix rows > cols (r2 > c2).
            # Upper blocks: the bottom-up remap flips the structure, so pick
            # matrix cols > rows (c2 > r2) -- exactly the entries the
            # relative rows consume, and c2 > r2 <=> rel_col < rel_row.
            # Padding rows/cols of a ragged trailing block load as zero /
            # diag 1, i.e. identity.
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

            if not IS_FP64:
                # Solved X panel loaded once per block; each m-tile's update
                # is one dense tl.dot (fp32, exact on maca_mma v2). This
                # replaces the serial per-element FMA loop, which issued one
                # strided 4B load per element and wasted ~8x bandwidth on
                # cache sectors.
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
                if IS_FP64:
                    # Metax fp64 dot silently produces wrong results (probe
                    # rel ~1.2) -- serial outer-product FMAs instead.
                    acc = tl.zeros((BM, K_SLICE), dtype=b_curr.dtype)
                    for ks in range(BLOCK_SIZE):
                        # ks >= blk_sz for a ragged trailing block: mask both
                        # loads so OOB columns/rows contribute 0.
                        valid = ks < blk_sz
                        a_col = tl.load(
                            A_ptr + rm * stride_a_n + (blk_start + ks),
                            mask=mask_m & valid,
                            other=0.0,
                        )
                        x_row = tl.load(
                            B_ptr + (blk_start + ks) * stride_b_k + col_offs_1d,
                            mask=col_mask_1d & valid,
                            other=0.0,
                        )
                        acc += a_col[:, None] * x_row[None, :]
                    b_curr = b_curr.to(acc.dtype) - acc
                else:
                    a_sub = tl.load(
                        A_ptr
                        + rm[:, None] * stride_a_n
                        + (blk_start + a_cols)[None, :],
                        mask=mask_m[:, None] & (a_cols[None, :] < blk_sz),
                        other=0.0,
                    )
                    acc = tl.dot(a_sub, x_panel, allow_tf32=False)
                    b_curr = b_curr - acc
                tl.store(b_base, b_curr, mask=mask_m[:, None] & col_mask_1xs)

        # Ensure this block's update is visible to the next block's diagonal load
        tl.debug_barrier()


def linalg_solve_triangular(A, B, *, upper, left=True, unitriangular=False, out=None):
    logger.debug("GEMS_METAX LINALG_SOLVE_TRIANGULAR")
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

    # Fast path: N <= 16 (serial per-row substitution, one CTA per (batch, k-tile))
    if n <= 16:
        BLOCK_K = 32 if k <= 32 else 64
        num_k_tiles = (k + BLOCK_K - 1) // BLOCK_K
        inv = torch.zeros(batch * num_k_tiles * 16, dtype=dtype, device=A_view.device)
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

    # K-slice kernel for N > 16: one CTA per column slice, intra-CTA sync only.
    upd_bm = 128
    # Metax constraints: K_SLICE must be >= 16 (dot N-dim; N=8 fails to
    # compile); BLOCK_SIZE must be 32 (bs64 dot staging needs 78-86 KB smem
    # vs the 64 KB CTA limit, num_stages-independent). fp32 uses the Neumann
    # diag for ALL n > 16 (metax sweep: 2-3x faster than the serial diag
    # even at n=32/64 -- maca_mma is real MMA, unlike Hygon's FMA emulation);
    # fp64 stays serial via the IS_FP64 gate inside the kernel.
    BLOCK_SIZE = 32
    K_SLICE = 16
    num_kslices = (k + K_SLICE - 1) // K_SLICE
    small_diag = False
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
