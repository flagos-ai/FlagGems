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

"""linalg_matrix_power override for the metax (MACA) backend.

Metax runs the generic NV flow with one difference: its fp64 MMA (maca_mma
v2) cannot lower tl.dot with any dimension < 16 and writes its fp64 output
tiles with rows scrambled by a fixed 4x4 transpose inside every 16-row group.
Every kernel that performs an fp64 tl.dot therefore lives here in a private
fp64-fixed copy (_metax_fix_fp64_rows un-scrambles the rows); fp64 host-side
matmuls use the private _metax_fp64_mm tiled kernel.  The shared hosts
(linalg_lu_solve / _inverse / _newton_refine) are hooked to the private TRSM
and matmul below, and the dispatch mirrors the generic NV entry.
"""

import importlib

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.ops.linalg_matrix_power import (
    SINGLE_TILE_MAX,
    TILE,
    TILED_MAX,
    TRITON_THRESHOLD,
    _eye_like,
    _inverse,
    _trsm_solve_register,
    logger,
)
from flag_gems.utils import libentry

# NB: bind the *module* explicitly - ``import flag_gems.ops.linalg_matrix_power
# as _generic`` would resolve through the package attribute, which the
# re-exported ``linalg_matrix_power`` function shadows.
_generic = importlib.import_module("flag_gems.ops.linalg_matrix_power")


@triton.jit
def _metax_fix_fp64_rows(
    z, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, G: tl.constexpr
):
    """Undo the MetaX fp64 MMA row permutation.

    MetaX's fp64 dot writes its output tile with rows scrambled by a 4x4
    transpose inside every 16-row group (out[i] = z[4*(i%4) + i//4 + 16*(i//16)]).
    The reshape/trans round-trip un-scrambles the rows and forces the MMA
    result into a blocked layout, so it can be stored or fed into the next dot.
    Only invoked when FIX_FP64 is set (MetaX + fp64); other backends never
    enable it.
    """
    z = tl.reshape(z, (G, 4, 4, BLOCK_N))
    z = tl.trans(z, (0, 2, 1, 3))
    z = tl.reshape(z, (BLOCK_M, BLOCK_N))
    return z


@triton.jit(do_not_specialize=["n"])
def _single_tile_kernel(
    A_ptr,
    out_ptr,
    M,
    n,
    batch_stride,
    BLOCK: tl.constexpr,
    FIX_FP64: tl.constexpr = False,
    G: tl.constexpr = 1,
):
    """One program per batch element.  M <= BLOCK, one tl.dot per matmul."""
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < M)

    a_base = A_ptr + pid * batch_stride
    out_base = out_ptr + pid * batch_stride

    a = tl.load(a_base + offs_m[:, None] * M + offs_n[None, :], mask=mask, other=0.0)

    # fp64 inputs accumulate in fp64 (tl.dot supports fp64 via FMA); fp32/half
    # accumulate in fp32 — matches the generic file's fp64-accumulation policy.
    acc_dtype = a.dtype
    if acc_dtype != tl.float64:
        acc_dtype = tl.float32
    z = a.to(acc_dtype)
    result = z
    has_result = False
    n_remaining = n

    while n_remaining > 0:
        if n_remaining & 1:
            if not has_result:
                result = z
                has_result = True
            else:
                result = tl.dot(result, z, allow_tf32=False)
                if FIX_FP64:
                    result = _metax_fix_fp64_rows(result, BLOCK, BLOCK, G)
            if not FIX_FP64:
                result = tl.where(mask, result, 0.0)
        n_remaining >>= 1
        if n_remaining > 0:
            z = tl.dot(z, z, allow_tf32=False)
            if FIX_FP64:
                z = _metax_fix_fp64_rows(z, BLOCK, BLOCK, G)
            if not FIX_FP64:
                z = tl.where(mask, z, 0.0)

    result = result.to(a.dtype)
    tl.store(out_base + offs_m[:, None] * M + offs_n[None, :], result, mask=mask)


# ---------------------------------------------------------------------------
# df64 (double-single) arithmetic — an fp32 (hi, lo) pair with ~48-bit
# mantissa.  Used by the no-fp64 backends and thead routes (policy table).
# ---------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["n"])
def _grid_sync_kernel(
    A_ptr,
    out_ptr,
    scratch_ptr,
    barrier_ptr,
    M,
    n,
    batch_stride,
    TILE_BLOCK: tl.constexpr,
    TILES: tl.constexpr,
    FIX_FP64: tl.constexpr = False,
    G: tl.constexpr = 1,
):
    """Single-kernel binary exponentiation with grid-level sync.

    Grid: ``(batch_size, TILES, TILES)``.
    Each program owns one TILE×TILE output tile and runs the entire
    binary-exponentiation loop.  Between matmul steps all programs
    synchronise via an atomic barrier in global memory, giving
    multi-SM parallelism while keeping everything in one kernel launch.
    """
    pid_batch = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)

    offs_m = tl.arange(0, TILE_BLOCK)
    offs_k = tl.arange(0, TILE_BLOCK)
    offs_n = tl.arange(0, TILE_BLOCK)

    # Row / col range for this program's tile.
    rm = pid_i * TILE_BLOCK + offs_m
    rn = pid_j * TILE_BLOCK + offs_n
    mask = (rm[:, None] < M) & (rn[None, :] < M)

    # Base pointers for this batch element.
    a_base = A_ptr + pid_batch * batch_stride
    out_base = out_ptr + pid_batch * batch_stride
    scratch_stride = M * M
    # Each batch element gets its own 4 scratch slots (scratch: (4*batch, M, M)).
    scratch_base = scratch_ptr + pid_batch * 4 * scratch_stride
    barrier_base = barrier_ptr + pid_batch * 64

    # total_progs = total programs per batch element = TILES * TILES.
    # Each batch element has its own barrier slot (barrier_ptr + pid_batch * 64).
    n_total = TILES * TILES

    # -----------------------------------------------------------------
    # Step 0 — copy input A tiles to scratch[0] (z) and scratch[2] (result)
    # -----------------------------------------------------------------
    a_tile = tl.load(
        a_base + rm[:, None] * M + rn[None, :],
        mask=mask,
        other=0.0,
    )
    tl.store(
        scratch_base + 0 * scratch_stride + rm[:, None] * M + rn[None, :],
        a_tile,
        mask=mask,
    )
    tl.store(
        scratch_base + 2 * scratch_stride + rm[:, None] * M + rn[None, :],
        a_tile,
        mask=mask,
    )

    # ---- Grid barrier: every program must finish writing its scratch[0]/[2]
    # tile before the first matmul reads tiles owned by other programs. ----
    my_count = tl.atomic_add(barrier_base, 1, sem="release")
    barrier_round = (my_count // n_total) + 1
    target = barrier_round * n_total
    while tl.atomic_add(barrier_base, 0, sem="acquire") < target:
        pass

    # Ping-pong indices for the scratch buffer (4 slots per batch).
    #   z_buf:   0 or 1   — current power of two
    #   r_buf:   2 or 3   — current result
    z_buf = 0
    r_buf = 2
    has_result = False
    n_remaining = n

    while n_remaining > 0:
        if n_remaining & 1:
            if not has_result:
                # result = current z (scratch[z_buf]).  z may have advanced past
                # the input A (even n), so scratch[2]'s Step-0 copy of A is not
                # the right starting value; copy this program's z tile instead.
                has_result = True
                zval = tl.load(
                    scratch_base
                    + z_buf * scratch_stride
                    + rm[:, None] * M
                    + rn[None, :],
                    mask=mask,
                    other=0.0,
                )
                tl.store(
                    scratch_base + 2 * scratch_stride + rm[:, None] * M + rn[None, :],
                    zval,
                    mask=mask,
                )
                r_buf = 2
            else:
                dst_r = 5 - r_buf
                _compute_tiled_matmul(
                    scratch_base + r_buf * scratch_stride,
                    scratch_base + z_buf * scratch_stride,
                    scratch_base + dst_r * scratch_stride,
                    M,
                    rm,
                    rn,
                    offs_m,
                    offs_k,
                    offs_n,
                    mask,
                    TILE_BLOCK,
                    TILES,
                    FIX_FP64,
                    G,
                )
                r_buf = dst_r
        n_remaining >>= 1
        if n_remaining > 0:
            dst_z = 1 - z_buf
            _compute_tiled_matmul(
                scratch_base + z_buf * scratch_stride,
                scratch_base + z_buf * scratch_stride,
                scratch_base + dst_z * scratch_stride,
                M,
                rm,
                rn,
                offs_m,
                offs_k,
                offs_n,
                mask,
                TILE_BLOCK,
                TILES,
                FIX_FP64,
                G,
            )
            z_buf = dst_z

        # ---- Grid-level barrier (release/acquire semantics) ----
        my_count = tl.atomic_add(barrier_base, 1, sem="release")
        barrier_round = (my_count // n_total) + 1
        target = barrier_round * n_total
        # Spin with acquire semantics for faster visibility
        while tl.atomic_add(barrier_base, 0, sem="acquire") < target:
            pass

    # ---- Store final result ----
    tl.store(
        out_base + rm[:, None] * M + rn[None, :],
        tl.load(
            scratch_base + r_buf * scratch_stride + rm[:, None] * M + rn[None, :],
            mask=mask,
            other=0.0,
        ),
        mask=mask,
    )


@triton.jit
def _compute_tiled_matmul(
    A_base,
    B_base,
    C_base,
    M,
    rm,
    rn,
    offs_m,
    offs_k,
    offs_n,
    mask_c,
    TILE_BLOCK: tl.constexpr,
    TILES: tl.constexpr,
    FIX_FP64: tl.constexpr = False,
    G: tl.constexpr = 1,
):
    """Compute one tile of C = A @ B, storing result to C_base."""
    acc_dtype = A_base.type.element_ty
    if acc_dtype != tl.float64:
        acc_dtype = tl.float32
    acc = tl.zeros((TILE_BLOCK, TILE_BLOCK), dtype=acc_dtype)
    for tk in range(TILES):
        rk = tk * TILE_BLOCK + offs_k
        mask_a = (rm[:, None] < M) & (rk[None, :] < M)
        mask_b = (rk[:, None] < M) & (rn[None, :] < M)
        a_tile = tl.load(
            A_base + rm[:, None] * M + rk[None, :],
            mask=mask_a,
            other=0.0,
        )
        b_tile = tl.load(
            B_base + rk[:, None] * M + rn[None, :],
            mask=mask_b,
            other=0.0,
        )
        acc += tl.dot(a_tile.to(acc_dtype), b_tile.to(acc_dtype), allow_tf32=False)
    if FIX_FP64:
        # acc has been accumulated in the MetaX fp64 MMA layout; converting it
        # to blocked (which also un-scrambles its rows) must happen before the
        # mask is applied, otherwise the mask would corrupt it in MMA layout.
        acc = _metax_fix_fp64_rows(acc, TILE_BLOCK, TILE_BLOCK, G)
    acc = tl.where(mask_c, acc, 0.0)
    tl.store(C_base + rm[:, None] * M + rn[None, :], acc, mask=mask_c)


# ===========================================================================
# Thresholds for dispatch
# ===========================================================================


@triton.jit
def _metax_fp64_mm_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    G: tl.constexpr,
):
    """Tiled fp64 C = A @ B for MetaX, grid (tiles, batch).

    The MetaX backend's own mm kernels cannot lower fp64 tl.dot (their
    accumulator is hardwired to fp32, and its fp64 MMA cannot run with any
    dimension < 16), so the M > 256 host binary-exponentiation path uses this
    self-contained kernel instead.  Each program computes one BLOCK_M x BLOCK_N
    output tile; the fp64 dot result's rows are un-scrambled with
    ``_metax_fix_fp64_rows`` before the store.
    """
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_base = A + pid_b * stride_ab
    b_base = B + pid_b * stride_bb
    c_base = C + pid_b * stride_cb

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        kr = k0 * BLOCK_K + rk
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
    acc = _metax_fix_fp64_rows(acc, BLOCK_M, BLOCK_N, G)
    cmask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(
        c_base + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc.to(C.dtype.element_ty),
        mask=cmask,
    )


def _metax_fp64_mm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """fp64 matmul for MetaX (self-contained, see ``_metax_fp64_mm_kernel``)."""
    *_, M, K = A.shape
    *_, K2, N = B.shape
    assert K == K2
    A2 = A.reshape(-1, M, K)
    B2 = B.reshape(-1, K, N)
    batch = A2.shape[0]
    C = torch.empty(batch, M, N, dtype=A.dtype, device=A.device)
    # fp64 tiles are 8 bytes/element; 64x32 + 32x64 per K-step keeps shared
    # memory under the MetaX 64KB limit with a single pipeline stage.
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), batch)
    _metax_fp64_mm_kernel[grid](
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
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        G=BLOCK_M // 16,
        num_warps=4,
        num_stages=1,
    )
    return C.reshape(A.shape[:-2] + (M, N))


@triton.jit
def _trsm_update_register(
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
    BM: tl.constexpr,
    K_SLICE: tl.constexpr,
):
    """Update phase of _trsm_kernel: reuses the register tile x_all as the
    X panel and accumulates the update gemm in fp64 (fp32 operands kept in
    fp32 storage)."""
    # Reuse the register tile (x_all) as the X panel.
    for m_start in range(0, M_REM, BM):
        rm = rem_s + m_start + rr
        mask_m = rm < bound
        a_sub = tl.load(
            A_ptr + rm[:, None] * stride_a_n + (blk_start + a_cols)[None, :],
            mask=mask_m[:, None] & (a_cols[None, :] < blk_sz),
            other=0.0,
        )
        acc = tl.dot(
            a_sub.to(tl.float64),
            x_all.to(tl.float64),
            allow_tf32=False,
        )
        # Metax's fp64 MMA scrambles the rows of its output tile: un-scramble
        # before the store (G = BM // 16).
        acc = _metax_fix_fp64_rows(acc, BM, K_SLICE, 8)
        acc = acc.to(A_ptr.dtype.element_ty)
        b_base = B_ptr + rm[:, None] * stride_b_k + col_offs[None, :]
        b_curr = tl.load(b_base, mask=mask_m[:, None] & col_mask[None, :], other=0.0)
        b_curr = b_curr.to(acc.dtype) - acc
        tl.store(b_base, b_curr, mask=mask_m[:, None] & col_mask[None, :])


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
):
    """Blocked triangular solve A X = B in place (B <- X), one program per
    K_SLICE-column group of the RHS.

    Rows are processed in BLOCK_SIZE blocks.  The diagonal block is solved by
    serial forward/backward substitution (row-by-row, parallel across the
    K_SLICE columns), then the remaining rows are updated with a tl.dot gemm —
    every data dependency stays within a single program, so the kernel is
    barrier-free.  This replaces the (batch, n)-grid scalar substitution
    kernels, whose one-program-per-column serial O(n^2) loop was the dominant
    cost of the negative-power path (~77 ms per solve at n=1024 vs ~1 ms
    here).
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

        # Serial solve of the diagonal block: the block's X stays in a
        # register tile (x_all) across the serial row chain.
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

        # ═══ Update: B[rest, kslice] -= A[rest, blk] @ X[blk, kslice] ═══
        need_update = tl.where(UPPER, bk > 0, blk_end < N)
        if need_update:
            M_REM = tl.where(UPPER, blk_start, N - blk_end)
            rem_s = tl.where(UPPER, 0, blk_end)
            bound = tl.where(UPPER, blk_start, N)
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
            )


def _trsm_solve_2d(A_tri, B, upper: bool, unitriangular: bool):
    """Blocked triangular solve for metax: K_SLICE=16 (the fp64 MMA cannot
    lower dots with any dimension < 16) and a 1-stage software pipeline
    (64 KB shared memory); the update gemm un-scrambles its fp64 output rows
    via the private _trsm_update_register above."""
    n = A_tri.shape[0]
    k = B.shape[1]
    K_SLICE = 16
    BM = 128
    num_kslices = (k + K_SLICE - 1) // K_SLICE
    if unitriangular:
        inv = B  # INV_ptr is only dereferenced when UNIT is false
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
        num_warps=4,
        num_stages=1,
    )
    return B


def _matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Matmul via the flag_gems Triton kernels (mm for 2D, bmm for batched).

    Looked up on the flag_gems namespace at call time so backend-specialized
    kernels are used when a backend overrides them: iluvatar's mm kernel
    accepts a SPLIT_K tune config that the general mm kernel does not, so the
    general ``flag_gems.ops.mm`` import would fail there with a KeyError.

    On MetaX the backend mm kernels cannot lower fp64 tl.dot at all (fp32-only
    accumulator, and the fp64 MMA needs every dimension >= 16), so fp64 matmuls
    use the self-contained tiled kernel above.
    """
    from flag_gems import bmm, mm

    if True and A.dtype == torch.float64:
        return _metax_fp64_mm(A, B)
    if A.dim() == 2:
        return mm(A, B)
    return bmm(A, B)


# ===========================================================================
# Triangular solves.  The scalar substitution kernels below are the legacy
# (batch, n)-grid row loops kept for the df64 route (linalg_lu_solve); the
# fast path for every other backend is the blocked _trsm_kernel that follows
# (see its header for the per-block phase helpers).
# ===========================================================================


# Hook the shared hosts so every solve/matmul inside the generic hosts runs
# the metax-fixed kernels (one backend per process - safe).
_generic._trsm_solve_2d = _trsm_solve_2d
_generic._matmul = _matmul


def linalg_matrix_power(
    A: torch.Tensor,
    n: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    logger.debug("GEMS LINALG_MATRIX_POWER")

    # ---- validation ----
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

    # ---- n == 0 ----
    if n == 0:
        eye = _eye_like(A)
        if out is not None:
            out.copy_(eye)
            return out
        return eye

    # ---- n == 1: A¹ = A — plain copy, no computation / kernel launch ----
    if n == 1:
        if out is not None:
            out.copy_(A)
            return out
        return A.clone()

    # ---- flag_gems Triton kernels are CUDA-only; n==0/n==1 above work on any
    # device, every computational path below requires the flag_gems device. ----
    if A.device.type != flag_gems.device:
        raise RuntimeError(
            f"linalg_matrix_power: flag_gems supports only {flag_gems.device}, "
            f"got {A.device}"
        )

    # ---- negative n ----
    upcast = False
    if n < 0:
        # f32 negatives: matrices above 512 use the f32-compute path — the
        # inverse is computed in f32 (fp32 storage + fp64 accumulation in the
        # LU/TRSM updates, fp64-accumulation Newton refinement), which is faster
        # once the barrier-bound LU is amortised.  Smaller f32 matrices keep the
        # fp64 upcast (the refinement overhead is not amortised there).
        # The power (A⁻¹)^|n| always runs in fp64 and the result is cast to f32.
        upcast = A.dtype == torch.float32
        if upcast and A.shape[-1] <= 512:
            A = A.double()
        A = _inverse(A)
        n = -n
        if upcast:
            A = A.double()

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
            FIX_FP64=A.dtype == torch.float64,
            G=BLOCK // 16,
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
            FIX_FP64=A.dtype == torch.float64,
            G=BLOCK // 16,
        )

    elif m <= 256 and A.device.type == flag_gems.device:
        # Tier 3: grid-level sync fused (65 <= M <= 256).
        TILES = triton.cdiv(m, TILE)
        # Fresh buffers per call: the kernel's Step 0 fully overwrites every
        # scratch slot it reads, and the round-based barrier logic works from a
        # zero-initialized counter (0 is a multiple of n_total), so nothing
        # needs to persist across calls.  Allocating fresh also keeps concurrent
        # calls on different streams from racing on shared buffers, and avoids
        # an ever-growing barrier counter that could overflow int32.
        scratch = torch.empty(4 * batch_size, m, m, dtype=A.dtype, device=A.device)
        barrier = torch.zeros(batch_size * 64, dtype=torch.int32, device=A.device)
        _grid_sync_kernel[(batch_size, TILES, TILES)](
            A_flat,
            out_flat,
            scratch,
            barrier,
            m,
            n,
            batch_stride,
            TILE_BLOCK=TILE,
            TILES=TILES,
            FIX_FP64=A.dtype == torch.float64,
            G=TILE // 16,
        )

    else:
        # M > 256: host-side binary exponentiation with the flag_gems Triton
        # matmul kernels (mm for 2D, bmm for batched), one launch per step.
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
