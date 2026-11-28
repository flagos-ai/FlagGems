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

import logging
import math

import torch
import triton
import triton.language as tl

# FlagGems CUDA-native computational operators (replace torch.abs/torch.all/torch.max/
# torch.min/torch.norm/torch.sqrt/torch.sum/torch.topk/torch.amax with FlagGems equivalents).
from flag_gems.ops.max import max as gems_max
from flag_gems.ops.min import min as gems_min
from flag_gems.ops.sqrt import sqrt as gems_sqrt
from flag_gems.ops.sum import sum_dim
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

# ===========================================================================
# Kernel: _fro_kernel -- unified Frobenius norm (sqrt(Σx²)).
# Grid=(batch,) for TILE_2D=False (per-row), Grid=(grid_m×grid_n,) for TILE_2D=True (tiled).
# ===========================================================================


@libentry()
@triton.jit
def _fro_kernel(
    X,
    Out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GRID_N: tl.constexpr,
    TILE_2D: tl.constexpr,
    USE_FP64: tl.constexpr = True,
):
    """Unified Frobenius norm kernel: sqrt(Σ x²).

    Two modes selected by TILE_2D:
      TILE_2D=False (1D per-row):
        Grid=(batch,).  Each program handles one row / one matrix,
        loops over N elements in BLOCK_N chunks, stores sqrt(Σx²)
        to Out[pid].  Used for batched inputs and small 2D matrices.

      TILE_2D=True (2D tiled):
        Grid=(grid_m × grid_n,).  Each program processes one
        BLOCK_M×BLOCK_N tile, atomically adds Σx² to Out[0].
        Host takes sqrt.  Used for large single matrices (M·N > 65536).
    """
    c_dtype = X.dtype.element_ty
    if c_dtype != tl.float64:
        c_dtype = tl.float32
    # Accumulate sum-of-squares in fp64 when available: fp32 summation error
    # grows as N * eps ≈ 4e-3 (before sqrt) for N=65536 in the 1D path, and
    # fp32 atomic_add across tiles gives ~2e-6 in the 2D path.  fp64
    # accumulation eliminates both noise floors so the Frobenius norm
    # meets CPU-LAPACK comparison tolerances regardless of path.
    # Backends without fp64 support (iluvatar, ascend, etc.) use fp32.
    acc_dtype = tl.float64 if USE_FP64 else tl.float32
    if TILE_2D:
        # --- 2D tiled mode: one matrix, many tile-blocks ---
        pid = tl.program_id(0)
        pid_m = pid // GRID_N
        pid_n = pid % GRID_N
        row_start = pid_m * BLOCK_M
        col_start = pid_n * BLOCK_N
        rows = row_start + tl.arange(0, BLOCK_M)[:, None]
        cols = col_start + tl.arange(0, BLOCK_N)[None, :]
        mask = (rows < M) & (cols < N)
        x = tl.load(X + rows * N + cols, mask=mask, other=0.0).to(acc_dtype)
        tile_sum = tl.sum(x * x)
        tl.atomic_add(Out, tile_sum)
    else:
        # --- 1D per-row mode: batch rows, each program = one row ---
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK_N)
        acc = tl.zeros([BLOCK_N], dtype=acc_dtype)
        for start in range(0, N, BLOCK_N):
            idx = start + offs
            mask = idx < N
            x = tl.load(X + pid * N + idx, mask=mask, other=0.0).to(acc_dtype)
            acc += tl.where(mask, x * x, 0.0)
        total = tl.sqrt(tl.sum(acc))
        tl.store(Out + pid, total.to(Out.dtype.element_ty))


# ===========================================================================
# Kernel: _rank2_svals_kernel -- closed-form SVD for k=2.
# Used by _svdvals_for_norm and C++ SVD dispatch.  No iteration needed.
# BLOCK_B=1 → one matrix per program (regular).
# BLOCK_B>1 → BLOCK_B matrices per program (vectorized, for tiny rows).
# ===========================================================================

_RANK2_BLOCK_R_MAX = 2048


def _use_fp64_acc(input_dtype):
    """Whether to use fp64 for internal accumulation in reduction kernels.

    - f32/f64 → fp64 accumulator (this generic file only runs on fp64-capable
      backends; the per-vendor backend files apply their own fp64 policy)
    - bf16/f16 → fp32 accumulator (fp64 is overkill for half precision)
    """
    # The GCU400 compiler marks fp32->fp64 arith.extf as an illegal op and segfaults at compile time;
    # this backend uniformly accumulates in fp32 to avoid generating fp64 extension instructions.
    return False


def _acc_dtype(input_dtype):
    """Accumulator dtype for host-side reduction buffers.

    Mirrors ``_use_fp64_acc`` but returns a concrete torch dtype for use
    with ``torch.zeros / torch.full``.
    """
    if _use_fp64_acc(input_dtype):
        return torch.float64
    return torch.float32


def _svd_shape(A):
    """Return (batch, M, N) for an SVD-shaped tensor."""
    if A.ndim < 2:
        return 0, 0, 0
    batch = 1
    for d in A.shape[:-2]:
        batch *= d
    return batch, A.shape[-2], A.shape[-1]


@libentry()
@triton.jit
def _rank2_svals_kernel(
    A,
    S,
    Out,
    BATCH: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    TALL: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_R: tl.constexpr,
    MODE: tl.constexpr,
):
    """Closed-form SVD for k=2. BLOCK_B matrices per program.

    MODE 0: plain svdvals -- both singular values stored to S (batch, 2).
    MODE 1/2/3: ord=2 / ord=-2 / nuc -- the requested norm value is stored
    straight into Out[b] with an in-kernel dtype cast, so no host-side
    slice + copy epilogue is needed."""
    pid = tl.program_id(0)
    eps = 1.0e-20
    c_dtype = A.type.element_ty
    if c_dtype != tl.float64:
        c_dtype = tl.float32

    if BLOCK_B == 1:
        # --- Single matrix per program (regular path) ---
        offs = tl.arange(0, BLOCK_R)
        if TALL:
            mask = offs < M
            base = A + pid * M * N
            x = tl.load(base + offs * N, mask=mask, other=0.0).to(c_dtype)
            y = tl.load(base + offs * N + 1, mask=mask, other=0.0).to(c_dtype)
        else:
            mask = offs < N
            base = A + pid * M * N
            x = tl.load(base + offs, mask=mask, other=0.0).to(c_dtype)
            y = tl.load(base + N + offs, mask=mask, other=0.0).to(c_dtype)

        aa = tl.sum(x * x)
        bbv = tl.sum(y * y)
        ab = tl.sum(x * y)
        diff = aa - bbv
        root = tl.sqrt(diff * diff + 4.0 * ab * ab)
        l0 = tl.maximum(0.0, 0.5 * (aa + bbv + root))
        det = tl.maximum(0.0, aa * bbv - ab * ab)
        l1 = tl.where(l0 > eps, det / l0, 0.0)

        if MODE == 0:
            sbase = S + pid * 2
            tl.store(sbase, tl.sqrt(l0))
            tl.store(sbase + 1, tl.sqrt(l1))
        elif MODE == 1:
            tl.store(Out + pid, tl.sqrt(l0).to(Out.dtype.element_ty))
        elif MODE == 2:
            tl.store(Out + pid, tl.sqrt(l1).to(Out.dtype.element_ty))
        else:
            tl.store(
                Out + pid,
                (tl.sqrt(l0) + tl.sqrt(l1)).to(Out.dtype.element_ty),
            )

    else:
        # --- BLOCK_B matrices per program (vectorized, tiny rows) ---
        b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        r = tl.arange(0, BLOCK_R)
        bb = b[:, None]
        rr = r[None, :]
        bmask = b < BATCH

        if TALL:
            mask = (bb < BATCH) & (rr < M)
            base = A + bb * M * N + rr * N
            x = tl.load(base, mask=mask, other=0.0).to(c_dtype)
            y = tl.load(base + 1, mask=mask, other=0.0).to(c_dtype)
        else:
            mask = (bb < BATCH) & (rr < N)
            base = A + bb * M * N + rr
            x = tl.load(base, mask=mask, other=0.0).to(c_dtype)
            y = tl.load(base + N, mask=mask, other=0.0).to(c_dtype)

        aa = tl.sum(x * x, axis=1)
        bbv = tl.sum(y * y, axis=1)
        ab = tl.sum(x * y, axis=1)
        diff = aa - bbv
        root = tl.sqrt(diff * diff + 4.0 * ab * ab)
        l0 = tl.maximum(0.0, 0.5 * (aa + bbv + root))
        det = tl.maximum(0.0, aa * bbv - ab * ab)
        l1 = tl.where(l0 > 1.0e-20, det / l0, 0.0)
        if MODE == 0:
            tl.store(S + b * 2, tl.sqrt(l0), mask=bmask)
            tl.store(S + b * 2 + 1, tl.sqrt(l1), mask=bmask)
        elif MODE == 1:
            tl.store(Out + b, tl.sqrt(l0).to(Out.dtype.element_ty), mask=bmask)
        elif MODE == 2:
            tl.store(Out + b, tl.sqrt(l1).to(Out.dtype.element_ty), mask=bmask)
        else:
            tl.store(
                Out + b,
                (tl.sqrt(l0) + tl.sqrt(l1)).to(Out.dtype.element_ty),
                mask=bmask,
            )


logger = logging.getLogger(__name__)

_SUPPORTED_NUMERIC = {1, -1, 2, -2, float("inf"), -float("inf")}

# Output dtypes the GCU400 kernels can store directly (in-kernel fp32->x
# cast at store).  fp64 needs the host-cast fallback (no fp64 arithmetic
# on this backend).
_FLOAT_CAST_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


# ===========================================================================
# Unified abs-norm kernel -- parameterized by SUM_AXIS, IS_MIN, TILED, BATCHED.
#
# SUM_AXIS=0  → 1-norm style  (reduce along rows, per-column result)
# SUM_AXIS=1  → inf-norm style (reduce along cols, per-row result)
# IS_MIN=False → max (for positive ord), IS_MIN=True → min (for negative ord)
# TILED=True   → 2D grid, atomic_add to partial buffer (large single matrix)
# TILED=False  → 1D stripe, atomic_max/atomic_min directly to Out
# BATCHED=True → 2D grid=(batch, grid_dim), output to Out[batch_idx]
#                (only used with TILED=False; TILED+BATCHED is not combined)
# ===========================================================================


@libentry()
@triton.jit
def _abs_norm_kernel(
    X,
    Out,
    Partial,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GRID_N: tl.constexpr,
    SUM_AXIS: tl.constexpr,
    IS_MIN: tl.constexpr,
    TILED: tl.constexpr,
    BATCHED: tl.constexpr = False,
    USE_FP64: tl.constexpr = False,
):
    # Accumulate in fp64 when USE_FP64=True for better precision on f32/f64
    # inputs.  bf16/f16 inputs always use fp32 (USE_FP64=False).
    # Final atomic_min/atomic_max still require fp32 on most hardware.
    acc_dtype = tl.float64 if USE_FP64 else tl.float32

    if TILED:
        # --- 2D tiled: atomic_add to Partial (single matrix, not batched) ---
        pid = tl.program_id(0)
        pid_m = pid // GRID_N
        pid_n = pid % GRID_N
        row_start = pid_m * BLOCK_M
        col_start = pid_n * BLOCK_N
        rows = row_start + tl.arange(0, BLOCK_M)[:, None]
        cols = col_start + tl.arange(0, BLOCK_N)[None, :]
        row_mask = rows < M
        col_mask = cols < N
        x = tl.load(X + rows * N + cols, mask=row_mask & col_mask, other=0.0).to(
            acc_dtype
        )

        if SUM_AXIS == 0:
            col_sum = tl.sum(tl.abs(x), axis=0)
            off = col_start + tl.arange(0, BLOCK_N)
            tl.atomic_add(Partial + off, col_sum)
        else:
            row_sum = tl.sum(tl.abs(x), axis=1)
            off = row_start + tl.arange(0, BLOCK_M)
            tl.atomic_add(Partial + off, row_sum)

    else:
        # --- 1D stripe: atomic_max / atomic_min to Out[batch_idx] ---
        if BATCHED:
            batch_idx = tl.program_id(0)
            block_idx = tl.program_id(1)
            base = batch_idx * M * N
            out_ptr = Out + batch_idx
        else:
            batch_idx = 0
            block_idx = tl.program_id(0)
            base = 0
            out_ptr = Out

        if SUM_AXIS == 0:
            # 1norm: per-column stripe
            block_start = block_idx * BLOCK_N
            cols = block_start + tl.arange(0, BLOCK_N)
            col_mask = cols < N
            acc = tl.zeros([BLOCK_N], dtype=acc_dtype)
            for row_start in range(0, M, BLOCK_M):
                rows = row_start + tl.arange(0, BLOCK_M)[:, None]
                mask = (rows < M) & col_mask[None, :]
                x = tl.load(
                    X + base + rows * N + cols[None, :], mask=mask, other=0.0
                ).to(acc_dtype)
                acc += tl.sum(tl.abs(x), axis=0)
            # atomic_max / atomic_min require fp32 on most hardware;
            # convert to fp32 for the final reduction.
            acc_f32 = acc.to(tl.float32)
            if IS_MIN:
                acc_f32 = tl.where(col_mask, acc_f32, float("inf"))
                tl.atomic_min(out_ptr, tl.min(acc_f32))
            else:
                tl.atomic_max(out_ptr, tl.max(acc_f32))

        else:
            # infnorm: per-row stripe
            row_start = block_idx * BLOCK_M
            rows = row_start + tl.arange(0, BLOCK_M)[:, None]
            row_mask = rows < M
            acc = tl.zeros([BLOCK_M, 1], dtype=acc_dtype)
            for col_start in range(0, N, BLOCK_N):
                cols = col_start + tl.arange(0, BLOCK_N)[None, :]
                mask = row_mask & (cols < N)
                x = tl.load(X + base + rows * N + cols, mask=mask, other=0.0).to(
                    acc_dtype
                )
                acc += tl.sum(tl.abs(x), axis=1)[:, None]
            acc_f32 = acc.to(tl.float32)
            if IS_MIN:
                acc_f32 = tl.where(row_mask, acc_f32, float("inf"))
                tl.atomic_min(out_ptr, tl.min(acc_f32))
            else:
                tl.atomic_max(out_ptr, tl.max(acc_f32))


# ---------------------------------------------------------------------------
# Fast two-phase abs-norm kernels (GCU400):
#   Phase 1 `_abs_norm_v2_kernel`: one program per (matrix, main-axis block);
#     loops over the other axis, plain-stores per-row/per-col abs sums to a
#     scratch buffer (no atomics -- vector atomic_add costs ~27ns/lane-op and
#     serializes to >100us on mid-size matrices), or -- when the matrix fits
#     a single block per matrix -- reduces in-program and stores the final
#     norm straight into Out with an in-kernel dtype cast.
#   Phase 2 `_abs_norm_final_kernel`: max/min over the scratch row plus cast.
# Both phases load fp16/bf16/fp32 and accumulate in fp32.
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def _abs_norm_v2_kernel(
    X,
    Out,
    Partial,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GRID_MAIN: tl.constexpr,
    SUM_AXIS: tl.constexpr,
    IS_MIN: tl.constexpr,
    SINGLE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // GRID_MAIN
    blk = pid % GRID_MAIN
    base = b * M * N
    if SUM_AXIS == 1:
        # inf-norm: row sums.  main axis = rows, loop over columns.
        rows = blk * BLOCK_M + tl.arange(0, BLOCK_M)
        rmask = rows < M
        acc = tl.zeros([BLOCK_M], dtype=tl.float32)
        for c0 in range(0, N, BLOCK_N):
            cols = c0 + tl.arange(0, BLOCK_N)
            x = tl.load(
                X + base + rows[:, None] * N + cols[None, :],
                mask=rmask[:, None] & (cols < N)[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(tl.abs(x), axis=1)
        if SINGLE:
            acc = tl.where(rmask, acc, float("inf") if IS_MIN else 0.0)
            v = tl.min(acc) if IS_MIN else tl.max(acc)
            tl.store(Out + b, v.to(Out.dtype.element_ty))
        else:
            tl.store(Partial + b * M + rows, acc, mask=rmask)
    else:
        # 1-norm: column sums.  main axis = columns, loop over rows.
        cols = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        cmask = cols < N
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for r0 in range(0, M, BLOCK_M):
            rows = r0 + tl.arange(0, BLOCK_M)
            x = tl.load(
                X + base + rows[:, None] * N + cols[None, :],
                mask=(rows < M)[:, None] & cmask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(tl.abs(x), axis=0)
        if SINGLE:
            acc = tl.where(cmask, acc, float("inf") if IS_MIN else 0.0)
            v = tl.min(acc) if IS_MIN else tl.max(acc)
            tl.store(Out + b, v.to(Out.dtype.element_ty))
        else:
            tl.store(Partial + b * N + cols, acc, mask=cmask)


@libentry()
@triton.jit
def _abs_norm_final_kernel(
    Partial,
    Out,
    L,
    BLOCK_L: tl.constexpr,
    IS_MIN: tl.constexpr,
):
    b = tl.program_id(0)
    base = b * L
    init = float("inf") if IS_MIN else float("-inf")
    run = init
    for l0 in range(0, L, BLOCK_L):
        idx = l0 + tl.arange(0, BLOCK_L)
        v = tl.load(Partial + base + idx, mask=idx < L, other=init)
        m = tl.min(v) if IS_MIN else tl.max(v)
        run = tl.minimum(run, m) if IS_MIN else tl.maximum(run, m)
    tl.store(Out + b, run.to(Out.dtype.element_ty))


# ---------------------------------------------------------------------------
# Fast two-phase Frobenius kernels for large single matrices (M*N > 65536):
#   `_fro_v2_kernel`: row-block partial sum of squares, plain store (no
#     atomics); `_fro_final_kernel`: sum + sqrt + cast in one tiny launch.
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def _fro_v2_kernel(
    X,
    Partial,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GRID_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < M
    acc = 0.0
    for c0 in range(0, N, BLOCK_N):
        cols = c0 + tl.arange(0, BLOCK_N)
        x = tl.load(
            X + rows[:, None] * N + cols[None, :],
            mask=rmask[:, None] & (cols < N)[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(x * x)
    tl.store(Partial + pid, acc)


@libentry()
@triton.jit
def _fro_final_kernel(
    Partial,
    Out,
    L,
    BLOCK_L: tl.constexpr,
):
    acc = 0.0
    for l0 in range(0, L, BLOCK_L):
        idx = l0 + tl.arange(0, BLOCK_L)
        acc += tl.sum(tl.load(Partial + idx, mask=idx < L, other=0.0))
    tl.store(Out, tl.sqrt(acc).to(Out.dtype.element_ty))


# ---------------------------------------------------------------------------
# Nuclear-norm sigma-sum: sums the (batch, K) singular-value matrix produced
# by the SVD path along K and stores per-batch results with a dtype cast,
# replacing the generic FlagGems sum (whose Python wrapper costs ~5us).
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def _nuc_sum_kernel(S, Out, K, BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    base = b * K
    acc = 0.0
    for k0 in range(0, K, BLOCK_K):
        idx = k0 + tl.arange(0, BLOCK_K)
        acc += tl.sum(tl.load(S + base + idx, mask=idx < K, other=0.0))
    tl.store(Out + b, acc.to(Out.dtype.element_ty))


# ---------------------------------------------------------------------------
# Per-ord helper functions -- each mirrors a path in the C++ dispatch
# ---------------------------------------------------------------------------


def _fro_norm(A, dim, keepdim, dtype, out=None):
    """Frobenius norm (ord=\"fro\").  Mirrors C++ ``fro_norm``.

    Uses the unified ``_fro_kernel``: TILE_2D=False for small/batched
    (per-row L2, grid=(batch,)), TILE_2D=True for large 2D (tiled + atomic,
    host sqrt).
    """
    d0, d1 = dim
    out_dtype = dtype if dtype is not None else A.dtype

    # Simple 2D case -- same kernel path as C++ fro_norm.
    if A.ndim == 2 and d0 == 0 and d1 == 1:
        M, N = A.shape
        total = M * N
        use_fp64 = _use_fp64_acc(A.dtype)
        if total <= 65536:
            # GCU400 has no fp64 arithmetic: an fp64 Out buffer would make the
            # kernel emit an in-kernel fp32->fp64 extf, which produces garbage
            # (-3.7e19) on this backend.  Direct kernel writes accept
            # fp32/fp16/bf16 out (in-kernel downcast is fine); anything else
            # computes into an fp32 buffer and casts on the host.
            direct = (
                out is not None
                and out.dtype in _FLOAT_CAST_DTYPES
                and out.is_contiguous()
                and out.numel() == 1
            )
            buf = (
                out.reshape(-1)
                if direct
                else torch.empty(1, dtype=torch.float32, device=A.device)
            )
            flat = A.reshape(1, total)
            _fro_kernel[(1,)](
                flat,
                buf,
                0,
                total,
                1,
                512,
                1,
                TILE_2D=False,
                USE_FP64=use_fp64,
                num_warps=8,
            )
            if direct:
                return out
            result = buf.view(())
        elif (
            A.is_contiguous()
            and out_dtype in _FLOAT_CAST_DTYPES
            and (out is None or (out.is_contiguous() and out.numel() == 1))
        ):
            # Fast two-phase path: row-block partial sums of squares (plain
            # stores, no atomics), then one tiny fused sum + sqrt + cast
            # kernel writing straight into the out/result buffer.
            bn = triton.next_power_of_2(min(N, 512))
            bm = triton.next_power_of_2(max(1, min(M, 4096 // bn)))
            grid_m = triton.cdiv(M, bm)
            direct = out is not None
            buf = (
                out.reshape(-1)
                if direct
                else torch.empty(1, dtype=out_dtype, device=A.device)
            )
            partial = torch.empty(grid_m, dtype=torch.float32, device=A.device)
            _fro_v2_kernel[(grid_m,)](
                A,
                partial,
                M,
                N,
                bm,
                bn,
                grid_m,
                num_warps=4,
            )
            _fro_final_kernel[(1,)](
                partial,
                buf,
                grid_m,
                triton.next_power_of_2(min(grid_m, 1024)),
                num_warps=4,
            )
            if direct:
                return out
            result = buf.view(())
        else:
            if M <= 1024 and N <= 1024:
                BM, BN = 64, 64
            elif N >= 8 * M or M >= 8 * N:
                BM, BN = 128, 128
            else:
                BM, BN = 128, 32
            grid_m = triton.cdiv(M, BM)
            grid_n = triton.cdiv(N, BN)
            grid_size = int(grid_m * grid_n)
            # Atomic accumulator must be fp32/fp64; the host sqrt+cast below is
            # the last step, so out= is finalized by the impl fallback copy.
            acc = torch.zeros((), dtype=_acc_dtype(A.dtype), device=A.device)
            _fro_kernel[(grid_size,)](
                A,
                acc,
                M,
                N,
                BM,
                BN,
                grid_n,
                TILE_2D=True,
                USE_FP64=use_fp64,
                num_warps=8,
            )
            result = gems_sqrt(acc).to(out_dtype)
        if result.dtype != out_dtype:
            result = result.to(out_dtype)
        if keepdim:
            result = result.reshape(1, 1)
        return result

    # Batched case
    ndim = A.ndim
    all_dims = list(range(ndim))
    remaining = [d for d in all_dims if d not in (d0, d1)]
    perm = remaining + [d0, d1]
    A_perm = A.permute(perm)
    batch = 1
    for i in range(A_perm.ndim - 2):
        batch *= A_perm.size(i)
    mat_size = A_perm.size(-2) * A_perm.size(-1)
    # A_perm is non-contiguous; reshape on a non-contiguous (...,1) view can
    # yield a strided (batch, mat_size) view instead of a copy, breaking the
    # _fro_kernel's contiguous linear indexing.  Force a contiguous copy.
    flat = A_perm.reshape(batch, mat_size).contiguous()

    # fp32 kernel buffer + fp32/fp16/bf16 direct write (same GCU400 fp64 rule
    # as the simple 2D path above); cast to out_dtype on the host afterwards.
    direct = (
        out is not None
        and out.dtype in _FLOAT_CAST_DTYPES
        and out.is_contiguous()
        and out.numel() == batch
    )
    buf = (
        out.reshape(-1)
        if direct
        else torch.empty(batch, dtype=torch.float32, device=flat.device)
    )
    blk_n = triton.next_power_of_2(min(mat_size, 1024))
    _fro_kernel[(batch,)](
        flat,
        buf,
        0,
        mat_size,
        1,
        blk_n,
        1,
        TILE_2D=False,
        USE_FP64=_use_fp64_acc(A.dtype),
        num_warps=8,
    )
    if direct:
        return out

    result = buf
    if result.dtype != out_dtype:
        result = result.to(out_dtype)
    if keepdim:
        out_shape = list(A.shape)
        out_shape[d0] = 1
        out_shape[d1] = 1
        result = result.reshape(out_shape)
    else:
        batch_shape = [A.size(i) for i in range(ndim) if i != d0 and i != d1]
        result = result.reshape(batch_shape)
    return result


def _ord2_norm(A, ord_val, dim, keepdim, dtype, out=None):
    """Spectral norm (ord=2 / ord=-2).  Mirrors C++ inline dispatch.

    Fast path (GCU400): reducing the last two dims of a contiguous
    fp16/bf16/fp32 input, the whole norm is produced by the SVD kernels
    writing straight into the (possibly half-precision) out/result buffer
    -- rank-2 closed form for k=2, Gram + tridiag + Sturm otherwise; the
    Sturm kernel's owning lane stores the max/min sigma directly, so there
    is no host-side slice + copy epilogue.  Anything else (odd dims, fp64)
    falls back to the generic permute + svdvals path.
    """
    d0, d1 = dim
    out_dtype = dtype if dtype is not None else A.dtype
    ndim = A.ndim

    fast = (
        ndim >= 2
        and A.is_contiguous()
        and A.dtype in _FLOAT_CAST_DTYPES
        and out_dtype in _FLOAT_CAST_DTYPES
        and (d0, d1) in ((ndim - 2, ndim - 1), (ndim - 1, ndim - 2))
        and (out is None or (out.is_contiguous() and out.numel() >= 1))
    )
    # ord=-2 needs the *smallest* singular value.  The Gram matrix squares it,
    # so in the fp32-only arithmetic of this backend it becomes a difference of
    # O(||A||^2) entries and loses its relative accuracy (measured on GCU400 vs
    # CPU fp64: 5.9 % error at 256x256, 17.5 % at 512x512, while ord=2 stays at
    # 1e-7).  Route the fp32 minimum through the backend's Jacobi SVD kernels,
    # which keep every singular value relatively accurate (measured 3e-5 / 9e-5
    # / 3e-5 on the same shapes).  Other dtypes keep the in-file path (the
    # svdvals kernels are float32-only).
    if ord_val < 0 and A.dtype == torch.float32:
        fast = False

    if fast:
        batch = 1
        for i in range(ndim - 2):
            batch *= A.shape[i]
        M, N = A.shape[-2], A.shape[-1]
        if batch * M * N >= 2**31:
            fast = False

    if fast:
        want_max = ord_val > 0
        direct = out is not None
        buf = (
            out.reshape(-1)
            if direct
            else torch.empty(batch, dtype=out_dtype, device=A.device)
        )
        _svd_norm_impl(A, batch, M, N, 1 if want_max else 2, buf)
        if direct:
            return out
        result = buf
        if keepdim:
            out_shape = list(A.shape)
            out_shape[d0] = 1
            out_shape[d1] = 1
            result = result.reshape(out_shape)
        else:
            batch_shape = [A.size(i) for i in range(ndim) if i != d0 and i != d1]
            result = result.reshape(batch_shape)
        return result

    # Generic path: permute target dims to the last two positions.
    all_dims = list(range(ndim))
    remaining = [d for d in all_dims if d not in (d0, d1)]
    perm = remaining + [d0, d1]
    A_perm = A.permute(perm) if perm != all_dims else A
    if dtype is not None:
        A_perm = A_perm.to(dtype)

    if ord_val < 0 and A_perm.dtype == torch.float32:
        # Accurate smallest singular value: the Jacobi SVD works on A directly
        # instead of the fp32 Gram matrix (see the note above).  The gcu400
        # svdvals override also folds its grid to 1-D for the grid.y <= 255
        # limit, so it runs for every k this path reaches; composing a FlagGems
        # op keeps the computation in device kernels.
        from .linalg_svdvals import linalg_svdvals as _device_svdvals

        s = _device_svdvals(A_perm.contiguous())
    else:
        s = _svdvals_for_norm(A_perm)  # descending singular values, (..., K)
    # Every svdvals path returns descending order (gram-tridiag Sturm, rank-2,
    # k==1), so spectral norm is just the first/last entry -- no reduction
    # kernel launch required (amax/-amin each cost ~0.1-0.25ms of launch
    # overhead on small batched tensors).
    result = s[..., 0] if ord_val > 0 else s[..., -1]

    if result.dtype != out_dtype:
        result = result.to(out_dtype)
    if keepdim:
        out_shape = list(A.shape)
        out_shape[d0] = 1
        out_shape[d1] = 1
        result = result.reshape(out_shape)
    return result


def _choose_fast_tile(M, N):
    """Shared tile size selection. Returns (BM, BN, grid_m, grid_n)."""
    if M <= 1024 and N <= 1024:
        BM, BN = 32, 32
    elif N >= 8 * M or M >= 8 * N:
        BM, BN = min(M, 128), min(N, 128)
    else:
        BM, BN = 128, 32
    if BM > M:
        BM = triton.next_power_of_2(M)  # tl.arange requires pow2
    if BN > N:
        BN = triton.next_power_of_2(N)
    grid_m = triton.cdiv(M, BM)
    grid_n = triton.cdiv(N, BN)
    return BM, BN, int(grid_m), int(grid_n)


def _batched_kernel_dispatch(A, dim, ord_val, out_dtype, keepdim, out=None):
    d0, d1 = dim
    ndim = A.ndim
    all_dims = list(range(ndim))
    remaining = [d for d in all_dims if d != d0 and d != d1]
    perm = remaining + [d0, d1]
    A_perm = A.permute(perm)
    batch = 1
    for i in range(A_perm.ndim - 2):
        batch *= A_perm.size(i)
    mat_M = A_perm.size(-2)
    mat_N = A_perm.size(-1)
    Ab = A_perm.reshape(batch, mat_M, mat_N).contiguous()

    is_min = ord_val < 0
    abs_ord = abs(float(ord_val))
    use_fp64 = _use_fp64_acc(Ab.dtype)
    # _abs_norm_kernel's atomic_min/max need an fp32 buffer; when ``out`` is
    # fp32 (the atomic store dtype) we write straight into it (zero-copy).
    direct = (
        out is not None
        and out.dtype == torch.float32
        and out.is_contiguous()
        and out.numel() == batch
    )

    if math.isinf(abs_ord):
        # --- inf/-inf: multi-block per matrix (row-parallel) ---
        tile_m = 16
        grid_dim = triton.cdiv(mat_M, tile_m)
        blk_dim = triton.next_power_of_2(min(mat_N, 256))
        init_val = float("inf") if is_min else 0.0
        # Output buffer must be fp32: tl.atomic_min / tl.atomic_max only
        # support fp32.  USE_FP64 controls internal summation precision.
        result = (
            out.reshape(-1)
            if direct
            else torch.full((batch,), init_val, dtype=torch.float32, device=Ab.device)
        )
        if direct:
            result.fill_(init_val)
        _dummy = torch.empty(1, device=Ab.device)
        _abs_norm_kernel[(batch, grid_dim)](
            Ab,
            result,
            _dummy,
            mat_M,
            mat_N,
            tile_m,
            blk_dim,
            1,
            SUM_AXIS=1,
            IS_MIN=is_min,
            TILED=False,
            BATCHED=True,
            USE_FP64=use_fp64,
            num_warps=8,
        )
    elif abs_ord == 1.0:
        # --- 1/-1: multi-block per matrix (column-parallel) ---
        tile_n_raw = min(mat_N, 128)
        tile_n = triton.next_power_of_2(tile_n_raw)
        grid_dim = triton.cdiv(mat_N, tile_n_raw)
        blk_dim = triton.next_power_of_2(min(mat_M, 32))
        init_val = float("inf") if is_min else 0.0
        result = (
            out.reshape(-1)
            if direct
            else torch.full((batch,), init_val, dtype=torch.float32, device=Ab.device)
        )
        if direct:
            result.fill_(init_val)
        _dummy = torch.empty(1, device=Ab.device)
        _abs_norm_kernel[(batch, grid_dim)](
            Ab,
            result,
            _dummy,
            mat_M,
            mat_N,
            blk_dim,
            tile_n,
            1,
            SUM_AXIS=0,
            IS_MIN=is_min,
            TILED=False,
            BATCHED=True,
            USE_FP64=use_fp64,
            num_warps=8,
        )
    else:
        raise RuntimeError(f"_batched_kernel_dispatch: unsupported ord {ord_val}")

    if direct:
        return out

    if result.dtype != out_dtype:
        result = result.to(out_dtype)

    if keepdim:
        out_shape = list(A.shape)
        out_shape[d0] = 1
        out_shape[d1] = 1
        result = result.reshape(out_shape)
    else:
        batch_shape = [A.size(i) for i in range(ndim) if i != d0 and i != d1]
        result = result.reshape(batch_shape)
    return result


def _abs_norm_lastdim_ok(A, d0, d1, out_dtype, out):
    """Whether the GCU400 fast two-phase abs-norm path applies:
    contiguous fp16/bf16/fp32 input reducing (either order of) the last two
    dims, fp32-or-finer output, and int32-safe linear indexing."""
    ndim = A.ndim
    if (
        not A.is_contiguous()
        or A.dtype not in _FLOAT_CAST_DTYPES
        or out_dtype not in _FLOAT_CAST_DTYPES
        or (d0, d1) not in ((ndim - 2, ndim - 1), (ndim - 1, ndim - 2))
        or (out is not None and not out.is_contiguous())
    ):
        return False
    total = 1
    for s in A.shape:
        total *= s
        if total >= 2**31:
            return False
    return True


def _abs_norm_impl(A, d0, d1, ord_val, out_dtype, keepdim, out, ndim):
    """Fast ord=1/-1/inf/-inf for reductions over the last two dims.

    Two-phase, atomic-free design (vector atomic_add measured ~27ns/lane-op
    and >100us on mid-size matrices; the old tiled/stripe paths also relied
    on aten zeros/fill_/copy_ around the kernel):
      phase 1 `_abs_norm_v2_kernel`: one program per (matrix, main-axis
      block); plain-stores per-row (inf) / per-column (1) abs sums, or --
      when the matrix fits one block per matrix -- reduces in-program and
      stores the final norm straight into Out with an in-kernel dtype cast;
      phase 2 `_abs_norm_final_kernel`: max/min over the scratch vector.
    Swapped (d0, d1) = (last, second-last) dims transpose the matrix, which
    exchanges the roles of the 1-norm and inf-norm reductions."""
    is_min = ord_val < 0
    sum_axis = 1 if math.isinf(abs(ord_val)) else 0
    if d0 == ndim - 1 and d1 == ndim - 2:
        sum_axis = 1 - sum_axis
    batch = 1
    for i in range(ndim - 2):
        batch *= A.shape[i]
    M, N = A.shape[-2], A.shape[-1]
    if sum_axis == 1:
        # inf-norm: main axis = rows, wide contiguous column loads
        bn = triton.next_power_of_2(min(N, 2048))
        bm = triton.next_power_of_2(max(1, min(M, 4096 // bn)))
        grid_main = triton.cdiv(M, bm)
        L = M
    else:
        # 1-norm: main axis = columns, loop over rows
        bm = triton.next_power_of_2(min(M, 64))
        bn = triton.next_power_of_2(min(N, max(64, 4096 // bm)))
        grid_main = triton.cdiv(N, bn)
        L = N
    direct = out is not None
    buf = (
        out.reshape(-1)
        if direct
        else torch.empty(batch, dtype=out_dtype, device=A.device)
    )
    if grid_main == 1:
        _abs_norm_v2_kernel[(batch,)](
            A,
            buf,
            buf,
            M,
            N,
            bm,
            bn,
            1,
            sum_axis,
            is_min,
            True,
            num_warps=4,
        )
    else:
        partial = torch.empty(batch * L, dtype=torch.float32, device=A.device)
        _abs_norm_v2_kernel[(batch * grid_main,)](
            A,
            buf,
            partial,
            M,
            N,
            bm,
            bn,
            grid_main,
            sum_axis,
            is_min,
            False,
            num_warps=4,
        )
        _abs_norm_final_kernel[(batch,)](
            partial,
            buf,
            L,
            triton.next_power_of_2(min(L, 1024)),
            is_min,
            num_warps=4,
        )
    if direct:
        return out
    result = buf
    if keepdim:
        out_shape = list(A.shape)
        out_shape[d0] = 1
        out_shape[d1] = 1
        result = result.reshape(out_shape)
    else:
        batch_shape = [A.size(i) for i in range(ndim) if i != d0 and i != d1]
        result = result.reshape(batch_shape)
    return result


def _ord1_norm(A, ord_val, dim, keepdim, dtype, out=None):
    """1-norm (ord=1 / ord=-1).  Mirrors C++ ``ord1_norm``.

    Uses ``_abs_norm_kernel``: TILED=True (2D grid → partial buffer → host
    max/min) for large matrices, TILED=False (1D stripe → atomic scalar)
    for small matrices.  Batched via ``_batched_kernel_dispatch`` →
    ``_abs_norm_kernel`` BATCHED=True.
    """
    d0, d1 = dim
    out_dtype = dtype if dtype is not None else A.dtype
    is_min = ord_val < 0

    # --- GCU400 fast two-phase path (2D and batched, last-two dims) ---
    if _abs_norm_lastdim_ok(A, d0, d1, out_dtype, out):
        return _abs_norm_impl(A, d0, d1, ord_val, out_dtype, keepdim, out, A.ndim)

    # --- Simple 2D matrix path  ---
    if A.ndim == 2 and d0 == 0 and d1 == 1:
        M, N = A.shape

        BM, BN, grid_m, grid_n = _choose_fast_tile(M, N)

        use_fp64 = _use_fp64_acc(A.dtype)

        if grid_m * grid_n >= 128 or grid_m >= 16:
            partial = torch.zeros(N, dtype=_acc_dtype(A.dtype), device=A.device)
            _abs_norm_kernel[(grid_m * grid_n,)](
                A,
                partial,
                partial,
                M,
                N,
                BM,
                BN,
                grid_n,
                SUM_AXIS=0,
                IS_MIN=is_min,
                TILED=True,
                USE_FP64=use_fp64,
                num_warps=8,
            )
            result = (gems_min(partial) if is_min else gems_max(partial)).view(())
        else:
            init_val = float("inf") if is_min else 0.0
            # Output buffer must be fp32: tl.atomic_min / tl.atomic_max only
            # support fp32.  Precision gain comes from internal fp64 summation
            # when USE_FP64=True, not from the output dtype.  An fp32 ``out``
            # is used directly as the atomic accumulator (zero-copy).
            direct = (
                out is not None
                and out_dtype == torch.float32
                and out.is_contiguous()
                and out.numel() == 1
            )
            acc = (
                out
                if direct
                else torch.full((), init_val, dtype=torch.float32, device=A.device)
            )
            if direct:
                acc.fill_(init_val)
            _dummy = torch.empty(1, device=A.device)
            _abs_norm_kernel[(grid_n,)](
                A,
                acc,
                _dummy,
                M,
                N,
                BM,
                BN,
                1,
                SUM_AXIS=0,
                IS_MIN=is_min,
                TILED=False,
                USE_FP64=use_fp64,
                num_warps=8,
            )
            if direct:
                return out
            result = acc.to(out_dtype).view(())

        if keepdim:
            result = result.reshape(1, 1)
        if result.dtype != out_dtype:
            result = result.to(out_dtype)
        return result

    # --- Batched path ---
    return _batched_kernel_dispatch(A, dim, ord_val, out_dtype, keepdim, out=out)


def _ordinf_norm(A, ord_val, dim, keepdim, dtype, out=None):
    """Infinity-norm (ord=inf / ord=-inf).  Mirrors C++ ``ordinf_norm``.

    Uses ``_abs_norm_kernel``: same TILED/BATCHED dispatch as ``_ord1_norm``,
    but with SUM_AXIS=1 (row-wise reduction).
    """
    d0, d1 = dim
    out_dtype = dtype if dtype is not None else A.dtype
    is_min = ord_val < 0

    # --- GCU400 fast two-phase path (2D and batched, last-two dims) ---
    if _abs_norm_lastdim_ok(A, d0, d1, out_dtype, out):
        return _abs_norm_impl(A, d0, d1, ord_val, out_dtype, keepdim, out, A.ndim)

    # --- Simple 2D matrix path ---
    if A.ndim == 2 and d0 == 0 and d1 == 1:
        M, N = A.shape

        use_fp64 = _use_fp64_acc(A.dtype)

        BM, BN, grid_m, grid_n = _choose_fast_tile(M, N)

        if grid_m * grid_n >= 512 or grid_n >= 16:
            partial = torch.zeros(M, dtype=_acc_dtype(A.dtype), device=A.device)
            _abs_norm_kernel[(grid_m * grid_n,)](
                A,
                partial,
                partial,
                M,
                N,
                BM,
                BN,
                grid_n,
                SUM_AXIS=1,
                IS_MIN=is_min,
                TILED=True,
                USE_FP64=use_fp64,
                num_warps=8,
            )
            result = (gems_min(partial) if is_min else gems_max(partial)).view(())
        else:
            init_val = float("inf") if is_min else 0.0
            # fp32 ``out`` is used directly as the atomic accumulator
            # (zero-copy); otherwise keep an fp32 acc and cast below.
            direct = (
                out is not None
                and out_dtype == torch.float32
                and out.is_contiguous()
                and out.numel() == 1
            )
            acc = (
                out
                if direct
                else torch.full((), init_val, dtype=torch.float32, device=A.device)
            )
            if direct:
                acc.fill_(init_val)
            _dummy = torch.empty(1, device=A.device)
            _abs_norm_kernel[(grid_m,)](
                A,
                acc,
                _dummy,
                M,
                N,
                BM,
                BN,
                1,
                SUM_AXIS=1,
                IS_MIN=is_min,
                TILED=False,
                USE_FP64=use_fp64,
                num_warps=8,
            )
            if direct:
                return out
            result = acc.to(out_dtype).view(())

        if keepdim:
            result = result.reshape(1, 1)
        if result.dtype != out_dtype:
            result = result.to(out_dtype)
        return result

    # --- Batched path ---
    return _batched_kernel_dispatch(A, dim, ord_val, out_dtype, keepdim, out=out)


def _nuc_norm(A, dim, keepdim=False, dtype=None, out=None):
    """Nuclear norm (ord='nuc').  Mirrors C++ ``nuc_norm``.

    GCU400 fast path: last-two-dim reduction on contiguous fp16/bf16/fp32
    input runs the SVD kernels with want=3 (nuclear-norm) mode, which sums
    the singular values in a tiny fused kernel and stores the per-matrix
    norm straight into the out/result buffer -- no generic FlagGems sum
    (whose wrapper costs ~5us) and no host copy epilogue."""
    d0, d1 = dim
    out_dtype = dtype if dtype is not None else A.dtype
    if A.dtype in (torch.float16, torch.bfloat16) and dtype is None:
        out_dtype = torch.float32  # torch promotes half-precision SVD results
    ndim = A.ndim

    if (
        A.is_contiguous()
        and A.dtype in _FLOAT_CAST_DTYPES
        and out_dtype in _FLOAT_CAST_DTYPES
        and (d0, d1) in ((ndim - 2, ndim - 1), (ndim - 1, ndim - 2))
        and (out is None or out.is_contiguous())
    ):
        total = 1
        fast = True
        for s in A.shape:
            total *= s
            if total >= 2**31:
                fast = False
                break
        if fast:
            batch = 1
            for i in range(ndim - 2):
                batch *= A.shape[i]
            M, N = A.shape[-2], A.shape[-1]
            direct = out is not None
            buf = (
                out.reshape(-1)
                if direct
                else torch.empty(batch, dtype=out_dtype, device=A.device)
            )
            _svd_norm_impl(A, batch, M, N, 3, buf)
            if direct:
                return out
            result = buf
            if keepdim:
                out_shape = list(A.shape)
                out_shape[d0] = 1
                out_shape[d1] = 1
                result = result.reshape(out_shape)
            else:
                batch_shape = [A.size(i) for i in range(ndim) if i != d0 and i != d1]
                result = result.reshape(batch_shape)
            return result

    # Move target dims to last two positions.
    all_dims = list(range(ndim))
    remaining = [d for d in all_dims if d not in (d0, d1)]
    perm = remaining + [d0, d1]
    A_perm = A.permute(perm) if perm != all_dims else A
    if dtype is not None:
        A_perm = A_perm.to(dtype)

    *batch_dims, M, N = A_perm.shape
    s = _svdvals_for_norm(A_perm)  # (..., K)
    result = sum_dim(s, dim=(-1,), keepdim=False)

    if keepdim:
        d0_sorted, d1_sorted = sorted([d0, d1])
        result = result.unsqueeze(d0_sorted).unsqueeze(d1_sorted)
    return result


# ===========================================================================
# Main entry point
# ===========================================================================


def _matrix_norm_out_shape(A, d0, d1, keepdim):
    """Final shape of a matrix-norm result over dims (d0, d1).

    ``keepdim`` keeps the reduced dims as size 1; otherwise they are dropped.
    The row-major linear index of batch element ``b`` is identical in this
    shape and in the ``(batch,)`` flat buffer the reduce kernels write, so an
    ``out`` resized to this shape can be written directly by those kernels."""
    out_shape = list(A.shape)
    if keepdim:
        out_shape[d0] = 1
        out_shape[d1] = 1
    else:
        del out_shape[max(d0, d1)]
        del out_shape[min(d0, d1)]
    return tuple(out_shape)


def _check_matrix_norm_out(out, expected_dtype, device):
    """Validate a user-provided ``out`` (mirrors lu_factor's out contract)."""
    if out.dtype != expected_dtype:
        raise RuntimeError(
            "linalg_matrix_norm expected out tensor dtype "
            f"{expected_dtype} but got: {out.dtype}"
        )
    if out.device != device:
        raise RuntimeError(
            "linalg_matrix_norm: expected out tensor to be on the same device "
            f"as the result, but got {out.device} and {device}"
        )


def _linalg_matrix_norm_impl(
    A, ord="fro", dim=(-2, -1), keepdim=False, dtype=None, *, out=None
):
    """Matrix norm compute -- validation + ord dispatch.

    ``out=None`` returns a freshly-allocated result.  When ``out`` is given,
    it is validated and resized up front (before the compute) and the ord
    helpers write the per-matrix norms straight into its storage where the
    final store allows it; otherwise a single final copy is made.  Pattern
    follows ``linalg_lu_factor``.

    Dispatch order::

        1. validate inputs + dtype guard for SVD-based ords
        2. string ord: "fro" → _fro_norm
                       "nuc" →  _nuc_norm
        3. numeric ord: abs==2  → _ord2_norm
                        abs==1  → _ord1_norm
                        isinf   → _ordinf_norm
    """
    # --- validate -----------------------------------------------------------
    if A.ndim < 2:
        raise RuntimeError(
            f"linalg_matrix_norm: A must be at least 2-D, got shape {A.shape}"
        )
    dim = list(dim)
    if len(dim) != 2:
        raise RuntimeError(f"linalg_matrix_norm: dim must be a 2-tuple, got {dim}")
    dim = [d % A.ndim for d in dim]
    if dim[0] == dim[1]:
        raise RuntimeError(
            f"linalg_matrix_norm: dims must be different, got ({dim[0]}, {dim[1]})"
        )

    # --- dtype guard for SVD-based ords -------------------------------------
    # torch promotes half-precision SVD results to fp32; honor that contract
    # without materializing an fp32 copy of A (the GCU400 kernels upcast
    # inside the kernel), by promoting the requested output dtype instead.
    _svd_ord = (isinstance(ord, str) and ord == "nuc") or (
        not isinstance(ord, str) and abs(float(ord)) == 2
    )
    if _svd_ord and dtype is None and A.dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32

    # --- out validation + resize up front (before any compute) --------------
    if out is not None:
        out_dtype = dtype if dtype is not None else A.dtype
        _check_matrix_norm_out(out, out_dtype, A.device)
        out.resize_(_matrix_norm_out_shape(A, dim[0], dim[1], keepdim))

    # --- string ord: fro / nuc ----------------------------------------------
    if isinstance(ord, str):
        if ord == "fro":
            result = _fro_norm(A, dim, keepdim, dtype, out=out)
        elif ord == "nuc":
            result = _nuc_norm(A, dim=dim, keepdim=keepdim, dtype=dtype, out=out)
        else:
            raise RuntimeError(
                f"linalg_matrix_norm: Order '{ord}' not supported. "
                "Use 'fro' or 'nuc'."
            )
    else:
        # --- numeric ord ----------------------------------------------------
        ord_val = float(ord)
        if ord_val not in _SUPPORTED_NUMERIC:
            raise RuntimeError(
                f"linalg_matrix_norm: Order {ord} not supported. "
                "Use 1, -1, 2, -2, inf, -inf."
            )
        abs_ord = abs(ord_val)
        if abs_ord == 2.0:
            result = _ord2_norm(A, ord_val, dim, keepdim, dtype, out=out)
        elif abs_ord == 1.0:
            result = _ord1_norm(A, ord_val, dim, keepdim, dtype, out=out)
        elif math.isinf(abs_ord):
            result = _ordinf_norm(A, ord_val, dim, keepdim, dtype, out=out)
        else:
            raise RuntimeError(f"linalg_matrix_norm: Order {ord} not supported.")

    # --- out variant epilogue: single fallback copy when not written direct -
    if out is not None and result is not out:
        out.resize_(result.shape)
        out.copy_(result)
        return out
    return result


def linalg_matrix_norm(
    A, ord="fro", dim=(-2, -1), keepdim=False, dtype=None, *, out=None
):
    """Matrix norm -- main entry point.

    Computes the norm over ``dim`` for the given ``ord``.  When ``out`` is
    provided, writes the result into it and returns the same (aliased) tensor
    (out variant); otherwise returns a freshly-allocated result.
    """
    logger.debug("GEMS_ENFLAME LINALG_MATRIX_NORM")
    return _linalg_matrix_norm_impl(
        A, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype, out=out
    )


def linalg_matrix_norm_out(
    A, ord="fro", dim=(-2, -1), keepdim=False, dtype=None, *, out=None
):
    """Matrix norm out variant (torch ``linalg_matrix_norm.out`` overload).

    Requires a pre-allocated ``out`` and delegates to the shared impl, which
    performs the validation / resize / direct write and returns the aliased
    ``out``.  Pattern follows ``linalg_lu_factor_out``.
    """
    logger.debug("GEMS_ENFLAME LINALG_MATRIX_NORM_OUT")
    if out is None:
        raise TypeError("linalg_matrix_norm(): out must be provided for out variant")
    return _linalg_matrix_norm_impl(
        A, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype, out=out
    )


# ===========================================================================
# SVD host wrappers — _svdvals_rank2, _svdvals_for_norm
# ===========================================================================


def _svdvals_rank2(input, out_flat=None, want=0):
    """Closed-form singular values for k=2 matrices.  Mirrors C++ ``svdvals_rank2``.

    want=0: plain svdvals -- returns (..., 2) in the input dtype.
    want=1/2/3 (ord=2/-2/nuc): the kernel writes the requested norm value
    straight into ``out_flat`` (batch,) with an in-kernel dtype cast and
    this returns None -- no host slice/copy epilogue."""
    batch, m, n = _svd_shape(input)
    a = input.contiguous().reshape(batch, m, n)
    largest = max(m, n)
    block_r = triton.next_power_of_2(largest)
    if want == 0:
        s = torch.empty((batch, 2), dtype=input.dtype, device=input.device)
        out_arg = s
    else:
        s = None  # mode>0 never stores S
        out_arg = out_flat
    with torch_device_fn.device(input.device):
        if largest <= 16 and batch >= 16:
            block_b = (
                2 if largest <= 2 else (2 if m >= n else 8) if largest == 16 else 16
            )
            _rank2_svals_kernel[(triton.cdiv(batch, block_b),)](
                a,
                s,
                out_arg,
                BATCH=batch,
                M=m,
                N=n,
                TALL=m >= n,
                BLOCK_B=block_b,
                BLOCK_R=block_r,
                MODE=want,
                num_warps=1,
            )
        else:
            _rank2_svals_kernel[(batch,)](
                a,
                s,
                out_arg,
                BATCH=batch,
                M=m,
                N=n,
                TALL=m >= n,
                BLOCK_B=1,
                BLOCK_R=block_r,
                MODE=want,
                num_warps=1 if block_r <= 64 else 4,
            )
    if want == 0:
        return s.reshape(*input.shape[:-2], 2)
    return None


# ===========================================================================
# Gram → Householder symmetric tridiagonalization → Sturm bisection.
# Fast SVD path for ord=2/-2/nuc (all k ≥ 3).  All computation in fp64.
#
#   σ(A)² = eig(G),  G = A Aᵀ (wide) or AᵀA (tall), k×k PSD.
#   G is reduced to a symmetric tridiagonal T (diag D, off-diag E) by
#   k-1 Householder similarities, each applied as a rank-2 update with the
#   c = τ(vᵀw) correction term (H G H = G - v wᵀ - w vᵀ + c v vᵀ).
#   The eigenvalues λ of T are then isolated fully in parallel by Sturm
#   bisection (one program per eigenvalue target); σ = √λ.
# ===========================================================================


@libentry()
@triton.jit
def _gram_sym_kernel(
    A,
    G,
    K,
    ROWS,
    TRANSPOSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """G = A@Aᵀ (TRANSPOSE=False, A is (batch, K, ROWS)) or
    G = Aᵀ@A (TRANSPOSE=True, A is (batch, ROWS, K)).  G: (batch, K, K) fp64.
    Element-wise accumulation (fp64 tl.dot is unavailable / not tensor-core).
    Half-precision inputs are upcast to fp32 inside the kernel so the host
    does not need a separate ``A.float()`` launch."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_base = A + pid_b * K * ROWS
    g_base = G + pid_b * K * K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, ROWS, BLOCK_K):
        if TRANSPOSE:
            # A is (batch, ROWS, K): G[i,j] = sum_r A[r,i] * A[r,j]
            a = tl.load(
                a_base + (k0 + offs_k)[None, :] * K + offs_m[:, None],
                mask=((k0 + offs_k)[None, :] < ROWS) & (offs_m[:, None] < K),
                other=0.0,
            ).to(tl.float32)
            b = tl.load(
                a_base + (k0 + offs_k)[None, :] * K + offs_n[:, None],
                mask=((k0 + offs_k)[None, :] < ROWS) & (offs_n[:, None] < K),
                other=0.0,
            ).to(tl.float32)
        else:
            # A is (batch, K, ROWS): G[i,j] = sum_r A[i,r] * A[j,r]
            a = tl.load(
                a_base + offs_m[:, None] * ROWS + (k0 + offs_k)[None, :],
                mask=(offs_m[:, None] < K) & ((k0 + offs_k)[None, :] < ROWS),
                other=0.0,
            ).to(tl.float32)
            b = tl.load(
                a_base + offs_n[:, None] * ROWS + (k0 + offs_k)[None, :],
                mask=(offs_n[:, None] < K) & ((k0 + offs_k)[None, :] < ROWS),
                other=0.0,
            ).to(tl.float32)
        acc += tl.sum(a[:, None, :] * b[None, :, :], axis=2)
    tl.store(
        g_base + offs_m[:, None] * K + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < K) & (offs_n[None, :] < K),
    )


@libentry()
@triton.jit
def _sym_tridiag_kernel(G, SCR, D, E, K, BLOCK_C: tl.constexpr, BLOCK_R: tl.constexpr):
    """Householder tridiagonalization of the symmetric G (k×k), fp64.
    Single block per matrix; the whole j-loop runs in one launch.
    SCR: scratch, ≥ 2k+8 fp64 per batch (v in [0,k), w in [k,2k)).
    Output: D (diagonal), E (sub-diagonal) per batch.
    Per step: extract column j → v (Householder), w = τ·G_trail·v,
    c = τ(vᵀw), then G_trail -= v w2ᵀ + w2 vᵀ with w2 = w - (c/2)v."""
    pid_b = tl.program_id(0)
    g = G + pid_b * K * K
    sc = SCR + pid_b * (2 * K + 8)
    d = D + pid_b * K
    e = E + pid_b * (K - 1)
    idxc = tl.arange(0, BLOCK_C)
    for j in range(K - 1):
        rem = K - j - 1
        cmask = idxc < rem
        col = tl.load(g + (j + 1 + idxc) * K + j, mask=cmask, other=0.0)
        sig = tl.sqrt(tl.sum(col * col))
        x0 = tl.load(g + (j + 1) * K + j)
        beta = tl.where(x0 >= 0.0, -sig, sig)
        v = tl.where(idxc == 0, col - beta, col)
        tau = 2.0 / tl.sum(v * v)
        tl.store(sc + idxc, v, mask=cmask)
        tl.debug_barrier()
        # Pass B1: w = tau * G_trail @ v  (matvec over row chunks);
        # cs = vᵀw accumulated across chunks; c = tau*cs.
        cs = tau * 0.0
        for r0 in range(0, rem, BLOCK_R):
            rr = r0 + tl.arange(0, BLOCK_R)
            rmask = rr < rem
            rrows = j + 1 + rr
            tile = tl.load(
                g + rrows[:, None] * K + (j + 1 + idxc)[None, :],
                mask=rmask[:, None] & cmask[None, :],
                other=0.0,
            )
            vv = tl.load(sc + idxc, mask=cmask, other=0.0)
            w_r = tau * tl.sum(tile * vv[None, :], axis=1)
            vr = tl.load(sc + rr, mask=rmask, other=0.0)
            cs += tl.sum(vr * w_r)
            tl.store(sc + K + rr, w_r, mask=rmask)
        c = tau * cs
        tl.debug_barrier()
        # Pass B2: G_trail -= v w2ᵀ + w2 vᵀ  (w2 = w - (c/2)v).
        # Element [r,c]:  G'[r,c] = G[r,c] - v[r]*w2[c] - w2[r]*v[c].
        # v is indexed by column (idxc), w2 by row (rr) — BOTH terms needed.
        for r0 in range(0, rem, BLOCK_R):
            rr = r0 + tl.arange(0, BLOCK_R)
            rmask = rr < rem
            rrows = j + 1 + rr
            tile = tl.load(
                g + rrows[:, None] * K + (j + 1 + idxc)[None, :],
                mask=rmask[:, None] & cmask[None, :],
                other=0.0,
            )
            v_c = tl.load(sc + idxc, mask=cmask, other=0.0)
            v_r = tl.load(sc + rr, mask=rmask, other=0.0)
            w_c = tl.load(sc + K + idxc, mask=cmask, other=0.0)
            w_r = tl.load(sc + K + rr, mask=rmask, other=0.0)
            w2_c = w_c - 0.5 * c * v_c
            w2_r = w_r - 0.5 * c * v_r
            new = tile - v_r[:, None] * w2_c[None, :] - w2_r[:, None] * v_c[None, :]
            tl.store(
                g + rrows[:, None] * K + (j + 1 + idxc)[None, :],
                new,
                mask=rmask[:, None] & cmask[None, :],
            )
        # NOTE: B2's writes to G[j+1:, j+1:] must be visible to the next
        # step's column load — the trailing barrier handles that.  Column j /
        # row j below the sub-diagonal are never read again (each step only
        # reads its own column), so they don't need explicit zeroing, and the
        # barrier between B2 and the diagonal store is redundant.
        a_j = tl.load(g + j * K + j)
        tl.store(d + j, a_j)
        tl.store(g + (j + 1) * K + j, beta)
        tl.store(g + j * K + (j + 1), beta)
        tl.store(e + j, beta)
        tl.debug_barrier()
    a_last = tl.load(g + (K - 1) * K + (K - 1))
    tl.store(d + K - 1, a_last)


@libentry()
@triton.jit
def _sym_tridiag_reg_kernel(G, D, E, K, BLOCK_K: tl.constexpr):
    """Register-resident Householder tridiagonalization for small K.

    Same math as `_sym_tridiag_kernel`, but the whole K x K tile lives in
    registers for the entire j-loop: no GMEM round-trips, no barriers, and
    D/E are accumulated in registers and stored once at the end.  The tile
    is held TRANSPOSED (T[c, r] = G[r, c]) so every in-loop reduction runs
    along axis 0 (axis-1 reductions cost 4x more on GCU400).

    GCU400-specific cost model (measured): a cross-lane reduce to scalar is
    ~190ns and a [32,32] axis-0 reduce ~114ns, so the kernel uses only 3
    scalar reductions per step (S, x0, cs); tau is recovered from
    S - 2*x0*beta + beta*2 instead of a 4th reduction.  Run with num_warps=1
    (measured 1.4x faster than num_warps=4: fewer cross-lane extract stages).
    """
    pid_b = tl.program_id(0)
    idx = tl.arange(0, BLOCK_K)
    inb = idx < K
    g = G + pid_b * K * K
    tile = tl.load(
        g + idx[None, :] * K + idx[:, None],
        mask=inb[None, :] & inb[:, None],
        other=0.0,
    )
    # diag[r] = G[r, r] (kept separately so step j needs no scalar extract)
    diag = tl.sum(tl.where(idx[:, None] == idx[None, :], tile, 0.0), axis=0)
    dvec = tl.zeros((BLOCK_K,), dtype=tl.float32)
    evec = tl.zeros((BLOCK_K,), dtype=tl.float32)
    d = D + pid_b * K
    e = E + pid_b * (K - 1)
    for j in range(K - 1):
        rem = idx > j
        # col[r] = G[r, j] = T[j, r]: extract row j of the transposed tile.
        col = tl.sum(tl.where(idx[:, None] == j, tile, 0.0), axis=0)
        col = tl.where(rem, col, 0.0)
        S = tl.sum(col * col)
        sig = tl.sqrt(S)
        x0 = tl.sum(tl.where(idx == j + 1, col, 0.0))
        beta = tl.where(x0 >= 0.0, -sig, sig)
        v = tl.where(idx == j + 1, col - beta, col)
        # ||v||^2 = S - 2*beta*x0 + beta^2 (reuses S; no extra reduction)
        tau = 2.0 / (S - 2.0 * x0 * beta + beta * beta)
        # w[r] = tau * sum_c G[r, c] * v[c] = tau * sum_c T[c, r] * v[c].
        # v is zero outside the trailing block, so rows <= j come out as
        # sums over zero contributions and only need an explicit zeroing.
        w = tau * tl.sum(tile * v[:, None], axis=0)
        w = tl.where(rem, w, 0.0)
        cs = tl.sum(v * w)
        w2 = w - 0.5 * (tau * cs) * v
        # Rank-2 update.  v and w2 are zero outside the trailing block, so
        # leading rows/columns are untouched without any tile-level mask.
        tile = tile - w2[:, None] * v[None, :] - v[:, None] * w2[None, :]
        diag = diag - 2.0 * v * w2
        dvec = tl.where(idx == j, diag, dvec)
        evec = tl.where(idx == j, beta, evec)
    dvec = tl.where(idx == K - 1, diag, dvec)
    tl.store(d + idx, dvec, mask=inb)
    tl.store(e + idx, evec, mask=idx < K - 1)


@libentry()
@triton.jit
def _sturm_m_kernel(
    D,
    E,
    S,
    Out,
    K,
    BISECT_ITERS,
    SLOTS: tl.constexpr,
    SPP: tl.constexpr,
    BLOCK_K: tl.constexpr,
    WRITE_OUT: tl.constexpr,
    WANT_SLOT: tl.constexpr,
):
    """Sturm bisection on the symmetric tridiagonal (D, E), LAPACK-style.

    Two GCU400-specific fixes vs `_sturm_sigmas_kernel`:
      * The eigenvalue count uses the leading-principal-minor recurrence
        m_new = (d_i - mid)*m_cur - e^2*m_prev (LAPACK dstebz form) instead
        of the division recurrence q = (d_i - mid) - e^2/q: fp32 divide is
        ~150 cycles of exposed latency per step, and the minor form removes
        it from the serial chain.  A power-of-two rescale (exact, no
        rounding) keeps |m| in range.
      * D/E offsets are compile-time constants (static_range over SLOTS)
        and the bisection loop is a runtime loop, so all 2K scalar loads are
        loop-invariant and hoist out of the 32 bisection iterations instead
        of being re-fetched ~1500 times (measured 3x faster).
      * The Gershgorin bound is one vectorized reduce instead of a serial
        K-loop of scalar loads.
    Grid: batch * cdiv(SLOTS, SPP) -- the OLD kernel launched
    batch*cdiv(k,8) programs with SPP=16, a 2x overshoot whose extra
    programs read/wrote out of bounds (silent corruption risk).

    WRITE_OUT / WANT_SLOT: the program owning the requested slot also
    stores its sigma straight into Out[b] (with dtype cast), so ord=+2/-2
    needs no host-side slice+copy epilogue.  With WRITE_OUT=0 the kernel
    behaves as plain svdvals (S only).
    """
    n_groups: tl.constexpr = (SLOTS + SPP - 1) // SPP
    pid = tl.program_id(0)
    b = pid // n_groups
    group = pid % n_groups
    d = D + b * K
    e = E + b * (K - 1)
    lane = tl.arange(0, SPP)
    slot = group * SPP + lane
    smask = slot < SLOTS
    # vectorized Gershgorin upper bound
    vidx = tl.arange(0, BLOCK_K)
    vm = vidx < K
    d0 = tl.load(d + vidx, mask=vm, other=0.0)
    e_next = tl.load(e + vidx, mask=vm & (vidx < K - 1), other=0.0)
    e_prev = tl.load(e + vidx - 1, mask=vm & (vidx >= 1), other=0.0)
    hi = tl.max(tl.abs(d0) + tl.abs(e_next) + tl.abs(e_prev))
    hi = hi * (1.0 + 1e-6) + 1e-30
    hi_v = tl.full((SPP,), hi, dtype=tl.float32)
    lo_v = tl.full((SPP,), 0.0, dtype=tl.float32)
    target_f = ((SLOTS - 1) - slot).to(tl.float32)
    BIG: tl.constexpr = 5.0e17
    IBIG: tl.constexpr = 1.0 / 5.0e17
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (lo_v + hi_v)
        m_prev = tl.full((SPP,), 1.0, dtype=tl.float32)
        m_cur = tl.load(d + 0) - mid
        m_cur = tl.where(m_cur == 0.0, -1e-300, m_cur)
        cnt = tl.where(m_cur < 0.0, 1.0, 0.0)
        for i in tl.static_range(1, SLOTS):
            a = tl.load(d + i)
            b2 = tl.load(e + i - 1)
            b2 = b2 * b2
            m_new = (a - mid) * m_cur - b2 * m_prev
            m_new = tl.where(m_new == 0.0, -1e-300, m_new)
            cnt += tl.where(m_new * m_cur < 0.0, 1.0, 0.0)
            # exact power-of-two rescale keeps the minor chain in fp32 range
            sc = tl.where(m_new > BIG, IBIG, 1.0)
            sc = tl.where(m_new < -BIG, IBIG, sc)
            m_prev = m_cur * sc
            m_cur = m_new * sc
        take = cnt >= (target_f + 1.0)
        hi_v = tl.where(take, mid, hi_v)
        lo_v = tl.where(take, lo_v, mid)
    res = tl.sqrt(tl.maximum(0.5 * (lo_v + hi_v), 0.0))
    tl.store(S + b * SLOTS + slot, res, mask=smask)
    if WRITE_OUT:
        # block-shaped pointer (all lanes -> Out[b]) so the masked scalar
        # lane store compiles; only the lane owning WANT_SLOT writes.
        out_ptrs = Out + b + tl.zeros([SPP], dtype=tl.int32)
        tl.store(out_ptrs, res.to(Out.dtype.element_ty), mask=slot == WANT_SLOT)


@libentry()
@triton.jit
def _sturm_sigmas_kernel(
    D,
    E,
    S,
    Out,
    K,
    BISECT_ITERS: tl.constexpr,
    SLOTS: tl.constexpr,
    SPP: tl.constexpr,
    WRITE_OUT: tl.constexpr = False,
    WANT_SLOT: tl.constexpr = 0,
):
    """Sturm bisection on the symmetric tridiagonal (D, E): returns the
    eigenvalues of G = A Aᵀ (i.e. σ(A)²) at S[b, slot], slot 0 = largest
    (descending output).  Each program bisects SPP independent eigenvalue
    targets (SPP lanes); the independent recurrence chains give ILP that
    hides fp64 division latency.  PSD eigenvalues are all in [0, Gershgorin].

    WRITE_OUT/WANT_SLOT: the program owning WANT_SLOT also stores its sigma
    straight into Out[b] with a dtype cast (ord=+2/-2 direct out write).
    NOTE the launch grid must be batch * cdiv(SLOTS, SPP); the historical
    batch*cdiv(k, 8) grid with SPP=16 launched 2x too many programs whose
    b >= batch lanes read and wrote OUT OF BOUNDS."""
    n_groups: tl.constexpr = (SLOTS + SPP - 1) // SPP
    pid = tl.program_id(0)
    b = pid // n_groups
    group = pid % n_groups
    d = D + b * K
    e = E + b * (K - 1)
    lane = tl.arange(0, SPP)
    slot = group * SPP + lane
    smask = slot < SLOTS
    # slot 0 -> target k-1 (largest eigenvalue), slot k-1 -> target 0.
    target = (SLOTS - 1) - slot
    # Gershgorin upper bound (scalar, shared by all lanes)
    hi = tl.load(d + 0) * 0.0
    for j in range(K):
        a = tl.load(d + j)
        s = tl.abs(a)
        if j > 0:
            s += tl.abs(tl.load(e + j - 1))
        if j < K - 1:
            s += tl.abs(tl.load(e + j))
        hi = tl.maximum(hi, s)
    hi = hi * (1.0 + 1e-6) + 1e-30
    hi_v = tl.full((SPP,), hi, dtype=tl.float32)
    lo_v = tl.full((SPP,), 0.0, dtype=tl.float32)
    target_f = target.to(tl.float32)
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (lo_v + hi_v)
        q = tl.load(d + 0) - mid
        q = tl.where(q == 0.0, -1e-300, q)
        cnt = tl.where(q < 0.0, 1.0, 0.0)
        for i in range(1, K):
            a = tl.load(d + i)
            b2 = tl.load(e + i - 1)
            b2 = b2 * b2
            q = (a - mid) - b2 / q
            q = tl.where(q == 0.0, -1e-300, q)
            cnt += tl.where(q < 0.0, 1.0, 0.0)
        take = cnt >= (target_f + 1.0)
        hi_v = tl.where(take, mid, hi_v)
        lo_v = tl.where(take, lo_v, mid)
    # Fuse clamp(>=0) + sqrt into the store so the host needs no extra
    # elementwise ops (each would be a separate kernel launch).
    res = tl.sqrt(tl.maximum(0.5 * (lo_v + hi_v), 0.0))
    tl.store(S + b * SLOTS + slot, res, mask=smask)
    if WRITE_OUT:
        # block-shaped pointer (all lanes -> Out[b]) so the masked scalar
        # lane store compiles; only the lane owning WANT_SLOT writes.
        out_ptrs = Out + b + tl.zeros([SPP], dtype=tl.int32)
        tl.store(out_ptrs, res.to(Out.dtype.element_ty), mask=slot == WANT_SLOT)


def _gram_tridiag_sturm(
    a, batch, k, rows, tall, want=0, out_flat=None, bisect_iters=32
):
    """Gram -> tridiagonalization -> Sturm for k >= 3 (GCU400 fast core).

    want=0: returns the (batch, k) descending sigma matrix S.
    want=1/2: the Sturm lane owning slot 0 / slot k-1 stores sigma_max /
      sigma_min straight into out_flat (dtype cast in kernel); returns None.
    want=3: nuc -- stores sum(sigma) into out_flat via `_nuc_sum_kernel`.

    Kernel selection (measured):
      * k <= 64: `_sym_tridiag_reg_kernel` -- register-resident Householder
        (442us -> ~130us at k=32, and more accurate: no GMEM round-trips).
      * k <= 128: `_sturm_m_kernel` -- LAPACK leading-minor recurrence
        (div-free, loads hoisted out of the bisection loop, vectorized
        Gershgorin): 448us -> ~96us at k=32 with 32 bisection iterations
        (the fp32 noise floor is reached well before 48 halvings).
      * larger k (accuracy-only shapes): the original kernels, but with the
        Sturm launch grid fixed to batch*cdiv(k, SPP) -- the historical
        batch*cdiv(k, 8) grid with SPP=16 launched 2x excess programs that
        read and wrote out of bounds."""
    dev = a.device
    G = torch.empty((batch, k, k), dtype=torch.float32, device=dev)
    _gram_sym_kernel[(triton.cdiv(k, 16), triton.cdiv(k, 16), batch)](
        a,
        G,
        k,
        rows,
        TRANSPOSE=tall,
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=32,
        num_warps=4,
    )

    D = torch.empty((batch, k), dtype=torch.float32, device=dev)
    E = torch.empty((batch, k - 1), dtype=torch.float32, device=dev)
    block_k = triton.next_power_of_2(k)
    if k <= 64:
        _sym_tridiag_reg_kernel[(batch,)](
            G,
            D,
            E,
            k,
            BLOCK_K=block_k,
            num_warps=1,
        )
    else:
        scratch = torch.empty((batch, 2 * k + 8), dtype=torch.float32, device=dev)
        _sym_tridiag_kernel[(batch,)](
            G,
            scratch,
            D,
            E,
            k,
            BLOCK_C=block_k,
            BLOCK_R=16,
            num_warps=8,
            num_stages=1,
        )

    S = torch.empty((batch, k), dtype=torch.float32, device=dev)
    out_arg = out_flat if out_flat is not None else S
    write_out = want in (1, 2)
    want_slot = 0 if want == 1 else k - 1
    # SPP=32 packs all 32 slots of one matrix into a single program; with
    # many matrices that halves the program count and avoids a second
    # occupancy wave (measured 202us -> 106us at batch=32, k=32).
    spp = 16
    if batch * triton.cdiv(k, 16) > 48 and k >= 32:
        spp = min(32, triton.next_power_of_2(k))
    grid = (batch * triton.cdiv(k, spp),)
    if k <= 128:
        _sturm_m_kernel[grid](
            D,
            E,
            S,
            out_arg,
            k,
            bisect_iters,
            SLOTS=k,
            SPP=spp,
            BLOCK_K=block_k,
            WRITE_OUT=write_out,
            WANT_SLOT=want_slot,
            num_warps=2,
        )
    else:
        _sturm_sigmas_kernel[grid](
            D,
            E,
            S,
            out_arg,
            k,
            BISECT_ITERS=bisect_iters,
            SLOTS=k,
            SPP=spp,
            WRITE_OUT=write_out,
            WANT_SLOT=want_slot,
            num_warps=2,
        )
    if want == 3:
        _nuc_sum_kernel[(batch,)](
            S,
            out_arg,
            k,
            triton.next_power_of_2(min(k, 1024)),
            num_warps=4,
        )
    return S if want == 0 else None


_BISECT_ITERS = 32


def _svdvals_gram_tridiag(A, bisect_iters=_BISECT_ITERS):
    """SVD via Gram → symmetric tridiagonalization → Sturm bisection.
    All computation in fp32 (GCU400 has no fp64 arithmetic).  Returns
    singular values, descending, shape (..., K)."""
    *batch_dims, M, N = A.shape
    batch = 1
    for d in batch_dims:
        batch *= d
    k = min(M, N)
    rows = max(M, N)
    tall = M >= N
    if A.dtype == torch.float64:
        A = A.float()
    A = A.contiguous()
    a = A.reshape(batch, M, N)
    S = _gram_tridiag_sturm(a, batch, k, rows, tall, 0, None, bisect_iters)
    return S.reshape(*batch_dims, k)


def _svd_norm_impl(A, batch, M, N, want, buf):
    """ord=2/-2/nuc (want=1/2/3) for contiguous fp16/bf16/fp32 (batch, M, N)
    matrices, writing per-matrix norm values straight into ``buf`` (batch,).

    Transpose-invariance of singular values means swapped last-two reduce
    dims need no special handling here.  All kernels accumulate in fp32 and
    cast at the final store, so half inputs match torch's promote-to-fp32
    SVD numerics."""
    k = min(M, N)
    rows = max(M, N)
    a = A.reshape(batch, M, N)  # view (A is contiguous)
    if k == 1:
        # single singular value = Frobenius norm of the (M, N) matrix
        flat = a.reshape(batch, M * N)
        blk_n = triton.next_power_of_2(min(M * N, 512))
        _fro_kernel[(batch,)](
            flat,
            buf,
            0,
            M * N,
            1,
            blk_n,
            1,
            TILE_2D=False,
            USE_FP64=False,
            num_warps=8,
        )
        return
    if k == 2 and rows <= _RANK2_BLOCK_R_MAX:
        _svdvals_rank2(a, out_flat=buf, want=want)
        return
    if 2 < k <= 512 and rows <= 2048:
        _gram_tridiag_sturm(a, batch, k, rows, M >= N, want, buf)
        return
    raise NotImplementedError(
        f"FlagGems svdvals: unsupported matrix shape. "
        f"Got batch={batch}, m={M}, n={N} (k={k}, rows={rows}). "
        f"Supported: k=1 (L2 norm), k==2 with rows<={_RANK2_BLOCK_R_MAX}, "
        f"or 2<k<=512 with rows<=2048."
    )


def _svdvals_for_norm(A):
    """SVD dispatch for ord=2/-2/nuc.  Returns (..., K).

    Precision strategy (matches PyTorch CUDA gesvdj / gesvd):
      fp64 → fp64,  fp32 → fp32,  fp16/bf16 → upcast to fp32 (in kernel).
    """
    in_dtype = A.dtype
    if in_dtype == torch.float64:
        # GCU400 has no fp64 kernel arithmetic: linalg_matrix_norm with
        # dtype=float64 upcasts A to fp64 on the host (a plain dtype cast,
        # which works), but the SVD kernels below must still compute in
        # fp32.  Downcast here; the host-side .to(in_dtype) at each return
        # restores the requested fp64 result dtype.
        A = A.float()
    A = A.contiguous()
    *batch_dims, M, N = A.shape
    batch = 1
    for d in batch_dims:
        batch *= d
    k = min(M, N)
    rows = max(M, N)

    # --- rank-1: single singular value = Frobenius norm via _fro_kernel ---
    if k == 1:
        flat = A.reshape(batch, M * N)
        s = torch.empty(batch, 1, dtype=torch.float32, device=A.device)
        blk_n = triton.next_power_of_2(min(M * N, 512))
        # A is fp32/fp64 here (fp16/bf16 upcast above), so this always uses
        # fp64 accumulation on the generic path.
        _fro_kernel[(batch,)](
            flat,
            s,
            0,
            M * N,
            1,
            blk_n,
            1,
            TILE_2D=False,
            USE_FP64=_use_fp64_acc(A.dtype),
            num_warps=8,
        )
        return s.reshape(*batch_dims, 1).to(in_dtype)

    # --- rank-2 closed form ------------------------------------------------
    if k == 2 and rows <= _RANK2_BLOCK_R_MAX:
        return _svdvals_rank2(A).to(in_dtype)

    # --- gesvd: all k≥3 through _svdvals_gram_tridiag (Gram → Householder
    # tridiagonalization → Sturm bisection; ~30x faster than QR+Jacobi) --
    if 2 < k <= 512 and rows <= 2048:
        return _svdvals_gram_tridiag(A).to(in_dtype)
    # --- unsupported -------------------------------------------------------
    raise NotImplementedError(
        f"FlagGems svdvals: unsupported matrix shape. "
        f"Got batch={batch}, m={M}, n={N} (k={k}, rows={rows}). "
        f"Supported: k=1 (L2 norm), k==2 with rows<={_RANK2_BLOCK_R_MAX}, "
        f"or 2<k<=512 with rows<=2048."
    )
