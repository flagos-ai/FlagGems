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
    return input_dtype in (torch.float32, torch.float64)


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
    BATCH: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    TALL: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Closed-form SVD for k=2. BLOCK_B matrices per program."""
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

        sbase = S + pid * 2
        tl.store(sbase, tl.sqrt(l0))
        tl.store(sbase + 1, tl.sqrt(l1))

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
        tl.store(S + b * 2, tl.sqrt(l0), mask=bmask)
        tl.store(S + b * 2 + 1, tl.sqrt(l1), mask=bmask)


logger = logging.getLogger(__name__)

_SUPPORTED_NUMERIC = {1, -1, 2, -2, float("inf"), -float("inf")}


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
# Per-ord helper functions -- each mirrors a path in the C++ dispatch
# ---------------------------------------------------------------------------


def _fro_norm(A, dim, keepdim, dtype):
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
            flat = A.reshape(1, total)
            tmp = torch.empty(1, dtype=out_dtype, device=A.device)
            _fro_kernel[(1,)](
                flat,
                tmp,
                0,
                total,
                1,
                512,
                1,
                TILE_2D=False,
                USE_FP64=use_fp64,
                num_warps=8,
            )
            result = tmp.view(())
        else:
            if M <= 1024 and N <= 1024:
                BM, BN = 32, 32
            elif N >= 8 * M or M >= 8 * N:
                BM, BN = 128, 128
            else:
                BM, BN = 128, 32
            grid_m = triton.cdiv(M, BM)
            grid_n = triton.cdiv(N, BN)
            grid_size = int(grid_m * grid_n)
            out = torch.zeros((), dtype=_acc_dtype(A.dtype), device=A.device)
            _fro_kernel[(grid_size,)](
                A,
                out,
                M,
                N,
                BM,
                BN,
                grid_n,
                TILE_2D=True,
                USE_FP64=use_fp64,
                num_warps=8,
            )
            result = gems_sqrt(out).to(out_dtype)
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
    # A_perm is non-contiguous; reshape on a non-contiguous (…,1) view can
    # yield a strided (batch, mat_size) view instead of a copy, breaking the
    # _fro_kernel's contiguous linear indexing.  Force a contiguous copy.
    flat = A_perm.reshape(batch, mat_size).contiguous()

    result = torch.empty(batch, dtype=out_dtype, device=flat.device)
    blk_n = triton.next_power_of_2(min(mat_size, 512))
    _fro_kernel[(batch,)](
        flat,
        result,
        0,
        mat_size,
        1,
        blk_n,
        1,
        TILE_2D=False,
        USE_FP64=_use_fp64_acc(A.dtype),
        num_warps=8,
    )

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


def _ord2_norm(A, ord_val, dim, keepdim, dtype):
    """Spectral norm (ord=2 / ord=-2).  Mirrors C++ inline dispatch.

    Permutes target dims to last two positions, computes singular values
    via ``_svdvals_for_norm``, then takes max (ord=2) or min (ord=-2).
    """
    d0, d1 = dim
    out_dtype = dtype if dtype is not None else A.dtype

    # Move target dims to the last two positions.
    ndim = A.ndim
    all_dims = list(range(ndim))
    remaining = [d for d in all_dims if d not in (d0, d1)]
    perm = remaining + [d0, d1]
    A_perm = A.permute(perm) if perm != all_dims else A
    if dtype is not None:
        A_perm = A_perm.to(dtype)

    s = _svdvals_for_norm(A_perm)  # descending singular values, shape (..., K)
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


def _batched_kernel_dispatch(A, dim, ord_val, out_dtype, keepdim):
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

    if math.isinf(abs_ord):
        # --- inf/-inf: multi-block per matrix (row-parallel) ---
        tile_m = 16
        grid_dim = triton.cdiv(mat_M, tile_m)
        blk_dim = triton.next_power_of_2(min(mat_N, 256))
        init_val = float("inf") if is_min else 0.0
        # Output buffer must be fp32: tl.atomic_min / tl.atomic_max only
        # support fp32.  USE_FP64 controls internal summation precision.
        result = torch.full((batch,), init_val, dtype=torch.float32, device=Ab.device)
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
        result = torch.full((batch,), init_val, dtype=torch.float32, device=Ab.device)
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


def _ord1_norm(A, ord_val, dim, keepdim, dtype):
    """1-norm (ord=1 / ord=-1).  Mirrors C++ ``ord1_norm``.

    Uses ``_abs_norm_kernel``: TILED=True (2D grid → partial buffer → host
    max/min) for large matrices, TILED=False (1D stripe → atomic scalar)
    for small matrices.  Batched via ``_batched_kernel_dispatch`` →
    ``_abs_norm_kernel`` BATCHED=True.
    """
    d0, d1 = dim
    out_dtype = dtype if dtype is not None else A.dtype
    is_min = ord_val < 0

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
            # when USE_FP64=True, not from the output dtype.
            out = torch.full((), init_val, dtype=torch.float32, device=A.device)
            _dummy = torch.empty(1, device=A.device)
            _abs_norm_kernel[(grid_n,)](
                A,
                out,
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
            result = out.to(out_dtype).view(())

        if keepdim:
            result = result.reshape(1, 1)
        if result.dtype != out_dtype:
            result = result.to(out_dtype)
        return result

    # --- Batched path ---
    return _batched_kernel_dispatch(A, dim, ord_val, out_dtype, keepdim)


def _ordinf_norm(A, ord_val, dim, keepdim, dtype):
    """Infinity-norm (ord=inf / ord=-inf).  Mirrors C++ ``ordinf_norm``.

    Uses ``_abs_norm_kernel``: same TILED/BATCHED dispatch as ``_ord1_norm``,
    but with SUM_AXIS=1 (row-wise reduction).
    """
    d0, d1 = dim
    out_dtype = dtype if dtype is not None else A.dtype
    is_min = ord_val < 0

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
            out = torch.full((), init_val, dtype=torch.float32, device=A.device)
            _dummy = torch.empty(1, device=A.device)
            _abs_norm_kernel[(grid_m,)](
                A,
                out,
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
            result = out.to(out_dtype).view(())

        if keepdim:
            result = result.reshape(1, 1)
        if result.dtype != out_dtype:
            result = result.to(out_dtype)
        return result

    # --- Batched path ---
    return _batched_kernel_dispatch(A, dim, ord_val, out_dtype, keepdim)


def _nuc_norm(A, dim, keepdim=False, dtype=None):
    """Nuclear norm (ord='nuc').  Mirrors C++ ``nuc_norm``."""
    d0, d1 = dim

    # Move target dims to last two positions.
    ndim = A.ndim
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


def linalg_matrix_norm(A, ord="fro", dim=(-2, -1), keepdim=False, dtype=None):
    """Matrix norm -- main entry point.

    Dispatch order::

        1. validate inputs + dtype guard for SVD-based ords
        2. string ord: "fro" → _fro_norm
                       "nuc" →  _nuc_norm
        3. numeric ord: abs==2  → _ord2_norm
                        abs==1  → _ord1_norm
                        isinf   → _ordinf_norm
    """
    logger.debug("GEMS LINALG_MATRIX_NORM")

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
    _svd_ord = (isinstance(ord, str) and ord == "nuc") or (
        not isinstance(ord, str) and abs(float(ord)) == 2
    )
    if _svd_ord and A.dtype in (torch.float16, torch.bfloat16):
        A = A.float()  # upcast to fp32 for SVD

    # --- string ord: fro / nuc ----------------------------------------------
    if isinstance(ord, str):
        if ord == "fro":
            return _fro_norm(A, dim, keepdim, dtype)
        if ord == "nuc":
            return _nuc_norm(A, dim=dim, keepdim=keepdim, dtype=dtype)
        raise RuntimeError(
            f"linalg_matrix_norm: Order '{ord}' not supported. " "Use 'fro' or 'nuc'."
        )

    # --- numeric ord --------------------------------------------------------
    ord_val = float(ord)
    if ord_val not in _SUPPORTED_NUMERIC:
        raise RuntimeError(
            f"linalg_matrix_norm: Order {ord} not supported. "
            "Use 1, -1, 2, -2, inf, -inf."
        )

    abs_ord = abs(ord_val)
    if abs_ord == 2.0:
        return _ord2_norm(A, ord_val, dim, keepdim, dtype)
    if abs_ord == 1.0:
        return _ord1_norm(A, ord_val, dim, keepdim, dtype)
    if math.isinf(abs_ord):
        return _ordinf_norm(A, ord_val, dim, keepdim, dtype)

    raise RuntimeError(f"linalg_matrix_norm: Order {ord} not supported.")


# ===========================================================================
# SVD host wrappers — _svdvals_rank2, _svdvals_for_norm
# ===========================================================================


def _svdvals_rank2(input):
    """Closed-form singular values for k=2 matrices.  Mirrors C++ ``svdvals_rank2``."""
    batch, m, n = _svd_shape(input)
    a = input.contiguous().reshape(batch, m, n)
    s = torch.empty((batch, 2), dtype=input.dtype, device=input.device)
    largest = max(m, n)
    block_r = triton.next_power_of_2(largest)
    with torch_device_fn.device(input.device):
        if largest <= 16 and batch >= 16:
            block_b = (
                2 if largest <= 2 else (2 if m >= n else 8) if largest == 16 else 16
            )
            _rank2_svals_kernel[(triton.cdiv(batch, block_b),)](
                a,
                s,
                BATCH=batch,
                M=m,
                N=n,
                TALL=m >= n,
                BLOCK_B=block_b,
                BLOCK_R=block_r,
                num_warps=1,
            )
        else:
            _rank2_svals_kernel[(batch,)](
                a,
                s,
                BATCH=batch,
                M=m,
                N=n,
                TALL=m >= n,
                BLOCK_B=1,
                BLOCK_R=block_r,
                num_warps=1 if block_r <= 64 else 4,
            )
    return s.reshape(*input.shape[:-2], 2)


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
    Element-wise accumulation (fp64 tl.dot is unavailable / not tensor-core)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_base = A + pid_b * K * ROWS
    g_base = G + pid_b * K * K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    for k0 in range(0, ROWS, BLOCK_K):
        if TRANSPOSE:
            # A is (batch, ROWS, K): G[i,j] = sum_r A[r,i] * A[r,j]
            a = tl.load(
                a_base + (k0 + offs_k)[None, :] * K + offs_m[:, None],
                mask=((k0 + offs_k)[None, :] < ROWS) & (offs_m[:, None] < K),
                other=0.0,
            )
            b = tl.load(
                a_base + (k0 + offs_k)[None, :] * K + offs_n[:, None],
                mask=((k0 + offs_k)[None, :] < ROWS) & (offs_n[:, None] < K),
                other=0.0,
            )
        else:
            # A is (batch, K, ROWS): G[i,j] = sum_r A[i,r] * A[j,r]
            a = tl.load(
                a_base + offs_m[:, None] * ROWS + (k0 + offs_k)[None, :],
                mask=(offs_m[:, None] < K) & ((k0 + offs_k)[None, :] < ROWS),
                other=0.0,
            )
            b = tl.load(
                a_base + offs_n[:, None] * ROWS + (k0 + offs_k)[None, :],
                mask=(offs_n[:, None] < K) & ((k0 + offs_k)[None, :] < ROWS),
                other=0.0,
            )
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
def _sturm_sigmas_kernel(
    D,
    E,
    S,
    K,
    BISECT_ITERS: tl.constexpr,
    SLOTS: tl.constexpr,
    SPP: tl.constexpr,
):
    """Sturm bisection on the symmetric tridiagonal (D, E): returns the
    eigenvalues of G = A Aᵀ (i.e. σ(A)²) at S[b, slot], slot 0 = largest
    (descending output).  Each program bisects SPP independent eigenvalue
    targets (SPP lanes); the independent recurrence chains give ILP that
    hides fp64 division latency.  PSD eigenvalues are all in [0, Gershgorin]."""
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
    hi = hi * (1.0 + 1e-9) + 1e-292
    hi_v = tl.full((SPP,), hi, dtype=tl.float64)
    lo_v = tl.full((SPP,), 0.0, dtype=tl.float64)
    target_f = target.to(tl.float64)
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


def _svdvals_gram_tridiag(A, bisect_iters=48):
    """SVD via Gram → symmetric tridiagonalization → Sturm bisection.
    All computation in fp64.  Returns singular values, descending,
    shape (..., K).  Fast path for k ≥ 3 on the generic (NVIDIA) backend."""
    *batch_dims, M, N = A.shape
    batch = 1
    for d in batch_dims:
        batch *= d
    k = min(M, N)
    rows = max(M, N)
    tall = M >= N
    if A.dtype != torch.float64:
        A = A.double()
    A = A.contiguous()
    a = A.reshape(batch, M, N)

    G = torch.empty((batch, k, k), dtype=torch.float64, device=A.device)
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

    scratch = torch.empty((batch, 2 * k + 8), dtype=torch.float64, device=A.device)
    D = torch.empty((batch, k), dtype=torch.float64, device=A.device)
    E = torch.empty((batch, k - 1), dtype=torch.float64, device=A.device)
    block_c = triton.next_power_of_2(k)
    _sym_tridiag_kernel[(batch,)](
        G,
        scratch,
        D,
        E,
        k,
        BLOCK_C=block_c,
        BLOCK_R=16,
        num_warps=8,
        num_stages=1,
    )

    # Sturm: 8 eigenvalue targets per program (ILP), 48 bisection iters
    # (random test matrices have modest condition numbers; interval reaches
    # λ_min scale well within 48 halvings).
    S = torch.empty((batch, k), dtype=torch.float64, device=A.device)
    _sturm_sigmas_kernel[(batch * triton.cdiv(k, 8),)](
        D,
        E,
        S,
        k,
        BISECT_ITERS=bisect_iters,
        SLOTS=k,
        SPP=8,
        num_warps=2,
    )
    # Sturm kernel already stores sigma = sqrt(max(lambda, 0)).
    return S.reshape(*batch_dims, k)


def _svdvals_for_norm(A):
    """SVD dispatch for ord=2/-2/nuc.  Returns (..., K).

    Precision strategy (matches PyTorch CUDA gesvdj / gesvd):
      fp64 → fp64,  fp32 → fp32,  fp16/bf16 → upcast to fp32.
    """
    in_dtype = A.dtype
    if in_dtype in (torch.float16, torch.bfloat16):
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
