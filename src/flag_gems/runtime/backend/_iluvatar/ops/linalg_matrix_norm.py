"""linalg_matrix_norm -- ILUVATAR (CoreX) override.

No fp64 compute path: all SVD accumulation runs in fp32, and fp64
inputs are rejected.  The 7 shared Triton kernels are imported from
flag_gems.ops.linalg_matrix_norm.
"""

import logging
import math

import torch
import triton

import flag_gems

# FlagGems CUDA-native computational operators (replace torch.abs/torch.all/torch.max/
# torch.min/torch.norm/torch.sqrt/torch.sum/torch.topk/torch.amax with FlagGems equivalents).
from flag_gems.ops.abs import abs as gems_abs
from flag_gems.ops.all import all as gems_all
from flag_gems.ops.amax import amax
from flag_gems.ops.linalg_matrix_norm import (
    _abs_norm_kernel,
    _bidiag_kernel,
    _fro_kernel,
    _fused_dbdsqr_kernel,
    _householder_qr_r_kernel,
    _parallel_jacobi_step_kernel,
    _rank2_svals_kernel,
)
from flag_gems.ops.max import max as gems_max
from flag_gems.ops.max import max_dim
from flag_gems.ops.min import min as gems_min
from flag_gems.ops.sqrt import sqrt as gems_sqrt
from flag_gems.ops.sum import sum_dim
from flag_gems.ops.topk import topk as gems_topk
from flag_gems.runtime import torch_device_fn

# ===========================================================================
# Kernel: _rank2_svals_kernel -- closed-form SVD for k=2.
# Used by _svdvals_for_norm and C++ SVD dispatch.  No iteration needed.
# BLOCK_B=1 → one matrix per program (regular).
# BLOCK_B>1 → BLOCK_B matrices per program (vectorized, for tiny rows).
# ===========================================================================

_RANK2_BLOCK_R_MAX = 2048


def _use_fp64_acc(input_dtype):
    """Whether to use fp64 for internal accumulation in reduction kernels.

    ILUVATAR (CoreX) has no fp64 compute path: fp64 inputs are rejected at
    the entry point, so every kernel reachable here accumulates in fp32."""
    return False


def _acc_dtype(input_dtype):
    """Accumulator dtype for host-side reduction buffers.

    Always fp32 -- mirrors ``_use_fp64_acc`` (fp64 inputs never reach the
    kernels on CoreX)."""
    return torch.float32


def _select_dbdsqr_params(k):
    """Autotune DBDSQR parameters by k: larger k needs more iterations.

    Returns (MAX_ITERS, num_warps, BLOCK_SWEEPS).  BLOCK_SWEEPS controls the
    inner Golub-Kahan QR sweep count inside ``_fused_dbdsqr_kernel``.

    ILUVATAR uses the generic parameters; k=3 is the only k that normally
    reaches DBDSQR on iluvatar (k ≥ 4 goes through the fp32 Jacobi path).
    """
    if k <= 32:
        return 30, 1, 50
    elif k <= 64:
        return 50, 1, 50
    elif k <= 128:
        return 100, 4, 50
    else:
        return 200, 4, 50


def _svd_shape(A):
    """Return (batch, M, N) for an SVD-shaped tensor."""
    if A.ndim < 2:
        return 0, 0, 0
    batch = 1
    for d in A.shape[:-2]:
        batch *= d
    return batch, A.shape[-2], A.shape[-1]


# ===========================================================================
# Householder QR kernel -- pure Triton left-Householder QR (R-only).
# Replaces torch.linalg.qr for A of shape (M, N) with M >= N, K = min(M,N).
# Outputs upper-triangular R (K×K).  Does NOT compute Q.
# Grid=(batch,).  Each program handles one matrix independently.
# ===========================================================================


# ===========================================================================
# gesvd kernels -- fp64 Householder bidiagonalization + fused DBDSQR.
# cuSOLVER-aligned: bidiag in fp64 matches GEBRD, DBDSQR tol 2.2e-14 converges.
# ===========================================================================


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
#
# Replaces: _fused_abs_tiled_kernel + _fused_abs_single_kernel
#           + _reduce_kernel + _batched_abs_multi_kernel
# ===========================================================================


# --- 2D tiled Frobenius -- single-launch with atomic_add to scalar ----------
# ===========================================================================
# Host dispatch -- mirrors the C++ dispatch in lib/linalg_matrix_norm.cpp.
#
# C++ and Python use the same function names (Python prefixed with _):
#
#   C++                       Python
#   ────────────────────────  ───────────────────────
#   fro_norm                  _fro_norm
#   nuc_norm                  _nuc_norm
#   ord1_norm                 _ord1_norm
#   ordinf_norm               _ordinf_norm
#   svdvals_hybrid            _svdvals_hybrid
#   svdvals_rank2             _svdvals_rank2
#   (inline dispatch)         _ord2_norm
#   (inline dispatch)         _svdvals_for_norm
# ===========================================================================


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _amin(x, dim, keepdim=False):
    """Drop-in for ``torch.amin`` -- works for all float dtypes.

    FlagGems ``amax`` requires *dim* as a list/tuple, not a plain int.
    """
    if isinstance(dim, int):
        dim = [dim]
    return -amax(-x, dim=dim, keepdim=keepdim)


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

    # Batched case -- same kernel as C++ wrapper (mirrors linalg_matrix_norm_str).
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

    s = _svdvals_for_norm(A_perm)
    fn = amax if ord_val > 0 else _amin
    result = fn(s, dim=[-1], keepdim=False)

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
    """Mirrors the C++ batched path in ``linalg_matrix_norm`` exactly.

    Permutes so the two matrix dims are last, reshapes to (batch, M, N),
    then dispatches to the same Triton kernels used by the C++ wrapper.
    """
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
        blk_dim = triton.next_power_of_2(min(mat_N, 256))
        # CoreX Triton codegen bug: a 512-element tile (tile_m × blk_dim == 512,
        # e.g. 16×32) with num_warps=8 (2 elements/thread) produces wrong
        # tl.sum results.  Bump tile_m to 32 so the tile becomes 32×32=1024
        # (4 elements/thread), which reduces correctly.
        if tile_m * blk_dim == 512:
            tile_m = 32
        grid_dim = triton.cdiv(mat_M, tile_m)
        init_val = float("inf") if is_min else 0.0
        # Output buffer must be fp32: tl.atomic_min / tl.atomic_max only
        # support fp32.
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

        if grid_m * grid_n >= 128:
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
            # support fp32.
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

    # --- Batched path -- same kernels as the C++ wrapper ---
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

        # Few rows/cols: Triton parallelism limited → direct computation.
        # ILUVATAR (CoreX): when both dims are < 32 the non-TILED path
        # creates pathological masked loads; fall back to direct PyTorch.
        if (
            M <= 2
            or N <= 2
            or (flag_gems.vendor_name == "iluvatar" and M < 32 and N < 32)
        ):
            row_sums = sum_dim(gems_abs(A), dim=(1,))
            result = gems_min(row_sums) if is_min else gems_max(row_sums)
            if result.dtype != out_dtype:
                result = result.to(out_dtype)
            if keepdim:
                result = result.reshape(1, 1)
            return result

        use_fp64 = _use_fp64_acc(A.dtype)

        BM, BN, grid_m, grid_n = _choose_fast_tile(M, N)

        if grid_m * grid_n >= 512:
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
            # ILUVATAR (CoreX): clamp tile dims ≥ 32 for correctness.
            # CoreX Triton reductions produce wrong results when a tile
            # dimension is < 32 elements.
            if flag_gems.vendor_name == "iluvatar":
                BM = max(BM, 32)
                BN = max(BN, 32)
                grid_m = triton.cdiv(M, BM)
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

    # --- Batched path -- same kernels as the C++ wrapper ---
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

    Mirrors the C++ dispatch in ``lib/linalg_matrix_norm.cpp``.  The C++
    wrapper is preferred when available (lower Python overhead); the Python
    helpers provide an equivalent fallback for backends without the C++
    extension.

    Dispatch order::

        1. validate inputs + dtype guard for SVD-based ords
        2. string ord: "fro" → C++ (if available) else _fro_norm
                       "nuc" → C++ (if available) else _nuc_norm
        3. numeric ord: C++ linalg_matrix_norm (if available and ord handled)
                        else per-ord Python helper:
                          abs==2  → _ord2_norm
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

    # --- fp64 guard: CoreX has no fp64 compute path -------------------------
    # Reject fp64 inputs with an explicit error instead of silently
    # downcasting.
    if A.dtype == torch.float64:
        raise RuntimeError(
            f"linalg_matrix_norm: fp64 input is not supported on the "
            f"{flag_gems.vendor_name} backend"
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
# ===========================================================================
# Parallel Jacobi step kernel -- Brent-Luk ordering, grid=(batch, K/2).
# ===========================================================================


# ===========================================================================
# SVD host wrappers — _svdvals_rank2, _svdvals_hybrid, _svdvals_for_norm
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


def _qr_rf(a, batch, m, n, k, device, tall):
    """QR factorisation producing the triangular R factor (k×k), fp32.

    Builds fresh scratch from ``a`` (the QR kernel overwrites its input with
    Householder vectors), then runs a single-block Householder QR.
    """
    if tall:
        a_qr = a.float().clone()
        M_qr, N_qr = m, n
    else:
        a_qr = a.transpose(-2, -1).contiguous()
        a_qr = a_qr.float().clone()
        M_qr, N_qr = n, m
    block_m = triton.next_power_of_2(min(M_qr, 256))

    Rf = torch.zeros((batch, k, k), dtype=torch.float32, device=device)

    # Fence before the QR kernel reads a_qr: its backing copies
    # (transpose/contiguous/clone) are torch ops, and on sm8.0-class devices
    # a Triton launch does not reliably observe prior writes without a full
    # device sync (see the QR->Jacobi fence note below).  Without this, the
    # QR factorisation can read a partially-copied matrix under load -- a
    # rare but real corruption (observed once as a ~1.5e-3 sigma_max error
    # on the k=3 keepdim test during a full-suite run).
    torch_device_fn.synchronize()
    with torch_device_fn.device(device):
        _householder_qr_r_kernel[(batch,)](
            a_qr,
            Rf,
            M_qr,
            N_qr,
            k,
            stride_b=a_qr.stride(0),
            stride_m=a_qr.stride(-2),
            stride_n=a_qr.stride(-1),
            BLOCK_M=block_m,
            BLOCK_N=32,
            USE_FP64=False,
            num_warps=4,
        )
    return Rf


def _svdvals_hybrid(input):
    """Hybrid SVD on ILUVATAR: fp32 QR + Jacobi (15/20 sweeps) + DBDSQR.

    Dispatch (fp32 throughout -- CoreX has no fp64 compute path):
      k ≥ 4:  fp32 QR + fp32 Jacobi (15/20 sweeps).
              Falls back to fp32 DBDSQR if Jacobi doesn't converge.
      k = 3:  fp32 DBDSQR directly (bidiagonalisation + Golub-Kahan QR).
    """

    batch, m, n = _svd_shape(input)
    k = min(m, n)
    tall = m >= n
    a = input.contiguous().reshape(batch, m, n)
    device = input.device

    # ---- 1. QR: A → triangular R (k×k), fp32 --------------------------------
    Rf = _qr_rf(a, batch, m, n, k, device, tall)

    # Fence before Jacobi reads Rf: on SM 8.0 (A100) Triton kernel launches
    # do not reliably act as memory fences under memory pressure, so a
    # subsequent read of Rf (via Rf.transpose().contiguous() below) can
    # observe stale data.  A full device sync guarantees the QR output is
    # visible before the Jacobi sweep reads it.  Mirrors the per-sweep
    # synchronize()+clone() fence already used inside the Jacobi loop.
    torch_device_fn.synchronize()

    # ---- 2. Jacobi SVD on R ------------------------------------------------
    # ILUVATAR: Jacobi for all k ≥ 4, 15/20 sweeps.  The fp32 precision
    # floor is higher, so additional sweeps don't help.  Non-converged →
    # fall through to DBDSQR below.
    use_jacobi = k >= 4

    if use_jacobi:
        _JACOBI_SWEEPS = 15 if k <= 48 else 20

        block_r = triton.next_power_of_2(k)
        a_work = Rf.transpose(1, 2).contiguous()  # column-major for Jacobi

        with torch_device_fn.device(device):
            for _ in range(_JACOBI_SWEEPS):
                for step in range(k - 1):
                    _parallel_jacobi_step_kernel[(batch, k // 2)](
                        a_work,
                        k,
                        k,
                        step,
                        BLOCK_R=block_r,
                        num_warps=1 if block_r <= 64 else 4,
                    )
                torch_device_fn.synchronize()
                a_work = a_work.clone()

        # Convergence check: Gram off-diagonal ≤ tol × max diagonal
        gram = a_work @ a_work.transpose(1, 2)
        k_idx = torch.arange(k, device=device)
        diag = gram[:, k_idx, k_idx]
        off_mask = ~torch.eye(k, dtype=torch.bool, device=device)
        max_off = max_dim(gems_abs(gram[:, off_mask]), dim=-1).values
        max_diag = max_dim(gems_abs(diag), dim=-1).values
        rel_tol = 5e-4  # Rf is always fp32 on iluvatar
        jacobi_ok = bool(gems_all(max_off <= rel_tol * max_diag).item())

        if jacobi_ok:
            col_norms = gems_sqrt((a_work * a_work).sum(dim=-1).clamp(min=0.0))
            s_sorted = gems_topk(col_norms, k, dim=-1, largest=True)[0]
            return s_sorted.reshape(*input.shape[:-2], k).to(input.dtype)
        # Non-converged → fall through to DBDSQR

    # ---- 3. DBDSQR: bidiagonalisation + Golub-Kahan QR ----------------------
    # Reached for k=3, or when the fp32 Jacobi above doesn't converge.
    # Reuse Rf from step 1: both are fp32 (CoreX has no fp64 compute path).
    work_dtype = torch.float32
    block_k = triton.next_power_of_2(k)
    d_out = torch.zeros((batch, k), dtype=work_dtype, device=device)
    e_out = torch.zeros((batch, k - 1), dtype=work_dtype, device=device)

    with torch_device_fn.device(device):
        R_dbdsqr = Rf
        # Fence before the bidiagonalisation kernel reads R_dbdsqr (= Rf from
        # step 1): guards the QR → bidiag handoff on sm8.0-class devices
        # (same class of failure as the a_qr-copy fence above).
        torch_device_fn.synchronize()
        _bidiag_kernel[(batch,)](
            R_dbdsqr,
            d_out,
            e_out,
            K=k,
            BLOCK_K=block_k,
            num_warps=1 if block_k <= 64 else 4,
        )

    torch_device_fn.synchronize()

    block_k_dbdsqr = triton.next_power_of_2(k)
    max_iters, num_w, block_sweeps = _select_dbdsqr_params(k)

    with torch_device_fn.device(device):
        _fused_dbdsqr_kernel[(batch,)](
            d_out,
            e_out,
            K=k,
            BLOCK_K=block_k_dbdsqr,
            EPS=1.1920928955078125e-07,  # fp32 machine epsilon
            MAX_ITERS=max_iters,
            BLOCK_SWEEPS=block_sweeps,
            num_warps=num_w,
        )

    # Fence before the torch-side reads of d_out (gems_abs): Triton->torch
    # handoff needs the same full device sync as the other stage boundaries
    # on sm8.0-class devices.
    torch_device_fn.synchronize()

    s_out = gems_abs(d_out).to(input.dtype)
    s_sorted = gems_topk(s_out, k, dim=-1, largest=True)[0]
    return s_sorted.reshape(*input.shape[:-2], k).to(input.dtype)


def _svdvals_for_norm(A):
    """SVD dispatch for ord=2/-2/nuc.  Returns (..., K).

    Precision strategy (matches PyTorch CUDA gesvdj / gesvd):
      fp32 → fp32,  fp16/bf16 → upcast to fp32.
      ILUVATAR (CoreX): fp64 inputs raise (no fp64 hardware support).
    """
    in_dtype = A.dtype
    if in_dtype in (torch.float16, torch.bfloat16):
        A = A.float()
    if A.dtype == torch.float64:
        raise RuntimeError(
            f"linalg_matrix_norm: fp64 input is not supported on the "
            f"{flag_gems.vendor_name} backend"
        )
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
        _fro_kernel[(batch,)](
            flat,
            s,
            0,
            M * N,
            1,
            blk_n,
            1,
            TILE_2D=False,
            USE_FP64=_use_fp64_acc(A.dtype),  # fp32 accumulation on CoreX
            num_warps=8,
        )
        return s.reshape(*batch_dims, 1).to(in_dtype)

    # --- rank-2 closed form ------------------------------------------------
    if k == 2 and rows <= _RANK2_BLOCK_R_MAX:
        return _svdvals_rank2(A).to(in_dtype)

    # --- gesvd: all k≥3 through _svdvals_hybrid --------------------
    if 2 < k <= 512 and rows <= 2048:
        return _svdvals_hybrid(A).to(in_dtype)
    # --- unsupported -------------------------------------------------------
    raise NotImplementedError(
        f"FlagGems svdvals: unsupported matrix shape. "
        f"Got batch={batch}, m={M}, n={N} (k={k}, rows={rows}). "
        f"Supported: k=1 (L2 norm), k==2 with rows<={_RANK2_BLOCK_R_MAX}, "
        f"or 2<k<=512 with rows<=2048."
    )
