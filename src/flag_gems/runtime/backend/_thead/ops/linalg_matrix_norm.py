"""linalg_matrix_norm -- THEAD (PingTouGe) override.

SVD ords (2/-2/nuc) use the on-device Gram + two-sided Jacobi path.
The 7 shared Triton kernels are imported from
flag_gems.ops.linalg_matrix_norm; the two Gram-Jacobi kernels are
defined locally.
"""

import logging
import math

import torch
import triton
import triton.language as tl

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
from flag_gems.utils import libentry

# ===========================================================================
# Kernel: _rank2_svals_kernel -- closed-form SVD for k=2.
# Used by _svdvals_for_norm and C++ SVD dispatch.  No iteration needed.
# BLOCK_B=1 → one matrix per program (regular).
# BLOCK_B>1 → BLOCK_B matrices per program (vectorized, for tiny rows).
# ===========================================================================

_RANK2_BLOCK_R_MAX = 2048
_TSQR_MIN_ASPECT_RATIO = 8  # Use TSQR when M/N >= 8


def _is_nofp64_backend():
    """Whether the current device backend lacks fp64 compute support.

    NVIDIA CUDA supports fp64; the per-vendor backend files handle their own
    fp64 policy, so this generic path never uses fp32-only accumulation."""
    return False


def _use_fp64_acc(input_dtype):
    """Whether to use fp64 for internal accumulation in reduction kernels.

    - bf16/f16 → fp32 accumulator (fp64 is overkill for half precision)
    - f32      → fp64 if backend supports it, else fp32
    - f64      → fp64
    """
    if input_dtype == torch.float64:
        return True
    if input_dtype == torch.float32:
        return not _is_nofp64_backend()
    return False  # fp16/bf16 → fp32


def _acc_dtype(input_dtype):
    """Accumulator dtype for host-side reduction buffers.

    Mirrors ``_use_fp64_acc`` but returns a concrete torch dtype for use
    with ``torch.zeros / torch.full``.
    """
    if _use_fp64_acc(input_dtype):
        return torch.float64
    return torch.float32


def _select_dbdsqr_params(k):
    """Autotune DBDSQR parameters by k: larger k needs more iterations.

    Returns (MAX_ITERS, num_warps, BLOCK_SWEEPS).  BLOCK_SWEEPS controls the
    inner Golub-Kahan QR sweep count inside ``_fused_dbdsqr_kernel``.

    THEAD uses more conservative parameters: extra sweep budget adds robustness
    for k=3 (the only k that normally reaches DBDSQR on thead; k ≥ 4 goes
    through _svdvals_gram_jacobi).
    """
    if k <= 32:
        return 60, 1, 80
    elif k <= 64:
        return 80, 1, 150
    elif k <= 128:
        return 200, 4, 300
    elif k <= 256:
        return 400, 4, 500
    else:
        return 600, 4, 700


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
        if M <= 2 or N <= 2:
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


# ===========================================================================
# DBDSQR fused kernel -- on-device Golub-Kahan QR (fp64, single launch, zero CPU sync).
# ===========================================================================


def _tsqr_r(a, M, N, device):
    """Two-level TSQR for tall matrices (M >= N, M/N >= 8).

    Stage 1: Partition A into ceil(M/BLOCK_ROWS) chunks, QR each independently.
    Stage 2: Stack R-factors vertically, QR the stack once.

    WARNING: consumes ``a`` as in-place scratch (Householder vectors overwrite
    the input).  Callers must not reuse ``a`` after calling.

    Args:
        a: (batch, M, N) in fp64, tall (M >= N), contiguous
        M, N: matrix dimensions
        device: CUDA device

    Returns:
        R_final: (batch, N, N) in fp64, upper-triangular, or None if
        num_chunks <= 1 (caller should fall back to single-block QR).
    """
    batch = a.shape[0]

    # Choose partition size: > N so stack is smaller than original;
    # target ~M/16 rows per chunk but at least 2*N for numerical benefit.
    # block_rows may be non-power-of-2 after the max() clamp; that is fine
    # because only BLOCK_M (the kernel tile size, always power-of-2) is used
    # for tl.arange inside the kernel.
    target_rows = max(N * 2, M // 16)
    block_rows = triton.next_power_of_2(min(target_rows, max(N * 2, M // 2)))
    block_rows = max(block_rows, N + 8)  # ensure > N

    num_chunks = triton.cdiv(M, block_rows)
    if num_chunks <= 1:
        return None  # degenerate: single chunk -> use single-block QR

    padded_M = num_chunks * block_rows

    # Pad A if needed (zero rows don't affect R)
    if padded_M > M:
        pad = torch.zeros(batch, padded_M - M, N, dtype=torch.float64, device=device)
        a = torch.cat([a, pad], dim=1)

    # Stage 1: QR on each chunk
    # Reshape (batch, padded_M, N) -> (batch * num_chunks, block_rows, N)
    A_chunks = a.reshape(batch * num_chunks, block_rows, N).contiguous()
    R_chunks = torch.zeros(batch * num_chunks, N, N, dtype=torch.float64, device=device)

    # Fence before the chunk QR reads A_chunks: the reshape/contiguous (and
    # the torch.cat pad above) are torch ops, and on sm8.0-class devices a
    # Triton launch does not reliably observe prior writes without a full
    # device sync.  Without this fence the chunk QR occasionally reads a
    # partially-copied matrix under load -- observed on MetaX as rare R
    # deviations of up to ~7e-2 (~2/30 runs), which the Jacobi stage then
    # amplifies into test failures.
    torch_device_fn.synchronize()

    block_m_qr = triton.next_power_of_2(min(block_rows, 256))
    with torch_device_fn.device(device):
        _householder_qr_r_kernel[(batch * num_chunks,)](
            A_chunks,
            R_chunks,
            block_rows,
            N,
            N,
            stride_b=A_chunks.stride(0),
            stride_m=A_chunks.stride(1),
            stride_n=A_chunks.stride(2),
            BLOCK_M=block_m_qr,
            BLOCK_N=32,
            USE_FP64=True,
            num_warps=4,
        )
    torch_device_fn.synchronize()

    # Stage 2: Stack R-factors and QR the stack.
    # R_chunks: (batch * num_chunks, N, N) -> (batch, num_chunks * N, N)
    # No power-of-2 padding needed: the kernel's BLOCK_M tile size is
    # separately power-of-2, and the row count (M arg) supports masking.
    R_stack = R_chunks.reshape(batch, num_chunks * N, N).contiguous()
    stack_rows = num_chunks * N

    # Fence before the stack QR reads R_stack: same class of failure as the
    # A_chunks fence above -- the contiguous copy is a torch op whose writes
    # are not reliably visible to the immediately following Triton launch on
    # sm8.0-class devices.
    torch_device_fn.synchronize()

    R_final = torch.zeros(batch, N, N, dtype=torch.float64, device=device)
    block_m_stack = triton.next_power_of_2(min(stack_rows, 256))
    with torch_device_fn.device(device):
        _householder_qr_r_kernel[(batch,)](
            R_stack,
            R_final,
            stack_rows,
            N,
            N,
            stride_b=R_stack.stride(0),
            stride_m=R_stack.stride(1),
            stride_n=R_stack.stride(2),
            BLOCK_M=block_m_stack,
            BLOCK_N=32,
            USE_FP64=True,
            num_warps=4,
        )

    return R_final


# ===========================================================================
# Two-sided symmetric Jacobi kernels (THEAD Gram-Jacobi SVD path).
# On-device eigensolver for the fp64 Gram matrix -- replaces the former CPU
# torch.linalg.eigvalsh approach (_svdvals_gram_eigh, removed).
# ===========================================================================


@libentry()
@triton.jit
def _symmetric_jacobi_row_kernel(
    A_WORK,
    ROT_C,
    ROT_S,
    ACTIVE_FLAG,
    K,
    STEP,
    BLOCK_K: tl.constexpr,
    PAIR_K: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    """Phase 1 (row rotation): compute Jacobi rotation and apply Jᵀ to rows.

    Grid=(batch, PAIR_K/2).  Each program reads the 2×2 block of its assigned
    pair (p, q) from the *current* G (column-major in A_WORK), computes the
    rotation (c, s) that would zero G[p,q], saves (c, s) to the rotation
    buffers, and updates rows p and q in A_WORK.

    PAIR_K is K for even K; for odd K it is K+1 and the extra index K acts
    as a dummy column: the Brent-Luk ordering runs on PAIR_K indices and
    pairs touching the dummy are skipped.  Without the dummy the ordering is
    incomplete for odd K (several off-diagonal pairs would never be rotated
    and the eigendecomposition could not converge); with it every pair is
    covered and convergence properties match the even case.

    Since pairs are disjoint, row writes are non-conflicting (no two programs
    touch the same row).  The result G' = Jᵀ G is temporarily asymmetric;
    the column-rotation kernel restores symmetry.
    """
    pid0 = tl.program_id(0)  # batch
    j = tl.program_id(1)  # pair index in [0, PAIR_K/2)
    idx = tl.arange(0, BLOCK_K)
    kmask = idx < K
    km1 = PAIR_K - 1

    # Normalise integer types
    j = j + (K - K)
    step_val = STEP + (K - K)

    # Brent-Luk pair assignment
    if j == 0:
        p = step_val
        q = km1
    else:
        p = (step_val + j) % km1
        q = (step_val - j + km1) % km1

    # Pairs touching the dummy column (odd K) do no work.
    valid = (p < K) & (q < K)

    base = A_WORK + pid0 * K * K
    rot_base_c = ROT_C + pid0 * K
    rot_base_s = ROT_S + pid0 * K

    # Scalar loads of the 2×2 block (a_work[r][c] = G[c][r]).
    alpha = tl.load(base + p * K + p, mask=valid, other=0.0)  # G[p,p]
    beta = tl.load(base + q * K + q, mask=valid, other=0.0)  # G[q,q]
    gamma = tl.load(base + q * K + p, mask=valid, other=0.0)  # G[p,q]

    # Jacobi rotation zeroing gamma (same as classical two-sided Jacobi).
    # THRESHOLD is host-controlled (1e-12): drive the off-diagonal as deep as
    # the fp64 ALU allows.  Each program stores its activity into its own
    # slot of ACTIVE_FLAG (slot = pair index j, no atomics); the host stops
    # once a full sweep has no active rotations anywhere.
    threshold = THRESHOLD * tl.maximum(tl.abs(alpha), tl.abs(beta))
    active = (tl.abs(gamma) > threshold) & valid
    # Per-program activity slot: ACTIVE_FLAG has shape (batch, PAIR_K/2) and
    # is fully overwritten every step, so no reset is needed between sweeps.
    tl.store(ACTIVE_FLAG + pid0 * (PAIR_K // 2) + j, tl.where(active, 1, 0))
    safe_gamma = tl.where(active, gamma, 1.0)
    tau = (beta - alpha) / (2.0 * safe_gamma)
    sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
    # THEAD's tl.rsqrt/tl.sqrt are fast-math approximations (~1e-7 relative):
    # a rotation built from them has c^2+s^2 = 1+2e, i.e. is not exactly
    # orthogonal, and each rotation leaks ~2e*(alpha+beta) into the trace
    # (observed +218 over 12 sweeps for k=512, shifting every eigenvalue by
    # ~8e-4 relative -- exactly the 4.3e-4 nuclear-norm bias).  Refine both
    # with Newton steps using only mul/add (probe-verified IEEE on thead):
    # each step squares the relative error, three steps reach machine
    # precision from any seed better than ~1e-2.
    x_t = 1.0 + tau * tau
    r_t = tl.rsqrt(x_t)
    r_t = r_t * (1.5 - 0.5 * x_t * r_t * r_t)
    r_t = r_t * (1.5 - 0.5 * x_t * r_t * r_t)
    r_t = r_t * (1.5 - 0.5 * x_t * r_t * r_t)
    t = sign_tau / (tl.abs(tau) + x_t * r_t)  # sqrt(1+tau^2) = x*rsqrt(x)
    x_c = 1.0 + t * t
    r_c = tl.rsqrt(x_c)
    r_c = r_c * (1.5 - 0.5 * x_c * r_c * r_c)
    r_c = r_c * (1.5 - 0.5 * x_c * r_c * r_c)
    r_c = r_c * (1.5 - 0.5 * x_c * r_c * r_c)
    c = r_c
    s_rot = t * c
    c = tl.where(active, c, 1.0)
    s_rot = tl.where(active, s_rot, 0.0)

    # Save rotation parameters for the column-rotation phase.
    tl.store(rot_base_c + p, c, mask=valid)
    tl.store(rot_base_s + p, s_rot, mask=valid)

    # Apply row rotation:  G' = Jᵀ G
    #   G'[p, i] =  c·G[p, i] - s·G[q, i]
    #   G'[q, i] =  s·G[p, i] + c·G[q, i]
    gip = tl.load(base + idx * K + p, mask=kmask & valid, other=0.0)  # G[p, i]
    giq = tl.load(base + idx * K + q, mask=kmask & valid, other=0.0)  # G[q, i]
    new_ip = c * gip - s_rot * giq
    new_iq = s_rot * gip + c * giq
    # Writes are disjoint across programs (each pair owns its two rows).
    tl.store(base + idx * K + p, new_ip, mask=kmask & valid)
    tl.store(base + idx * K + q, new_iq, mask=kmask & valid)


@libentry()
@triton.jit
def _symmetric_jacobi_col_kernel(
    A_WORK,
    ROT_C,
    ROT_S,
    K,
    STEP,
    BLOCK_K: tl.constexpr,
    PAIR_K: tl.constexpr,
):
    """Phase 2 (column rotation): apply J to the columns of the row-rotated G'.

    Grid=(batch, PAIR_K/2).  Each program reads the saved rotation (c, s) and
    applies the column rotation to the post-row-rotation matrix:
      G'' = G' J = (Jᵀ G) J = Jᵀ G J

    PAIR_K: see _symmetric_jacobi_row_kernel -- for odd K it is K+1 and
    pairs touching the dummy column K are skipped.

    Column writes are disjoint across programs.  After this kernel the
    matrix is again symmetric (G''[i, j] = G''[j, i]).
    """
    pid0 = tl.program_id(0)  # batch
    j = tl.program_id(1)  # pair index in [0, PAIR_K/2)
    idx = tl.arange(0, BLOCK_K)
    kmask = idx < K
    km1 = PAIR_K - 1

    j = j + (K - K)
    step_val = STEP + (K - K)

    if j == 0:
        p = step_val
        q = km1
    else:
        p = (step_val + j) % km1
        q = (step_val - j + km1) % km1

    # Pairs touching the dummy column (odd K) do no work.
    valid = (p < K) & (q < K)

    base = A_WORK + pid0 * K * K
    rot_base_c = ROT_C + pid0 * K
    rot_base_s = ROT_S + pid0 * K

    # Read saved rotation parameters (computed by the row kernel from the
    # pre-step G; these are immutable across the step).
    c = tl.load(rot_base_c + p, mask=valid, other=1.0)
    s_rot = tl.load(rot_base_s + p, mask=valid, other=0.0)

    # Apply column rotation to G' = Jᵀ G (post-row-rotation, asymmetric):
    #   G''[i, p] =  c·G'[i, p] - s·G'[i, q]
    #   G''[i, q] =  s·G'[i, p] + c·G'[i, q]
    # In column-major storage a_work[r][c] = G[c][r]:
    #   G'[i, p] is at a_work[p][i] = base + p*K + i
    gip = tl.load(base + p * K + idx, mask=kmask & valid, other=0.0)  # G'[idx, p]
    giq = tl.load(base + q * K + idx, mask=kmask & valid, other=0.0)  # G'[idx, q]
    new_ip = c * gip - s_rot * giq  # G''[idx, p]
    new_iq = s_rot * gip + c * giq  # G''[idx, q]
    # Writes are disjoint across programs (each pair owns its two columns).
    tl.store(base + p * K + idx, new_ip, mask=kmask & valid)
    tl.store(base + q * K + idx, new_iq, mask=kmask & valid)


def _check_gram_convergence(a_work, conv_tol):
    """Check whether the off-diagonal of the column-major Gram matrix
    is below *conv_tol* times the maximum absolute diagonal value.

    Avoids ``a_work - diag_embed(diag)`` because on thead the fp64
    subtraction re-rounds residual diagonal values (~7.5e-7 relative)
    that can inflate the apparent ``moff``.  Instead we clone and
    explicitly zero the diagonal.
    """
    off = a_work.clone()
    # Zero out the diagonal so only genuine off-diagonal entries remain.
    off_diag = off.diagonal(dim1=-2, dim2=-1)
    off_diag.copy_(torch.zeros_like(off_diag))
    moff = max_dim(max_dim(gems_abs(off), dim=-1).values, dim=-1).values
    diag = a_work.diagonal(dim1=-2, dim2=-1)
    mdiag = max_dim(gems_abs(diag), dim=-1).values
    return bool(gems_all(moff <= conv_tol * mdiag).item())


def _svdvals_gram_jacobi(input, k, batch, m, n):
    """THEAD / HYGON: two-sided parallel Jacobi on the fp64 Gram matrix.

    On-device replacement for the former CPU torch.linalg.eigvalsh path:
    G = AᵀA (or AAᵀ) is formed with one torch fp64 GEMM (IEEE-accurate in
    the thead/hygon torch runtimes) and diagonalised entirely on device by
    the parallel two-sided Jacobi kernels; singular values are
    sqrt(clamp(eigvals, 0)).

    Returns (..., k) in input.dtype, or None on numerical breakdown (caller
    falls back to the legacy QR/Jacobi/DBDSQR pipeline).
    """
    device = input.device
    a = input.contiguous().reshape(batch, m, n).double()

    with torch_device_fn.device(device):
        if m >= n:
            G = a.transpose(-2, -1) @ a  # (batch, k, k)
        else:
            G = a @ a.transpose(-2, -1)

    # Column-major work buffer (a_work[i][j] = G[j][i]): the kernel reads and
    # writes each "column" (a row of G) contiguously.
    a_work = G.transpose(1, 2).contiguous()
    # Rotation-parameter buffers (one (c, s) per column, shared across batches).
    rot_c = torch.empty((batch, k), dtype=torch.float64, device=device)
    rot_s = torch.empty((batch, k), dtype=torch.float64, device=device)
    # Per-program activity flags for the last step of each sweep (host
    # early-exit, no atomics).  Slot = (batch, pair j); overwritten every step.
    # PAIR_K/2 slots per batch: PAIR_K = k for even k, k+1 (dummy column)
    # for odd k -- see _symmetric_jacobi_row_kernel.
    pair_k = k if k % 2 == 0 else k + 1
    active_flag = torch.zeros((batch, pair_k // 2), dtype=torch.int32, device=device)
    block_k = triton.next_power_of_2(k)
    num_w = 1 if block_k <= 64 else 4

    # Rotation threshold 1e-12: keep rotating until the off-diagonal reaches
    # the fp64 rounding floor.  The old 1e-4 threshold left a ~1e-4-relative
    # off-diagonal residue whose eigenvalue error grows like k·τ² — that is
    # what pushed ord=2/-2/nuc past tolerance for k ≥ 128.
    # Dense spectra (large square Gram matrices) converge only linearly until
    # the off-diagonal drops below the minimum eigenvalue gap, so the sweep
    # budget must be generous; the early exits bound the actual cost.
    threshold = 1e-12
    max_sweeps = 60
    # Fence so the first row kernel observes the torch-built a_work (GEMM +
    # transpose/copy above): on sm8.0-class devices Triton launches do not
    # reliably act as memory fences under pressure.
    torch_device_fn.synchronize()
    with torch_device_fn.device(device):
        for sweep in range(max_sweeps):
            # One sweep runs PAIR_K-1 steps (Brent-Luk full round): for odd k
            # this is k steps -- one more than the k-1 used for even k -- and
            # the extra step rotates the pairs the dummy-column ordering
            # leaves for last (e.g. (0,1) for k=3).
            for step in range(pair_k - 1):
                # Phase 1: compute rotation + apply row rotation (Jᵀ G).
                _symmetric_jacobi_row_kernel[(batch, pair_k // 2)](
                    a_work,
                    rot_c,
                    rot_s,
                    active_flag,
                    k,
                    step,
                    BLOCK_K=block_k,
                    PAIR_K=pair_k,
                    THRESHOLD=threshold,
                    num_warps=num_w,
                )
                # Synchronise so column-rotation kernel reads the
                # row-rotated (temporarily asymmetric) G'.
                torch_device_fn.synchronize()
                # Phase 2: apply column rotation ((Jᵀ G) J = Jᵀ G J).
                # Uses the same (c, s) stored in rot_c / rot_s by phase 1.
                _symmetric_jacobi_col_kernel[(batch, pair_k // 2)](
                    a_work,
                    rot_c,
                    rot_s,
                    k,
                    step,
                    BLOCK_K=block_k,
                    PAIR_K=pair_k,
                    num_warps=num_w,
                )
            # Synchronise + clone after each sweep so all writes are
            # globally visible before the next sweep.
            torch_device_fn.synchronize()
            a_work = a_work.clone()

            # A full sweep with zero active rotations means the off-diagonal
            # is at/below THRESHOLD everywhere: further sweeps change nothing.
            if int(active_flag.sum().item()) == 0:
                break
            # Periodic residue check: stop early once the off-diagonal is
            # deep enough that its eigenvalue error is negligible (1e-8).
            if sweep % 8 == 7 and _check_gram_convergence(a_work, 1e-8):
                break

    diag = a_work.diagonal(dim1=-2, dim2=-1)
    if not bool(torch.isfinite(diag).all().item()):
        return None

    # Acceptance ladder (tightest achievable first).  The result is used
    # whenever the off-diagonal reached at least the loosest bound; only a
    # genuine breakdown (no convergence at all) falls back to the legacy
    # QR/Jacobi/DBDSQR pipeline.
    for tol in (1e-10, 1e-8, 1e-6, 1e-4):
        if _check_gram_convergence(a_work, tol):
            break
    else:
        return None

    s = gems_sqrt(torch.clamp(diag, min=0.0))
    s_sorted = gems_topk(s, k, dim=-1, largest=True)[0]
    return s_sorted.reshape(*input.shape[:-2], k).to(input.dtype)


def _qr_rf(a, batch, m, n, k, device, tall, qr_use_fp64, block_m, block_n):
    """QR factorisation producing the triangular R factor (k×k).

    Builds fresh scratch from ``a`` (the QR kernel overwrites its input with
    Householder vectors), then runs either the two-level TSQR (tall,
    high-aspect-ratio inputs) or a single-block Householder QR.  Returns
    (Rf, tsqr_applied).
    """
    if tall:
        a_qr = a.float().clone() if not qr_use_fp64 else a.double().clone()
        M_qr, N_qr = m, n
    else:
        a_qr = a.transpose(-2, -1).contiguous()
        a_qr = a_qr.float().clone() if not qr_use_fp64 else a_qr.double().clone()
        M_qr, N_qr = n, m
    block_m = triton.next_power_of_2(min(M_qr, 256))

    Rf = torch.zeros(
        (batch, k, k),
        dtype=torch.float64 if qr_use_fp64 else torch.float32,
        device=device,
    )

    _tsqr_applied = False
    # Fence before the QR kernel reads a_qr: its backing copies
    # (transpose/contiguous/clone) are torch ops, and on sm8.0-class devices
    # a Triton launch does not reliably observe prior writes without a full
    # device sync (see the QR->Jacobi fence note below).  Without this, the
    # QR factorisation can read a partially-copied matrix under load -- a
    # rare but real corruption (observed once as a ~1.5e-3 sigma_max error
    # on the k=3 keepdim test during a full-suite run).
    torch_device_fn.synchronize()
    with torch_device_fn.device(device):
        # TSQR disabled for THEAD: two-level QR introduces R-factor
        # perturbations that degrade DBDSQR convergence for k ≥ 32.
        if (
            qr_use_fp64
            and M_qr // N_qr >= _TSQR_MIN_ASPECT_RATIO
            and flag_gems.vendor_name != "thead"
        ):
            R_tsqr = _tsqr_r(a_qr, M_qr, N_qr, device)
            if R_tsqr is not None:
                Rf = R_tsqr
                _tsqr_applied = True
            else:
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
                    BLOCK_N=block_n,
                    USE_FP64=qr_use_fp64,
                    num_warps=4,
                )
        else:
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
                BLOCK_N=block_n,
                USE_FP64=qr_use_fp64,
                num_warps=4,
            )
    return Rf, _tsqr_applied


def _qr_r_dbdsqr(a, batch, m, n, k, device, tall, work_dtype, block_m, block_n):
    """Single Householder QR pass on original A for the DBDSQR section.

    Re-factors A in ``work_dtype`` (used when step 1's Rf dtype does not
    match the DBDSQR work dtype, i.e. k=3 on fp64-capable backends).  The
    QR kernel overwrites its input, so fresh scratch is built per call.
    """
    M_qr, N_qr = (m, n) if tall else (n, m)
    if tall:
        a_qr_dbdsqr = a.to(work_dtype).clone()
    else:
        a_qr_dbdsqr = a.transpose(-2, -1).contiguous().to(work_dtype)
    R_dbdsqr = torch.zeros((batch, k, k), dtype=work_dtype, device=device)
    # Fence before the QR kernel reads a_qr_dbdsqr (same class of failure as
    # the a_qr-copy fence in _qr_rf).
    torch_device_fn.synchronize()
    with torch_device_fn.device(device):
        _householder_qr_r_kernel[(batch,)](
            a_qr_dbdsqr,
            R_dbdsqr,
            M_qr,
            N_qr,
            k,
            stride_b=a_qr_dbdsqr.stride(0),
            stride_m=a_qr_dbdsqr.stride(-2),
            stride_n=a_qr_dbdsqr.stride(-1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            USE_FP64=(work_dtype == torch.float64),
            num_warps=4,
        )
    return R_dbdsqr


def _svdvals_hybrid(input):
    """Hybrid SVD: Gram-Jacobi (thead, hygon) / QR+Jacobi+DBDSQR (others).

    Per-backend dispatch:

      THEAD (PingTouGe)
        k ≥ 4:  _svdvals_gram_jacobi (fp64 Gram GEMM on device + parallel
                two-sided Jacobi eigensolver in Triton).  No CPU LAPACK.
        k = 3:  DBDSQR fallback (same as generic, fp64).

      HYGON (DCU)
        k ≥ 3:  _svdvals_gram_jacobi, same as THEAD.  The Triton Householder
                QR kernel is nondeterministic on hygon, so the QR/Jacobi/
                DBDSQR stages are bypassed entirely (odd k uses a dummy
                column in the Brent-Luk pairing, PAIR_K = k + 1).

      ILUVATAR (CoreX)
        k ≥ 4:  fp32 QR + fp32 Jacobi (15–20 sweeps).
                Falls back to fp32 DBDSQR if Jacobi doesn't converge.
        k = 3:  fp32 DBDSQR directly.

      CUDA / MetaX / other
        k > 10:      QR (fp64) + Jacobi (60/50 sweeps fp64, 30/40 sweeps fp32).
        k ∈ (3, 10]: DBDSQR (fp64 or fp32).
        Jacobi → DBDSQR fallback on non-convergence.
    """

    batch, m, n = _svd_shape(input)
    k = min(m, n)
    tall = m >= n
    a = input.contiguous().reshape(batch, m, n)
    device = input.device

    # ---- Gram + on-device Jacobi eigensolver (k ≥ 4) -----------------------
    # A single torch fp64 GEMM (IEEE-accurate, see probe_fp64_precision.py)
    # plus the parallel two-sided Jacobi kernels -- the whole
    # eigendecomposition stays on device, no CPU LAPACK involved.
    if k >= 4:
        s = _svdvals_gram_jacobi(input, k, batch, m, n)
        if s is not None:
            return s
        # Numerical breakdown (rare): fall through to the legacy
        # QR/Jacobi/DBDSQR path below.

    # ---- 1. QR: A → triangular R (k×k) -------------------------------------
    # fp64 for k ≥ 4 on backends that support it (CUDA, MetaX, THEAD).
    # ILUVATAR / Ascend use fp32 throughout (_is_nofp64_backend → True).
    M_qr, N_qr = (m, n) if tall else (n, m)
    block_m = triton.next_power_of_2(min(M_qr, 256))
    block_n = 32
    qr_use_fp64 = k >= 4 and not _is_nofp64_backend()

    Rf, _tsqr_applied = _qr_rf(
        a, batch, m, n, k, device, tall, qr_use_fp64, block_m, block_n
    )

    # Fence before Jacobi reads Rf: on SM 8.0 (A100) Triton kernel launches
    # do not reliably act as memory fences under memory pressure, so a
    # subsequent read of Rf (via Rf.transpose().contiguous() below) can
    # observe stale data.  A full device sync guarantees the QR output is
    # visible before the Jacobi sweep reads it.  Mirrors the per-sweep
    # synchronize()+clone() fence already used inside the Jacobi loop.
    torch_device_fn.synchronize()

    # ---- 2. Jacobi SVD on R ------------------------------------------------
    # THEAD: Jacobi only for 10 < k < 32 (narrow window where Jacobi is
    #   faster than DBDSQR but small enough that thead's fp64 noise is
    #   tolerable).  For k ≥ 4, THEAD is normally caught by the Gram-Jacobi
    #   early-exit above and only reaches here if that breaks down.
    # ILUVATAR: Jacobi for all k ≥ 4, lower sweep counts (fp32 precision
    #   floor is higher, so additional sweeps don't help).
    # OTHER: Jacobi for k > 10.
    # Per-backend Jacobi dispatch.  TSQR matrices always retain Jacobi:
    # DBDSQR doesn't drown out the ULPs-level differences from the two-level
    # QR partitioning, so final singular values vary non-deterministically
    # across runs on SM 8.0 GPUs.  Jacobi's 60 sweeps drive the off-diagonal
    # deep below the convergence floor, producing identical results regardless
    # of GPU arch.
    use_jacobi = k > 10 and k < 32

    if use_jacobi:
        # Per-backend Jacobi sweep counts
        if Rf.dtype == torch.float64:
            _JACOBI_SWEEPS = 80 if k <= 48 else 70
        else:
            _JACOBI_SWEEPS = 40 if k <= 48 else 50

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
        rel_tol = 1e-6 if Rf.dtype == torch.float64 else 5e-4
        jacobi_ok = bool(gems_all(max_off <= rel_tol * max_diag).item())

        if jacobi_ok:
            if Rf.dtype == torch.float64:
                col_norms = a_work.norm(dim=-1)
            else:
                col_norms = gems_sqrt((a_work * a_work).sum(dim=-1).clamp(min=0.0))
            s_sorted = gems_topk(col_norms, k, dim=-1, largest=True)[0]
            return s_sorted.reshape(*input.shape[:-2], k).to(input.dtype)
        # Non-converged → fall through to DBDSQR

    # ---- 3. DBDSQR: bidiagonalisation + Golub-Kahan QR ----------------------
    # THEAD reuses Rf from step 1 (normally only reaches here for k=3, where
    # Rf is clean; k ≥ 4 arrives only if Gram-Jacobi breaks down).
    # ILUVATAR re-does QR from original A (step 1 was in fp32, but
    # work_dtype is also fp32, so reuse is fine).
    # OTHER backends re-do QR from original A because step 1's a_qr is
    # corrupted by in-place Householder vectors.
    work_dtype = torch.float32 if _is_nofp64_backend() else torch.float64
    block_k = triton.next_power_of_2(k)
    d_out = torch.zeros((batch, k), dtype=work_dtype, device=device)
    e_out = torch.zeros((batch, k - 1), dtype=work_dtype, device=device)

    with torch_device_fn.device(device):
        # Reuse Rf from step 1 whenever dtypes match — preserves TSQR precision
        # for tall matrices (step 1's QR already applied two-level partitioning).
        # Re-doing single-block QR would lose that precision and re-introduce
        # non-deterministic run-to-run variation for M/N >= 8 on SM 8.0 GPUs.
        # THEAD already unconditionally reuses Rf; iluvatar always takes the
        # Jacobi path above for k >= 4, so this fallback only matters for
        # CUDA/MetaX k <= 10 (no Jacobi).
        # THEAD reuses Rf from step 1 (normally only reaches here for k=3, where
        # Rf is clean; k ≥ 4 arrives only if Gram-Jacobi breaks down).
        R_dbdsqr = Rf if Rf.dtype == work_dtype else Rf.to(work_dtype)
        # Fence before the bidiagonalisation kernel reads R_dbdsqr: the
        # Rf.to(work_dtype) copy above is a torch op, and without a full
        # device sync the kernel may read it before the copy's writes are
        # visible on sm8.0-class devices (same class of failure as the
        # a_qr-copy fence above).
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
            EPS=(
                1.1920928955078125e-07
                if _is_nofp64_backend()
                else 2.220446049250313e-16
            ),
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
      fp64 → fp64,  fp32 → fp32,  fp16/bf16 → upcast to fp32.
      ILUVATAR / Ascend: fp64 inputs raise (no fp64 hardware support).
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
        _fro_kernel[(batch,)](
            flat,
            s,
            0,
            M * N,
            1,
            blk_n,
            1,
            TILE_2D=False,
            USE_FP64=True,
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
