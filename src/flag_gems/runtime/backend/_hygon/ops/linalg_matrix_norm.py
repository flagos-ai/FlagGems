import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_matrix_norm import _fro_norm as _fro_norm_generic
from flag_gems.ops.linalg_matrix_norm import _nuc_norm as _nuc_norm_generic
from flag_gems.ops.linalg_matrix_norm import _ord1_norm as _ord1_norm_generic
from flag_gems.ops.linalg_matrix_norm import _ord2_norm as _ord2_norm_generic
from flag_gems.ops.linalg_matrix_norm import _use_fp64_acc
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_SUPPORTED_NUMERIC = {1, -1, 2, -2, float("inf"), -float("inf")}


# ===========================================================================
# Kernel: _fro_single_kernel -- single-launch Frobenius norm for large 2D
# matrices (total > 65536).
#
# grid=(NB,).  Each program grid-strides over the flattened matrix in BLOCK
# element chunks, accumulating a local sum-of-squares, then atomically adds it
# into a single scalar Partial.  A second atomic counter Count records arrivals;
# the LAST program to arrive (the one whose increment returns NB-1) reads the
# accumulated Partial, stores sqrt to Out and resets Partial/Count for the next
# call.  The ``sem="acq_rel"`` atomics guarantee the last program observes every
# earlier partial before reading Partial.  No second launch and no spin barrier.
# ===========================================================================


@libentry()
@triton.jit
def _fro_single_kernel(
    X,
    Partial,
    Count,
    Out,
    TOT,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
    ACC: tl.constexpr,
):
    pid = tl.program_id(0)
    acc = tl.zeros([BLOCK], dtype=ACC)
    # grid-stride: program pid handles BLOCK-wide chunks pid, pid+NB, ... so the
    # NB programs together cover [0, TOT).
    for start in range(pid * BLOCK, TOT, NB * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < TOT
        x = tl.load(X + offs, mask=mask, other=0.0).to(ACC)
        acc += x * x
    s = tl.sum(acc)
    tl.atomic_add(Partial, s, sem="acq_rel")
    old = tl.atomic_add(Count, 1, sem="acq_rel")
    if old == NB - 1:
        # Last program to arrive: reduce and reset the scratch for the next call.
        tl.store(Out, tl.sqrt(tl.load(Partial)))
        tl.store(Partial, 0.0)
        tl.store(Count, 0)


# ===========================================================================
# Kernel: _row_atomic_absmax_kernel -- single-launch inf-norm for 2D matrices.
#
# grid=(M,): one program per row reduces the full contiguous row (1D, coalesced)
# and folds its row sum into ONE atomic_max/min into a scalar Out.  Works for
# every M (including very small M like 2) and odd shapes.  ACC/Out may be fp32
# or fp64 (Hygon's atomic_max/min support fp64, which the generic kernels
# assume is impossible).  This replaces both the 2-launch ``_row_abssum`` path
# and the row-block ``_row_absmax`` path used before.
# ===========================================================================


@libentry()
@triton.jit
def _row_atomic_absmax_kernel(
    X,
    Out,
    M,
    N,
    BLOCK: tl.constexpr,
    ACC: tl.constexpr,
    IS_MIN: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=ACC)
    for start in range(0, N, BLOCK):
        idx = start + offs
        mask = idx < N
        x = tl.load(X + pid * N + idx, mask=mask, other=0.0).to(ACC)
        acc += tl.where(mask, tl.abs(x), 0.0)
    s = tl.sum(acc)
    if IS_MIN:
        tl.atomic_min(Out, s)
    else:
        tl.atomic_max(Out, s)


def _row_atomic_absmax(A, M, N, is_min, use_fp64):
    """Single-launch inf-norm over (M, N) -> scalar (fp32 or fp64 Out)."""
    blk = triton.next_power_of_2(min(N, 2048))
    acc_d = torch.float64 if use_fp64 else torch.float32
    init = float("inf") if is_min else 0.0
    out = torch.full((), init, dtype=acc_d, device=A.device)
    acc = tl.float64 if use_fp64 else tl.float32
    _row_atomic_absmax_kernel[(M,)](
        A,
        out,
        M,
        N,
        blk,
        acc,
        IS_MIN=is_min,
        num_warps=4,
    )
    return out


# ===========================================================================
# Kernel: _batched_row_absmax_kernel -- batched inf-norm in a single launch.
# grid=(batch*M,): one program per row computes the |x| row sum and
# atomic_max/min's it into Out[batch].  Per-matrix row count M is typically
# small, so the per-address atomic contention is low.  This avoids both the
# extra launch of a two-stage reduction and the generic ``_abs_norm_kernel``
# tile that miscompiles on Hygon.
# ===========================================================================


@libentry()
@triton.jit
def _batched_row_absmax_kernel(
    X,
    Out,
    M,
    N,
    BLOCK: tl.constexpr,
    ACC: tl.constexpr,
    IS_MIN: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // M
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=ACC)
    for start in range(0, N, BLOCK):
        idx = start + offs
        mask = idx < N
        x = tl.load(X + pid * N + idx, mask=mask, other=0.0).to(ACC)
        acc += tl.where(mask, tl.abs(x), 0.0)
    s = tl.sum(acc)
    if IS_MIN:
        tl.atomic_min(Out + b, s)
    else:
        tl.atomic_max(Out + b, s)


def _batched_inf_norm(Ab, batch, M, N, is_min, use_fp64):
    """Correct batched inf-norm over (batch, M, N) -> (batch,).

    Single launch via ``_batched_row_absmax_kernel``, avoiding the generic
    ``_abs_norm_kernel`` tile that miscompiles on Hygon.  Out is fp32 or
    fp64 (Hygon's atomic_max/min support fp64, matching the 2D
    ``_row_atomic_absmax`` path and ``_batched_col_absmax``); the host casts
    it to ``out_dtype``.
    """
    blk = triton.next_power_of_2(min(N, 2048))
    acc_d = torch.float64 if use_fp64 else torch.float32
    init = float("inf") if is_min else 0.0
    out = torch.full((batch,), init, dtype=acc_d, device=Ab.device)
    acc = tl.float64 if use_fp64 else tl.float32
    _batched_row_absmax_kernel[(batch * M,)](
        Ab.reshape(batch * M, N),
        out,
        M,
        N,
        blk,
        acc,
        IS_MIN=is_min,
        num_warps=4,
    )
    return out


# ===========================================================================
# Kernel: _col_absmax_kernel -- single-launch 1-norm for 2D matrices.
#
# Each program owns a BLOCK of contiguous columns (coalesced row reads), sums
# |x| down the rows locally, and folds its own columns into ONE atomic_max/min
# into a scalar Out.  No partial buffer, no second launch, and only
# ``cdiv(N, COL_BLOCK)`` atomic ops.  ACC/Out may be fp32 (f16/bf16/f32) or
# fp64 (fp64), matching the precision torch itself uses per dtype.
# ===========================================================================


@libentry()
@triton.jit
def _col_absmax_kernel(
    X,
    Out,
    M,
    N,
    COL_BLOCK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    ACC: tl.constexpr,
    IS_MIN: tl.constexpr,
):
    pid = tl.program_id(0)
    col_start = pid * COL_BLOCK
    offs = col_start + tl.arange(0, COL_BLOCK)
    col_mask = offs < N
    acc = tl.zeros([COL_BLOCK], dtype=ACC)
    for row_start in range(0, M, ROW_BLOCK):
        rows = row_start + tl.arange(0, ROW_BLOCK)[:, None]
        x = tl.load(
            X + rows * N + offs[None, :],
            mask=(rows < M) & col_mask[None, :],
            other=0.0,
        ).to(ACC)
        acc += tl.sum(tl.abs(x), axis=0)
    if IS_MIN:
        tl.atomic_min(Out, tl.min(tl.where(col_mask, acc, float("inf"))))
    else:
        tl.atomic_max(Out, tl.max(tl.where(col_mask, acc, float("-inf"))))


def _col_absmax(A, M, N, is_min, use_fp64):
    """Single-launch 1-norm over (M, N) -> scalar (fp32 or fp64 Out)."""
    cb = 32 if N > 32 else triton.next_power_of_2(N)
    rb = 128 if M > 128 else triton.next_power_of_2(M)
    acc_d = torch.float64 if use_fp64 else torch.float32
    init = float("inf") if is_min else 0.0
    out = torch.full((), init, dtype=acc_d, device=A.device)
    _col_absmax_kernel[(triton.cdiv(N, cb),)](
        A,
        out,
        M,
        N,
        cb,
        rb,
        tl.float64 if use_fp64 else tl.float32,
        IS_MIN=is_min,
        num_warps=4,
    )
    return out


# ===========================================================================
# Kernel: _batched_col_absmax_kernel -- single-launch 1-norm for batched
# matrices.  grid=(batch * cdiv(N, COL_BLOCK),): each program reduces one
# column block of one matrix and atomic_max/min's the block's partial max into
# Out[batch].  ACC/Out may be fp32 or fp64.
# ===========================================================================


@libentry()
@triton.jit
def _batched_col_absmax_kernel(
    X,
    Out,
    M,
    N,
    NCB: tl.constexpr,
    COL_BLOCK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    ACC: tl.constexpr,
    IS_MIN: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // NCB
    cb = pid % NCB
    col_start = cb * COL_BLOCK
    base = b * (M * N)
    offs = col_start + tl.arange(0, COL_BLOCK)
    col_mask = offs < N
    acc = tl.zeros([COL_BLOCK], dtype=ACC)
    for row_start in range(0, M, ROW_BLOCK):
        rows = row_start + tl.arange(0, ROW_BLOCK)[:, None]
        x = tl.load(
            X + base + rows * N + offs[None, :],
            mask=(rows < M) & col_mask[None, :],
            other=0.0,
        ).to(ACC)
        acc += tl.sum(tl.abs(x), axis=0)
    if IS_MIN:
        tl.atomic_min(Out + b, tl.min(tl.where(col_mask, acc, float("inf"))))
    else:
        tl.atomic_max(Out + b, tl.max(tl.where(col_mask, acc, float("-inf"))))


def _batched_col_absmax(Ab, batch, M, N, is_min, use_fp64):
    """Single-launch 1-norm over (batch, M, N) -> (batch,) (fp32 or fp64 Out)."""
    cb = 32 if N > 32 else triton.next_power_of_2(N)
    rb = 128 if M > 128 else triton.next_power_of_2(M)
    ncb = triton.cdiv(N, cb)
    acc_d = torch.float64 if use_fp64 else torch.float32
    init = float("inf") if is_min else 0.0
    out = torch.full((batch,), init, dtype=acc_d, device=Ab.device)
    acc = tl.float64 if use_fp64 else tl.float32
    _batched_col_absmax_kernel[(batch * ncb,)](
        Ab,
        out,
        M,
        N,
        ncb,
        cb,
        rb,
        acc,
        IS_MIN=is_min,
        num_warps=4,
    )
    return out


# ===========================================================================
# Per-ord helpers
# ===========================================================================


def _fro_norm(A, dim, keepdim, dtype):
    """Frobenius norm -- single-launch large-2D path, otherwise generic."""
    d0, d1 = dim
    if A.ndim == 2 and d0 == 0 and d1 == 1:
        M, N = A.shape
        total = M * N
        if total > 65536:
            out_dtype = dtype if dtype is not None else A.dtype
            use_fp64 = _use_fp64_acc(A.dtype)
            acc = tl.float64 if use_fp64 else tl.float32
            acc_dtype = torch.float64 if use_fp64 else torch.float32
            nb = min(triton.cdiv(total, 2048), 128)
            nb = max(nb, 1)
            flat = A.reshape(total)
            partial = torch.zeros((), dtype=acc_dtype, device=A.device)
            count = torch.zeros((), dtype=torch.int32, device=A.device)
            out = torch.empty((), dtype=acc_dtype, device=A.device)
            _fro_single_kernel[(nb,)](
                flat,
                partial,
                count,
                out,
                total,
                nb,
                2048,
                acc,
                num_warps=4,
            )
            result = out.to(out_dtype)
            if keepdim:
                result = result.reshape(1, 1)
            return result
    return _fro_norm_generic(A, dim, keepdim, dtype)


def _ord1_norm(A, ord_val, dim, keepdim, dtype):
    """1-norm -- single-launch 2D column path and batched path, else generic."""
    d0, d1 = dim
    out_dtype = dtype if dtype is not None else A.dtype
    is_min = ord_val < 0

    if A.ndim == 2 and d0 == 0 and d1 == 1:
        M, N = A.shape
        use_fp64 = _use_fp64_acc(A.dtype)
        result = _col_absmax(A, M, N, is_min, use_fp64).to(out_dtype).view(())
        if keepdim:
            result = result.reshape(1, 1)
        return result

    if A.ndim > 2:
        ndim = A.ndim
        all_dims = list(range(ndim))
        remaining = [d for d in all_dims if d != d0 and d != d1]
        perm = remaining + [d0, d1]
        A_perm = A.permute(perm) if perm != all_dims else A
        if dtype is not None:
            A_perm = A_perm.to(dtype)
        batch = 1
        for i in range(A_perm.ndim - 2):
            batch *= A_perm.size(i)
        mat_M = A_perm.size(-2)
        mat_N = A_perm.size(-1)
        Ab = A_perm.reshape(batch, mat_M, mat_N).contiguous()

        use_fp64 = _use_fp64_acc(Ab.dtype)
        result = _batched_col_absmax(Ab, batch, mat_M, mat_N, is_min, use_fp64)

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

    # 2D with non-standard dims (dim != (0, 1)): reuse the generic
    # implementation, which permutes the target dims correctly.
    return _ord1_norm_generic(A, ord_val, dim, keepdim, dtype)


def _ordinf_norm(A, ord_val, dim, keepdim, dtype):
    """Infinity-norm -- single-launch per-row reduction for 2D and batched."""
    d0, d1 = dim
    out_dtype = dtype if dtype is not None else A.dtype
    is_min = ord_val < 0

    if A.ndim == 2 and d0 == 0 and d1 == 1:
        M, N = A.shape
        use_fp64 = _use_fp64_acc(A.dtype)
        result = _row_atomic_absmax(A, M, N, is_min, use_fp64).to(out_dtype).view(())
        if keepdim:
            result = result.reshape(1, 1)
        return result

    # Batched path.
    ndim = A.ndim
    all_dims = list(range(ndim))
    remaining = [d for d in all_dims if d != d0 and d != d1]
    perm = remaining + [d0, d1]
    A_perm = A.permute(perm) if perm != all_dims else A
    if dtype is not None:
        A_perm = A_perm.to(dtype)
    batch = 1
    for i in range(A_perm.ndim - 2):
        batch *= A_perm.size(i)
    mat_M = A_perm.size(-2)
    mat_N = A_perm.size(-1)
    Ab = A_perm.reshape(batch, mat_M, mat_N).contiguous()

    use_fp64 = _use_fp64_acc(Ab.dtype)
    result = _batched_inf_norm(Ab, batch, mat_M, mat_N, is_min, use_fp64)

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


# ===========================================================================
# Main entry point
# ===========================================================================


def linalg_matrix_norm(A, ord="fro", dim=(-2, -1), keepdim=False, dtype=None):
    """Matrix norm -- Hygon dispatch, mirrors the generic entry point."""
    logger.debug("GEMS LINALG_MATRIX_NORM (hygon)")

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

    # dtype guard for SVD-based ords
    _svd_ord = (isinstance(ord, str) and ord == "nuc") or (
        not isinstance(ord, str) and abs(float(ord)) == 2
    )
    if _svd_ord and A.dtype in (torch.float16, torch.bfloat16):
        A = A.float()  # upcast to fp32 for SVD

    if isinstance(ord, str):
        if ord == "fro":
            _r = _fro_norm(A, dim, keepdim, dtype)
        elif ord == "nuc":
            _r = _nuc_norm_generic(A, dim=dim, keepdim=keepdim, dtype=dtype)
        else:
            raise RuntimeError(
                f"linalg_matrix_norm: Order '{ord}' not supported. "
                "Use 'fro' or 'nuc'."
            )
        return _r

    ord_val = float(ord)
    if ord_val not in _SUPPORTED_NUMERIC:
        raise RuntimeError(
            f"linalg_matrix_norm: Order {ord} not supported. "
            "Use 1, -1, 2, -2, inf, -inf."
        )

    abs_ord = abs(ord_val)
    if abs_ord == 2.0:
        _r = _ord2_norm_generic(A, ord_val, dim, keepdim, dtype)
    elif abs_ord == 1.0:
        _r = _ord1_norm(A, ord_val, dim, keepdim, dtype)
    elif math.isinf(abs_ord):
        _r = _ordinf_norm(A, ord_val, dim, keepdim, dtype)
    else:
        raise RuntimeError(f"linalg_matrix_norm: Order {ord} not supported.")
    return _r
