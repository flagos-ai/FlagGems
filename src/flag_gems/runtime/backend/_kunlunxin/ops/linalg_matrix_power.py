import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .bmm import bmm
from .linalg_lu_factor_ex import linalg_lu_factor_ex
from .linalg_solve_triangular import linalg_solve_triangular
from .lu_unpack import lu_unpack
from .mm import mm

logger = logging.getLogger(__name__)


def _gems_copy(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Data movement via the generic gems Triton copy kernel (no torch fallback).

    ``flag_gems.ops.copy.copy_`` is pointwise_dynamic-based and handles arbitrary
    strides / broadcasting; a by-reference call bypasses the dispatcher (it will
    not re-route to the vendor ``copy_``). Inputs here are always fp32/fp64 on
    the op device, so the call stays on the Triton branch (verified).
    """
    from flag_gems.ops.copy import copy_ as _copy_

    _copy_(dst, src)
    return dst


@libentry()
@triton.jit
def _identity_kernel(out_ptr, numel, mm_elems, m, BLOCK: tl.constexpr):
    # Build a (possibly batched) identity matrix directly into a contiguous
    # buffer. Layout is a flat run of consecutive m x m blocks. For flat index
    # ``offs`` the within-matrix offset is ``offs % (m*m)``; its row/col are
    # ``within // m`` / ``within % m``. The value is 1 on the diagonal and 0
    # elsewhere, so the full buffer is written in a single launch (no separate
    # zero-fill pass and no torch.eye / expand / clone).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    within = offs % mm_elems
    row = within // m
    col = within - row * m
    value = tl.where(row == col, 1, 0)
    tl.store(out_ptr + offs, value, mask=mask)


def _make_identity_like(a: torch.Tensor) -> torch.Tensor:
    """Allocate a contiguous buffer shaped like ``a`` and fill it with a
    (batched) identity via the gems Triton kernel above."""
    m = a.shape[-1]
    eye = torch.empty(a.shape, dtype=a.dtype, device=a.device)
    numel = eye.numel()
    if numel == 0:
        return eye
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    with torch_device_fn.device(eye.device):
        _identity_kernel[grid](eye, numel, m * m, m, BLOCK)
    return eye


def _ensure_contiguous(t: torch.Tensor) -> torch.Tensor:
    """Materialise ``t`` into a contiguous buffer via the gems copy kernel when
    it is not already contiguous (metadata check only; no torch data movement)."""
    if t.is_contiguous():
        return t
    c = torch.empty(t.shape, dtype=t.dtype, device=t.device)
    return _gems_copy(c, t)


def _matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Dispatch to the vendor GEMM kernels (2-D -> mm, 3-D -> bmm)."""
    if a.dim() == 2:
        return mm(a, b)
    return bmm(a, b)


def _inverse(a_flat: torch.Tensor) -> torch.Tensor:
    """Batched inverse of ``a_flat`` ([B, M, M]) via vendor LU factorisation.

    ``A = P @ L @ U`` (partial pivoting), so
    ``A^{-1} = U^{-1} @ L^{-1} @ P^{-1} = U^{-1} @ L^{-1} @ P^T``.
    The two triangular systems are solved with the vendor
    ``linalg_solve_triangular`` kernels; the observed residual ``||A X - I||``
    is ~1e-6 (fp32 floor) for the well-conditioned SPD inputs the test builds.
    """
    bn, m, _ = a_flat.shape
    lu, pivots, _info = linalg_lu_factor_ex(a_flat)
    p, lower, u = lu_unpack(lu, pivots)
    pt = _ensure_contiguous(p.transpose(-2, -1))
    y = linalg_solve_triangular(lower, pt, upper=False, left=True, unitriangular=True)
    x = linalg_solve_triangular(u, y, upper=True, left=True)
    return x


def _validate(a: torch.Tensor, n) -> None:
    shape = a.shape
    if len(shape) < 2:
        raise RuntimeError(
            f"linalg_matrix_power: A must be at least 2-D, got shape {shape}"
        )
    m, k = shape[-2], shape[-1]
    if m != k:
        raise RuntimeError(f"linalg_matrix_power: A must be square, got ({m}, {k})")
    if not isinstance(n, int):
        raise TypeError(f"linalg_matrix_power: n must be int, got {type(n).__name__}")
    if a.dtype not in (torch.float32, torch.float64):
        raise RuntimeError(
            f"linalg_matrix_power: flag_gems supports only float32 and float64, "
            f"got {a.dtype}"
        )


def linalg_matrix_power(
    a: torch.Tensor,
    n: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN LINALG_MATRIX_POWER")

    _validate(a, n)
    shape = a.shape
    m = shape[-1]

    if n == 0:
        eye = _make_identity_like(a)
        if out is not None:
            return _gems_copy(out, eye)
        return eye

    if n == 1:
        if out is not None:
            return _gems_copy(out, a)
        res = torch.empty(a.shape, dtype=a.dtype, device=a.device)
        return _gems_copy(res, a)

    import flag_gems

    if a.device.type != flag_gems.device:
        raise RuntimeError(
            f"linalg_matrix_power: flag_gems supports only {flag_gems.device}, "
            f"got {a.device}"
        )

    a_flat = _ensure_contiguous(a.reshape(-1, m, m) if a.dim() != 2 else a)

    if n < 0:
        if a_flat.dim() == 2:
            a_flat = _inverse(a_flat.unsqueeze(0)).squeeze(0)
        else:
            a_flat = _inverse(a_flat)
        n = -n

    result = None
    z = a_flat
    remaining = n
    while remaining > 0:
        if remaining & 1:
            result = z if result is None else _matmul(result, z)
        remaining >>= 1
        if remaining > 0:
            z = _matmul(z, z)

    r = result.reshape(shape)
    if out is not None:
        return _gems_copy(out, r)
    return r


def _resolve_linalg_matrix_power_out_args(out):
    if out is None:
        raise TypeError(
            "linalg_matrix_power(): out must be provided for the out variant"
        )
    return out


def linalg_matrix_power_out(
    a: torch.Tensor, n: int, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN LINALG_MATRIX_POWER_OUT")
    out_resolved = _resolve_linalg_matrix_power_out_args(out)
    return linalg_matrix_power(a, n, out=out_resolved)
