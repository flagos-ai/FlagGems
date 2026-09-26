import logging

import torch

from .bmm import bmm
from .linalg_lu_factor_ex import linalg_lu_factor_ex
from .linalg_solve_triangular import linalg_solve_triangular
from .lu_unpack import lu_unpack
from .mm import mm

logger = logging.getLogger(__name__)


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
    p, l, u = lu_unpack(lu, pivots)
    pt = p.transpose(-2, -1).contiguous()
    y = linalg_solve_triangular(l, pt, upper=False, left=True, unitriangular=True)
    x = linalg_solve_triangular(u, y, upper=True, left=True)
    return x


def _eye_like(a: torch.Tensor) -> torch.Tensor:
    m = a.shape[-1]
    shape = a.shape
    eye = torch.eye(m, dtype=a.dtype, device=a.device)
    if len(shape) > 2:
        eye = eye.expand(shape[:-2] + (m, m)).clone()
    return eye


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
        eye = _eye_like(a)
        if out is not None:
            out.copy_(eye)
            return out
        return eye

    if n == 1:
        if out is not None:
            out.copy_(a)
            return out
        return a.clone()

    import flag_gems

    if a.device.type != flag_gems.device:
        raise RuntimeError(
            f"linalg_matrix_power: flag_gems supports only {flag_gems.device}, "
            f"got {a.device}"
        )

    a_flat = a.reshape(-1, m, m).contiguous() if a.dim() != 2 else a.contiguous()

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
        out.copy_(r)
        return out
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
