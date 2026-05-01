"""Singular value decomposition (reference path).

Numerics and batching follow PyTorch by running ``torch.svd`` /
``torch.linalg.svd`` on a detached CPU tensor, then moving results back to
``inp.device``. This avoids recursive dispatch into the same CUDA/MPS
implementation while the library does not yet ship a fused Triton SVD.
"""

from __future__ import annotations

import logging
import torch

logger = logging.getLogger(__name__)


def _require_matrix(a: torch.Tensor) -> None:
    if a.ndim < 2:
        raise RuntimeError("svd: input should have at least 2 dimensions.")


def _require_float_or_complex(a: torch.Tensor) -> None:
    if not (a.dtype.is_floating_point or a.dtype.is_complex):
        raise RuntimeError(
            f"svd: Expected a floating point or complex tensor. Got {a.dtype}"
        )


def _cpu_work_tensor(a: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return (cpu_tensor, promoted_from_low_precision)."""
    if a.dtype in (torch.float16, torch.bfloat16):
        return a.detach().cpu().to(torch.float32), True
    return a.detach().cpu(), False


def _s_dtype_for_orig(orig_dtype: torch.dtype) -> torch.dtype:
    if orig_dtype.is_complex:
        return torch.float32 if orig_dtype == torch.complex64 else torch.float64
    if orig_dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return orig_dtype


def _restore(
    u: torch.Tensor,
    s: torch.Tensor,
    v_or_vh: torch.Tensor,
    *,
    device: torch.device,
    orig_dtype: torch.dtype,
    promoted: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if promoted:
        return (
            u.to(device=device, dtype=orig_dtype),
            s.to(device=device, dtype=_s_dtype_for_orig(orig_dtype)),
            v_or_vh.to(device=device, dtype=orig_dtype),
        )
    return u.to(device=device), s.to(device=device), v_or_vh.to(device=device)


def svd(inp: torch.Tensor, some: bool = True, compute_uv: bool = True):
    """``aten::svd`` — batched SVD with optional economy-size factors."""
    logger.debug("GEMS SVD")
    _require_matrix(inp)
    _require_float_or_complex(inp)

    work, promoted = _cpu_work_tensor(inp)
    u, s, v = torch.svd(work, some=some, compute_uv=compute_uv)
    return _restore(u, s, v, device=inp.device, orig_dtype=inp.dtype, promoted=promoted)


def linalg_svd(
    a: torch.Tensor,
    full_matrices: bool = True,
    *,
    driver: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``aten::linalg_svd`` — returns ``(U, S, Vh)`` matching :func:`torch.linalg.svd`."""
    logger.debug("GEMS LINALG SVD")
    _require_matrix(a)
    _require_float_or_complex(a)

    work, promoted = _cpu_work_tensor(a)
    if driver is None:
        u, s, vh = torch.linalg.svd(work, full_matrices=full_matrices)
    else:
        u, s, vh = torch.linalg.svd(work, full_matrices=full_matrices, driver=driver)
    return _restore(
        u,
        s,
        vh,
        device=a.device,
        orig_dtype=a.dtype,
        promoted=promoted,
    )
