"""Kunlunxin/XPU implementation of ``pinverse`` (legacy alias of ``linalg_pinv``).

The generic ``flag_gems.ops.pinverse`` builds the pseudoinverse from the generic
``svd``, which on XPU refuses several shapes this operator supports (4x4,
rectangular batches, ...).  ``torch.pinverse(A, rcond)`` is
``torch.linalg.pinv(A, rtol=rcond)``: SVD ``A = U S V^H`` then
``pinv = V diag(1/s) U^H`` with singular values ``<= rcond * s_max`` zeroed.

This backend module uses the backend's own one-sided Jacobi SVD
(``_kunlunxin/ops/linalg_svd.py``, pure Triton) and the generic reconstruction
kernel ``flag_gems.ops.pinverse._pinverse_reconstruct_kernel`` (a Triton
kernel, so reusing it is not a torch/aten fallback).  ``V^H`` is fed with
swapped strides instead of materialising ``V``: ``V[n, k] == Vh[k, n]``.
"""

import importlib
import logging

import torch
import triton

from flag_gems.ops.pinverse import _pinverse_reconstruct_kernel
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


# Largest dimension routed through the SVD path; above it linalg_pinv is used.
_SVD_MAX_DIM = 64
# Thin matrices (min dim <= this) above _SVD_MAX_DIM use the normal equations.
_THIN_MAX_DIM = 8


def flat_of(t, batch, m, n):
    return t.reshape(batch, m, n)


def _backend_bmm_module():
    return importlib.import_module(".bmm", __package__)


def _thin_pinv(a):
    """Full-rank thin matrix pseudoinverse via G^-1 (G = A^H A or A A^H)."""
    b, m, n = a.shape
    bmm = _backend_bmm_module().bmm
    inv = importlib.import_module(".inverse", __package__).inverse
    at = a.transpose(-2, -1).contiguous()
    if m >= n:
        g = bmm(at, a)  # (b, n, n)
        pinv = bmm(inv(g), at)  # (b, n, m)
    else:
        g = bmm(a, at)  # (b, m, m)
        pinv = bmm(at, inv(g))  # (b, n, m)
    return pinv


def _backend_pinv_module():
    return importlib.import_module(".linalg_pinv", __package__)


def _backend_svd_module():
    # Submodule import, not the same-named function (see slogdet.py).
    return importlib.import_module(".linalg_svd", __package__)


def pinverse(inp, rcond=1e-15):
    """Compute the Moore-Penrose pseudoinverse of a matrix or matrix batch."""
    logger.debug("GEMS_KUNLUNXIN PINVERSE")

    if inp.ndim < 2:
        raise RuntimeError("pinverse: expected a tensor with at least 2 dimensions")
    if inp.dtype != torch.float32:
        raise NotImplementedError(
            "FlagGems pinverse currently supports only float32 tensors"
        )
    if inp.requires_grad:
        raise NotImplementedError(
            "FlagGems pinverse does not yet support autograd inputs"
        )
    if rcond < 0.0:
        raise NotImplementedError(
            "FlagGems pinverse does not yet support negative rcond values"
        )

    m, n = inp.shape[-2:]
    if inp.numel() == 0:
        return torch.empty((*inp.shape[:-2], n, m), dtype=inp.dtype, device=inp.device)

    if not inp.is_contiguous():
        inp = inp.contiguous()

    batch = inp.numel() // (m * n)
    k = min(m, n)

    # Large matrices: the backend ``linalg_pinv`` (register/blocked one-sided
    # Jacobi with a fused or matmul contraction) is far cheaper than running
    # the 1024-wide SVD.  It is numerically weaker on a few ill-conditioned
    # mid-size cases, hence the size split.
    if min(m, n) <= _THIN_MAX_DIM and max(m, n) > _SVD_MAX_DIM:
        # Very thin matrices (benchmark's (256,2)/(1024,2)/...): the generic
        # blocked linalg_pinv kernel fails to compile for this aspect ratio,
        # while the normal equations G^-1 reduce to a tiny r x r system.
        return _thin_pinv(flat_of(inp, batch, m, n))
    if max(m, n) > _SVD_MAX_DIM:
        return _backend_pinv_module().linalg_pinv(inp, rtol=float(rcond))

    # The backend SVD consumes a single batch dimension; flatten multi-dim
    # batches here (the caller's shape is restored at the end).
    flat = inp.reshape(batch, m, n)
    u, s, vh = _backend_svd_module().linalg_svd(flat, full_matrices=False)
    u = u.reshape(batch, m, k)
    s = s.reshape(batch, k)
    vh = vh.reshape(batch, k, n)

    output = torch.empty((batch, n, m), dtype=inp.dtype, device=inp.device)

    block_m = 16 if m <= 16 else 32
    block_n = 16 if n <= 16 else 32
    block_k = 32
    grid = (triton.cdiv(n, block_n) * triton.cdiv(m, block_m) * batch,)
    with torch_device_fn.device(inp.device):
        _pinverse_reconstruct_kernel[grid](
            u,
            s,
            vh,
            output,
            rcond,
            u.stride(0),
            u.stride(1),
            u.stride(2),
            s.stride(0),
            s.stride(1),
            # V is consumed as V[n, k] = Vh[k, n] via swapped strides; feeding
            # Vh directly avoids materialising V (a data copy).
            vh.stride(0),
            vh.stride(2),
            vh.stride(1),
            M=m,
            N=n,
            K=k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=1 if block_m == 16 and block_n == 16 else 4,
        )
    return output.reshape((*inp.shape[:-2], n, m))
