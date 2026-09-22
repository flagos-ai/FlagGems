import logging

import torch

from flag_gems.ops.linalg_norm import _parse_ord, _v_norm

from .linalg_matrix_norm import linalg_matrix_norm
from .vector_norm import vector_norm

logger = logging.getLogger(__name__)


def _matrix_ord_supported(ord):
    """Ascend's linalg_matrix_norm covers the SVD-based ords (2, -2, nuc)
    only; other matrix ords (fro/1/-1/inf/-inf) crash or hang CANN native,
    so linalg_norm rejects them up front instead of delegating."""
    if isinstance(ord, str):
        return ord == "nuc"
    return abs(float(ord)) == 2


def linalg_norm(A, ord=None, dim=None, keepdim=False, *, dtype=None):
    """Mirror ``torch.linalg.norm`` dispatch on Ascend.

    Matrix branch (ord="fro"/"nuc", dim as 2-tuple, or 2D input with
    dim=None) reuses the Ascend ``linalg_matrix_norm``; non-SVD matrix ords
    are rejected up front since they crash or hang CANN native.  Vector
    branch reuses the Ascend ``vector_norm``; the per-row p-norm cases run
    the fixed ``_v_norm`` kernel shared with the generic implementation.

    ``ord`` and ``dtype`` are normalized at this boundary because the Ascend
    ``vector_norm`` cannot take either as-is: an int ord reaches its p-norm
    kernels as an i32 scalar that libdevice ``pow`` has no dispatch key for,
    and its ``dtype=`` branch re-derives the dtype through ``torch.dtype``,
    which raises for every input.
    """
    logger.debug("GEMS_ASCEND LINALG_NORM")
    ord = _parse_ord(ord)
    if dim is not None:
        dim = [dim] if isinstance(dim, int) else list(dim)
        if len(dim) not in (1, 2):
            raise RuntimeError(
                f"linalg.norm: If dim is specified, it must be of length 1 or 2. "
                f"Got {dim}."
            )
    elif ord is not None:
        if A.ndim not in (1, 2):
            raise RuntimeError(
                "linalg.norm: If dim is not specified but ord is, "
                f"the input must be 1D or 2D. Got {A.ndim}D."
            )
    if (
        isinstance(ord, str)
        or (dim is not None and len(dim) == 2)
        or (dim is None and A.ndim == 2)
    ):
        ord = "fro" if ord is None else ord
        if not _matrix_ord_supported(ord):
            raise NotImplementedError(
                f"FlagGems Ascend linalg_norm: matrix norm ord '{ord}' is not "
                "supported; Ascend matrix_norm supports 2, -2, nuc only."
            )
        return linalg_matrix_norm(
            A,
            ord,
            (-2, -1) if dim is None else dim,
            keepdim,
            dtype=dtype,
        )
    ord = 2 if ord is None else ord
    # The Ascend backend's libdevice ``pow`` dispatches through an exact
    # (dtype, dtype) key table -- {(fp32, fp32), (fp16, fp16), (bf16, bf16)} --
    # and never promotes, so ``pow(tl.abs(x), ord)`` in the p-norm kernels
    # raises KeyError((float32, int32)) at compile time when ord arrives as an
    # int (an i32 runtime scalar).  Hand the kernels a float so the exponent is
    # fp32, matching the (fp32, fp32) key -- same workaround as the shared
    # _v_norm kernel's ``ord.to(acc_dtype)`` and _ascend/ops/pow.py.
    ord = float(ord)
    if (
        dim is not None
        and len(dim) == 1
        and len(dim) < A.ndim
        and ord not in (2, float("inf"), float("-inf"), 0)
    ):
        return _v_norm(A, ord, dim, keepdim, dtype)
    # Ascend's vector_norm cannot take dtype=: it re-derives it through
    # torch.dtype(dtype), which raises for any input.  Convert the input here
    # instead and let vector_norm read the dtype off x.
    if dtype is not None:
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        elif not isinstance(dtype, torch.dtype):
            dtype = torch.float32
        A = A.to(dtype)
    return vector_norm(A, ord, dim, keepdim)
