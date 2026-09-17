import logging

from flag_gems.ops.linalg_norm import _parse_ord, _v_norm
from flag_gems.ops.vector_norm import vector_norm

from .linalg_matrix_norm import linalg_matrix_norm

logger = logging.getLogger(__name__)


def linalg_norm(A, ord=None, dim=None, keepdim=False, *, dtype=None):
    """Mirror ``torch.linalg.norm`` dispatch on Iluvatar.

    Matrix branch (ord="fro"/"nuc", dim as 2-tuple, or 2D input with
    dim=None) reuses the Iluvatar ``linalg_matrix_norm``.  Vector branch
    reuses the generic ``vector_norm`` (Iluvatar has no platform-specific
    vector_norm); the per-row p-norm cases run the fixed ``_v_norm`` kernel
    shared with the generic implementation.
    """
    logger.debug("GEMS_ILUVATAR LINALG_NORM")
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
        return linalg_matrix_norm(
            A,
            "fro" if ord is None else ord,
            (-2, -1) if dim is None else dim,
            keepdim,
            dtype=dtype,
        )
    ord = 2 if ord is None else ord
    if (
        dim is not None
        and len(dim) == 1
        and len(dim) < A.ndim
        and ord not in (2, float("inf"), float("-inf"), 0)
    ):
        return _v_norm(A, ord, dim, keepdim, dtype)
    return vector_norm(A, ord, dim, keepdim, dtype=dtype)
