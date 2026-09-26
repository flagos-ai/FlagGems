import logging
import math
from collections import namedtuple

import torch

from .bmm import bmm
from .linalg_lu_factor_ex import linalg_lu_factor_ex
from .linalg_solve_triangular import linalg_solve_triangular
from .lu_unpack import lu_unpack

logger = logging.getLogger(__name__)

LinalgSolveExResult = namedtuple("LinalgSolveExResult", ["result", "info"])


def linalg_solve_ex(A, B, *, left=True, check_errors=False):
    """Solve ``AX = B`` returning ``(result, info)`` via vendor LU + trsm."""
    logger.debug("GEMS_KUNLUNXIN LINALG_SOLVE_EX")

    if not left:
        raise NotImplementedError("right=True (XA = B) is not yet supported")
    assert A.dtype in (
        torch.float32,
        torch.float64,
    ), f"linalg_solve_ex requires float32/float64, got {A.dtype}"
    if A.ndim < 2 or B.ndim < 2:
        raise ValueError("A and B must be at least 2D")
    if A.shape[-1] != A.shape[-2]:
        raise ValueError("A must be a square matrix")
    n = A.shape[-1]
    if B.shape[-2] != n:
        raise ValueError("B must have compatible dimensions with A")

    batch_shape = A.shape[:-2]
    if A.numel() == 0 or B.numel() == 0:
        info = torch.zeros(batch_shape, dtype=torch.int32, device=A.device)
        return LinalgSolveExResult(B.clone(), info)

    batch = math.prod(batch_shape) if batch_shape else 1
    nrhs = B.shape[-1]

    A_flat = A.reshape(batch, n, n).contiguous()
    B_flat = B.reshape(batch, n, nrhs).contiguous()

    lu, pivots, info = linalg_lu_factor_ex(A_flat)
    p, l, u = lu_unpack(lu, pivots)

    pt = p.transpose(-2, -1).contiguous()
    ptb = bmm(pt, B_flat)
    y = linalg_solve_triangular(l, ptb, upper=False, left=True, unitriangular=True)
    x = linalg_solve_triangular(u, y, upper=True, left=True)

    result = x.reshape(B.shape)
    info = info.reshape(batch_shape) if batch_shape else info.reshape(())

    if check_errors and torch.any(info != 0):
        raise torch.linalg.LinAlgError(
            "linalg.solve_ex: The diagonal element of the LU decomposition is zero."
        )

    return LinalgSolveExResult(result, info)
