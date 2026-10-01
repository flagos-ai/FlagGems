import logging
import math
from collections import namedtuple

import torch

from .linalg_lu_factor_ex import linalg_lu_factor_ex
from .linalg_solve_triangular import linalg_solve_triangular
from .lu_unpack import lu_unpack

logger = logging.getLogger(__name__)

LinalgInvExResult = namedtuple("LinalgInvExResult", ["inverse", "info"])


def linalg_inv_ex(A, *, check_errors=False):
    """Compute ``A^{-1}`` returning ``(inverse, info)`` via vendor LU + trsm."""
    logger.debug("GEMS_KUNLUNXIN LINALG_INV_EX")

    assert A.ndim >= 2, "Input must be at least 2D"
    n = A.shape[-1]
    assert A.shape[-2] == n, "Input must be a square matrix"
    assert A.dtype in (
        torch.float32,
        torch.float64,
    ), f"linalg_inv_ex: unsupported dtype {A.dtype}, requires float32 or float64"

    device = A.device
    dtype = A.dtype
    batch_shape = A.shape[:-2]

    if A.numel() == 0:
        inverse = A.clone()
        info = torch.zeros(batch_shape, dtype=torch.int32, device=device)
        return LinalgInvExResult(inverse, info)

    batch = math.prod(batch_shape) if batch_shape else 1
    A_flat = A.reshape(batch, n, n).contiguous()

    lu, pivots, info = linalg_lu_factor_ex(A_flat)
    p, l, u = lu_unpack(lu, pivots)

    ptb = p.transpose(-2, -1).contiguous()
    y = linalg_solve_triangular(l, ptb, upper=False, left=True, unitriangular=True)
    x = linalg_solve_triangular(u, y, upper=True, left=True)

    inverse = x.reshape(*batch_shape, n, n).to(dtype)
    info = info.reshape(list(batch_shape))

    if check_errors and torch.any(info != 0):
        raise torch.linalg.LinAlgError(
            "torch.linalg.inv_ex: The diagonal element of the LU "
            "decomposition is zero."
        )

    return LinalgInvExResult(inverse, info)
