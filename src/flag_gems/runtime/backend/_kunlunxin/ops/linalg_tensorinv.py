import logging

import torch

from .linalg_inv_ex import linalg_inv_ex

logger = logging.getLogger(__name__)


def check_inv_input(A, ind):
    """Validate input for tensorinv: ind is strictly positive, A is >=2D, and
    prod(A.shape[:ind]) == prod(A.shape[ind:]). Raises RuntimeError on
    violation, matching torch.linalg.tensorinv's behaviour.
    """
    if ind <= 0:
        raise RuntimeError(
            "linalg.tensorinv: Expected a strictly positive integer for "
            f"'ind', but got {ind}"
        )
    if A.dim() < 2:
        raise RuntimeError(
            "linalg.tensorinv: Expected input to be at least 2D, " f"got {A.dim()}D"
        )
    if ind > A.dim():
        raise RuntimeError(
            "linalg.tensorinv: Expected 0 <= ind <= input.dim(), "
            f"got ind={ind} for input.dim()={A.dim()}"
        )
    m = 1
    for i in range(ind):
        m *= A.shape[i]
    n = 1
    for i in range(ind, A.dim()):
        n *= A.shape[i]
    if m != n:
        raise RuntimeError(
            "linalg.tensorinv: Expected self to satisfy the requirement "
            "prod(self.shape[ind:]) == prod(self.shape[:ind]), "
            f"but got {n} != {m}"
        )


def linalg_tensorinv(A, ind=2, *, out=None):
    """Compute the multiplicative inverse of tensordot.

    Flattens the first ``ind`` dims into rows and the rest into cols to form an
    N x N matrix, inverts it via the vendor ``linalg_inv_ex`` (LU + triangular
    solves), then reshapes the inverse to ``A.shape[ind:] + A.shape[:ind]``.
    """
    logger.debug("GEMS_KUNLUNXIN LINALG_TENSORINV")
    check_inv_input(A, ind)

    matrix_size = 1
    for i in range(ind):
        matrix_size *= A.shape[i]
    output_shape = A.shape[ind:] + A.shape[:ind]
    n = matrix_size
    orig_dtype = A.dtype

    A_work = (
        A.contiguous()
        .to(torch.float32)
        .reshape(n, n)
        .clone(memory_format=torch.contiguous_format)
    )

    inverse, _info = linalg_inv_ex(A_work)

    result = inverse.reshape(output_shape).to(orig_dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


def linalg_tensorinv_out(A, ind=2, *, out=None):
    """Out-of-place variant of linalg_tensorinv: computes the tensor inverse
    and writes the result into the provided ``out`` tensor."""
    logger.debug("GEMS_KUNLUNXIN LINALG_TENSORINV_OUT")
    return linalg_tensorinv(A, ind=ind, out=out)
