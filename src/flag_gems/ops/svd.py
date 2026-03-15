import logging

import torch

logger = logging.getLogger(__name__)


def svd(A, some=True, compute_uv=True):
    """Singular Value Decomposition.

    Computes the SVD of a matrix (or batch of matrices) A such that
    A = U * diag(S) * V^T.

    This wraps torch.linalg.svd which dispatches to cuSOLVER on GPU
    (or LAPACK on CPU), providing a FlagGems-compatible interface that
    matches the torch.svd convention of returning (U, S, V) rather than
    (U, S, Vh).

    Args:
        A: Input tensor of shape (..., m, n).
        some: If True (default), compute the reduced SVD (economy-size).
            If False, compute the full SVD.
        compute_uv: If True (default), compute U and V in addition to S.
            If False, only singular values are computed.

    Returns:
        A tuple (U, S, V) where:
            - U: Left singular vectors. Shape (..., m, k) if some=True,
              (..., m, m) if some=False, or (..., m, 0) if compute_uv=False.
            - S: Singular values. Shape (..., min(m, n)).
            - V: Right singular vectors (not conjugate-transposed).
              Shape (..., n, k) if some=True, (..., n, n) if some=False,
              or (..., n, 0) if compute_uv=False.
              k = min(m, n).
    """
    logger.debug("GEMS SVD")

    if A.ndim < 2:
        raise RuntimeError(
            f"svd: expected a tensor with at least 2 dimensions, "
            f"but got a {A.ndim}D tensor"
        )

    full_matrices = not some

    if compute_uv:
        U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
        # torch.linalg.svd returns Vh (conjugate transpose of V).
        # torch.svd convention returns V, so transpose back.
        V = Vh.mH
        return U, S, V

    # When compute_uv is False, only compute singular values.
    S = torch.linalg.svdvals(A)

    # Return empty U and V tensors with shapes matching the convention.
    m = A.shape[-2]
    n = A.shape[-1]
    batch_shape = A.shape[:-2]
    U = A.new_empty((*batch_shape, m, 0))
    V = A.new_empty((*batch_shape, n, 0))

    return U, S, V
