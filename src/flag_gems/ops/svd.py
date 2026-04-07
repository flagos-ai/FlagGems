import logging

import torch

logger = logging.getLogger(__name__)


# SVD (Singular Value Decomposition): A = U * diag(S) * V^T
# Returns (U, S, V) matching torch.svd convention.
# torch.svd returns V (not V^H), so we transpose linalg.svd's Vh.
def svd(A, some=True, compute_uv=True):
    logger.debug("GEMS SVD")
    U, S, Vh = torch.linalg.svd(A, full_matrices=not some)
    if compute_uv:
        # torch.svd returns V, not Vh
        V = Vh.mH
        return U, S, V
    else:
        empty = torch.empty(0, dtype=A.dtype, device=A.device)
        return empty, S, empty
