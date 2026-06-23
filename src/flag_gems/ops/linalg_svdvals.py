import logging

import torch

from flag_gems.ops.svd import svd

logger = logging.getLogger(__name__)


def linalg_svdvals(A: torch.Tensor, driver: str = None) -> torch.Tensor:
    """
    Compute the singular values of a matrix.

    The driver argument is part of the aten::linalg_svdvals schema. It selects
    cuSOLVER algorithms in PyTorch, but the native FlagGems implementation does
    not call cuSOLVER, so non-None driver values are not supported.
    """
    logger.debug("GEMS LINALG_SVDVALS")
    if driver is not None:
        raise NotImplementedError(
            "linalg_svdvals: driver is not supported by the FlagGems "
            "Triton implementation"
        )
    return svd(A, some=True, compute_uv=False).S
