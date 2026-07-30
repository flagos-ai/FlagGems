import logging

import torch

from flag_gems.ops.linalg_svdvals import linalg_svdvals as _generic_linalg_svdvals

logger = logging.getLogger("flag_gems." + __name__)


def linalg_svdvals(A: torch.Tensor, driver: str = None) -> torch.Tensor:
    """Computes the singular values of a matrix.

    This is a Metax backend specialization that delegates to the generic kernel.

    Args:
        A: Input tensor of shape (*, m, n) where * is zero or more batch dimensions.
        driver: Optional cuSOLVER method (forwarded to generic kernel, not used on Metax).

    Returns:
        Singular values in descending order, shape (*, min(m, n)).
    """
    logger.debug("GEMS_METAX LINALG_SVDVALS")
    return _generic_linalg_svdvals(A, driver=driver)
