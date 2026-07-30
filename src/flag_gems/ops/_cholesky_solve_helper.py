# Copyright 2026 FlagOS Contributors

import logging

import torch

logger = logging.getLogger(__name__)


def _cholesky_solve_helper(
    self: torch.Tensor, A: torch.Tensor, upper: bool
) -> torch.Tensor:
    """Solve a system using a Cholesky factor.

    The original generated kernel used device memory as temporary sequential
    solve storage.  Two triangular CUDA solves keep the same factorization
    semantics without relying on cross-program synchronization.
    """
    logger.debug("GEMS _CHOLESKY_SOLVE_HELPER")
    if upper:
        intermediate = torch.linalg.solve_triangular(
            A.mT, self, upper=False, left=True, unitriangular=False
        )
        return torch.linalg.solve_triangular(
            A, intermediate, upper=True, left=True, unitriangular=False
        )

    intermediate = torch.linalg.solve_triangular(
        A, self, upper=False, left=True, unitriangular=False
    )
    return torch.linalg.solve_triangular(
        A.mT, intermediate, upper=True, left=True, unitriangular=False
    )


def _cholesky_solve_helper_out(
    self: torch.Tensor, A: torch.Tensor, upper: bool, *, out: torch.Tensor
) -> torch.Tensor:
    logger.debug("GEMS _CHOLESKY_SOLVE_HELPER_OUT")
    result = _cholesky_solve_helper(self, A, upper)
    out.resize_as_(result)
    out.copy_(result)
    return out
