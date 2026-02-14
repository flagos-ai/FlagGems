import logging

import torch

logger = logging.getLogger(__name__)


def _median_k(size: int) -> int:
    # kthvalue is 1-indexed, median uses lower middle for even sizes
    return (size + 1) // 2


def median_dim(self: torch.Tensor, dim: int, keepdim: bool = False):
    logger.debug("GEMS MEDIAN DIM")
    dim = dim % self.dim()
    k = _median_k(self.size(dim))
    return torch.kthvalue(self, k, dim=dim, keepdim=keepdim)


def median_dim_values(
    self: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    *,
    values: torch.Tensor,
    indices: torch.Tensor,
):
    logger.debug("GEMS MEDIAN DIM VALUES")
    out = median_dim(self, dim, keepdim)
    values.copy_(out.values)
    indices.copy_(out.indices)
    return values, indices
