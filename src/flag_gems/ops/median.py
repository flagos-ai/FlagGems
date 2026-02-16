import logging

import torch

logger = logging.getLogger(__name__)

def _median_k(size: int) -> int:
    # median uses lower middle for even sizes
    return (size + 1) // 2


def median_dim(self: torch.Tensor, dim: int, keepdim: bool = False):
    logger.debug("GEMS MEDIAN DIM")
    dim = dim % self.dim()
    if self.dtype in (torch.float16, torch.bfloat16):
        k = _median_k(self.size(dim))
        values = torch.kthvalue(self, k, dim=dim, keepdim=True).values
        size = self.size(dim)
        index_shape = [1] * self.dim()
        index_shape[dim] = size
        base_index = torch.arange(
            size, device=self.device, dtype=torch.long
        ).view(index_shape)
        base_index = base_index.expand_as(self)
        if self.dtype.is_floating_point:
            mask = (self == values) | (torch.isnan(self) & torch.isnan(values))
        else:
            mask = self == values
        sentinel = torch.full_like(base_index, -1)
        masked_index = torch.where(mask, base_index, sentinel)
        indices = torch.max(masked_index, dim=dim, keepdim=True).values
        values = torch.take_along_dim(self, indices, dim=dim)
    else:
        k = _median_k(self.size(dim)) - 1
        sorted_idx = torch.argsort(self, dim=dim, stable=True)
        sorted_vals = torch.take_along_dim(self, sorted_idx, dim=dim)
        index_shape = list(sorted_idx.shape)
        index_shape[dim] = 1
        gather_index = torch.full(
            index_shape, k, device=sorted_idx.device, dtype=sorted_idx.dtype
        )
        values = torch.take_along_dim(sorted_vals, gather_index, dim=dim)
        indices = torch.take_along_dim(sorted_idx, gather_index, dim=dim)
    if not keepdim:
        values = values.squeeze(dim)
        indices = indices.squeeze(dim)
    return values, indices


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
