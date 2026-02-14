import logging

import torch

logger = logging.getLogger(__name__)

def _median_k(size: int) -> int:
    # median uses lower middle for even sizes
    return (size + 1) // 2


def _ordered_key_fp16(x: torch.Tensor) -> torch.Tensor:
    bits = x.view(torch.uint16)
    zero_mask = x == 0
    if zero_mask.any():
        bits = torch.where(zero_mask, torch.zeros_like(bits), bits)
    sign = bits >> 15
    mask = torch.where(
        sign == 1,
        torch.full_like(bits, 0xFFFF),
        torch.full_like(bits, 0x8000),
    )
    ordered = bits ^ mask
    return ordered.to(torch.int64)


def median_dim(self: torch.Tensor, dim: int, keepdim: bool = False):
    logger.debug("GEMS MEDIAN DIM")
    dim = dim % self.dim()
    k = _median_k(self.size(dim)) - 1
    if self.dtype in (torch.float16, torch.bfloat16):
        ordered = _ordered_key_fp16(self)
        size = self.size(dim)
        index_shape = [1] * self.dim()
        index_shape[dim] = size
        base_index = torch.arange(
            size, device=self.device, dtype=torch.int64
        ).view(index_shape)
        key = ordered * size + base_index
        sorted_idx = torch.argsort(key, dim=dim)
    else:
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
