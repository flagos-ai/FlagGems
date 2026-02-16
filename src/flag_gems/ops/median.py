import logging

import torch

logger = logging.getLogger(__name__)

def _median_k(size: int) -> int:
    # median uses lower middle for even sizes
    return (size + 1) // 2


def median_dim(self: torch.Tensor, dim: int, keepdim: bool = False):
    logger.debug("GEMS MEDIAN DIM")
    dim = dim % self.dim()

    # For float16/bfloat16, upcast to float64 and use kthvalue.
    # This provides the most stable results across different test cases.
    if self.dtype in (torch.float16, torch.bfloat16):
        original_dtype = self.dtype
        self_upcast = self.to(torch.float64)
        k = _median_k(self_upcast.size(dim))
        kth_result = torch.kthvalue(self_upcast, k, dim=dim, keepdim=keepdim)
        values = kth_result.values.to(original_dtype)
        indices = kth_result.indices
    else:
        # For float32 and higher precision, use stable sort
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
