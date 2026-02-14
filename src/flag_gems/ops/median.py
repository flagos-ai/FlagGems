import logging

import torch

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def _median_k(size: int) -> int:
    # median uses lower middle for even sizes
    return (size + 1) // 2


def median_dim(self: torch.Tensor, dim: int, keepdim: bool = False):
    logger.debug("GEMS MEDIAN DIM")
    dim = dim % self.dim()
    k = _median_k(self.size(dim))

    if self.dtype in (torch.float16, torch.bfloat16):
        return torch.kthvalue(self, k, dim=dim, keepdim=keepdim)

    k = k - 1
    sorted_vals, sorted_idx = torch.ops.aten.sort.stable.redispatch(
        _FALLBACK_KEYSET, self, stable=True, dim=dim, descending=False
    )
    index_shape = list(sorted_vals.shape)
    index_shape[dim] = 1
    gather_index = torch.full(
        index_shape, k, device=sorted_vals.device, dtype=torch.long
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
