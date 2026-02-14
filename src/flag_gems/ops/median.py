import logging

import torch

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def median_dim(self: torch.Tensor, dim: int, keepdim: bool = False):
    logger.debug("GEMS MEDIAN DIM")
    # Use _DisableTorchDispatch to avoid recursion when calling redispatch
    with torch._C._DisableTorchDispatch():
        return torch.ops.aten.median.dim.redispatch(
            _FALLBACK_KEYSET, self, dim, keepdim
        )


def median_dim_values(
    self: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    *,
    values: torch.Tensor,
    indices: torch.Tensor,
):
    logger.debug("GEMS MEDIAN DIM VALUES")
    # Use _DisableTorchDispatch to avoid recursion when calling redispatch
    with torch._C._DisableTorchDispatch():
        out = torch.ops.aten.median.dim.redispatch(
            _FALLBACK_KEYSET, self, dim, keepdim
        )
    values.copy_(out.values)
    indices.copy_(out.indices)
    return values, indices
