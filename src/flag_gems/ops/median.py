import logging

import torch

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def median_dim(self: torch.Tensor, dim: int, keepdim: bool = False):
    logger.debug("GEMS MEDIAN DIM")
    return torch.ops.aten.median.dim.default.redispatch(
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
    out = torch.ops.aten.median.dim.default.redispatch(
        _FALLBACK_KEYSET, self, dim, keepdim
    )
    values.copy_(out.values)
    indices.copy_(out.indices)
    return values, indices
