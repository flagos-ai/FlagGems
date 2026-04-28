import logging

import torch

logger = logging.getLogger(__name__)

# Supported reduce operations
_REDUCE_OPS = {"sum", "prod", "mean", "amax", "amin"}


def scatter_reduce(
    input: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    reduce: str,
    include_self: bool = True,
) -> torch.Tensor:
    """Scatter-reduce: reduce src values into input at positions given by index.

    Matches torch.Tensor.scatter_reduce_ / torch.scatter_reduce semantics.

    Args:
        input: Self tensor (base values).
        dim: Dimension along which to scatter.
        index: LongTensor of indices.
        src: Source tensor to scatter.
        reduce: Reduction op: 'sum', 'prod', 'mean', 'amax', 'amin'.
        include_self: Whether to include self values in the reduction.

    Returns:
        Result tensor (same shape as input).
    """
    logger.debug("GEMS SCATTER_REDUCE")
    assert reduce in _REDUCE_OPS, f"Unsupported reduce op: {reduce}. Choose from {_REDUCE_OPS}"

    out = input.clone()
    out.scatter_reduce_(dim, index, src, reduce=reduce, include_self=include_self)
    return out
