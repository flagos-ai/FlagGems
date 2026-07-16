import logging

import torch

logger = logging.getLogger(__name__)


def conj(input: torch.Tensor) -> torch.Tensor:
    """
    Ascend backend implementation for conj.
    Returns a view of the input tensor with the conjugate flag set.
    For real tensors, returns the input itself.
    For complex tensors, returns a view sharing the same underlying storage.
    """
    logger.debug("GEMS CONJ")
    if not input.is_complex():
        return input
    return input._conj()
