import logging

import torch

logger = logging.getLogger(__name__)


def gcd(A, B):
    """Element-wise greatest common divisor using iterative Euclidean algorithm.

    Args:
        A: Integer tensor.
        B: Integer tensor (broadcastable with A).

    Returns:
        Tensor of GCD values with the same shape as the broadcasted inputs.
    """
    logger.debug("GEMS GCD")
    A, B = torch.broadcast_tensors(A, B)
    a = A.abs().clone()
    b = B.abs().clone()
    for _ in range(64):
        mask = b != 0
        new_a = torch.where(mask, b, a)
        new_b = torch.where(mask, a % torch.clamp(b, min=1), torch.zeros_like(b))
        a = new_a
        b = new_b
    return a
