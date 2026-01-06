import logging
import torch

logger = logging.getLogger(__name__)

def channel_shuffle(self: torch.Tensor, groups: int) -> torch.Tensor:
    logger.debug("GEMS CHANNEL_SHUFFLE")
    if not isinstance(groups, int):
        groups = int(groups)
    if groups <= 0:
        raise ValueError("groups must be a positive integer")
    if self.dim() < 2:
        raise ValueError("input must have at least (N, C)")
    n = self.size(0)
    c = self.size(1)
    if c % groups != 0:
        raise ValueError("C must be divisible by groups")
    rest = self.shape[2:]
    x = self.view(n, groups, c // groups, *rest)
    x = x.transpose(1, 2).contiguous()
    return x.view(n, c, *rest)