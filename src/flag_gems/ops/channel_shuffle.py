import logging

import torch

logger = logging.getLogger(__name__)


def channel_shuffle(inp: torch.Tensor, groups: int) -> torch.Tensor:
    logger.debug("GEMS CHANNEL_SHUFFLE")
    N, C = inp.shape[0], inp.shape[1]
    spatial = inp.shape[2:]
    out = inp.view(N, groups, C // groups, *spatial)
    out = out.transpose(1, 2).contiguous()
    return out.view(N, C, *spatial)


def channel_shuffle_out(
    inp: torch.Tensor, groups: int, *, out: torch.Tensor
) -> torch.Tensor:
    logger.debug("GEMS CHANNEL_SHUFFLE_OUT")
    result = channel_shuffle(inp, groups)
    out.copy_(result)
    return out
