import logging

import torch

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def roll(self, shifts, dims=None):
    logger.debug("GEMS ROLL")
    return torch.ops.aten.roll.default.redispatch(
        _FALLBACK_KEYSET, self, shifts, dims
    )
