import logging

import torch

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def gcd(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS GCD")
    return torch.ops.aten.gcd.default.redispatch(_FALLBACK_KEYSET, self, other)
