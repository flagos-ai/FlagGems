import logging
from typing import List, Tuple, Union

import torch

from .cat import cat

logger = logging.getLogger(__name__)


def concatenate(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS CONCATENATE GCU400")
    return cat(A, dim=dim)
