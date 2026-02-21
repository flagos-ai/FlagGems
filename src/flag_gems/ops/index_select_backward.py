import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def index_select_backward(
    grad: torch.Tensor,
    self_sizes: List[int],
    dim: int,
    index: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS INDEX SELECT BACKWARD")
    result = torch.zeros(
        self_sizes,
        dtype=grad.dtype,
        device=grad.device,
        requires_grad=grad.requires_grad,
    )
    result.index_add_(dim, index, grad)
    return result
