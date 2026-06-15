import logging
from typing import List, Tuple, Union

import torch

from flag_gems.ops.cat import cat

logger = logging.getLogger(__name__)


def concatenate(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    """
    Concatenate tensors along a given dimension.

    This is an alias for torch.cat. The function signature matches
    aten::concatenate(Tensor[] tensors, int dim=0) -> Tensor
    """
    logger.debug("GEMS CONCATENATE")
    return cat(A, dim=dim)
