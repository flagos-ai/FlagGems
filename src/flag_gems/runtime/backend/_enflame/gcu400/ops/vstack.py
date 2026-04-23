import logging
from typing import List, Tuple, Union

import torch

from flag_gems.runtime.backend._enflame.gcu400.ops.cat import cat

logger = logging.getLogger(__name__)


def vstack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]
) -> torch.Tensor:
    logger.debug("GEMS GCU400 VSTACK")

    n = len(tensors)
    if n == 0:
        raise RuntimeError("vstack expected a non-empty TensorList")

    t0 = tensors[0]
    if t0.ndim < 2:
        aligned = list(torch.atleast_2d(tensors))
    else:
        aligned = list(tensors) if not isinstance(tensors, list) else tensors

    return cat(aligned, dim=0)
