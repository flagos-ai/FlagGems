import logging
from typing import List, Tuple, Union

import torch

from flag_gems.runtime.backend._enflame.gcu400.ops.cat import cat

logger = logging.getLogger(__name__)


def hstack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]
) -> torch.Tensor:
    logger.debug("GEMS GCU400 HSTACK")

    n = len(tensors)
    if n == 0:
        raise RuntimeError("hstack expected a non-empty TensorList")

    t0 = tensors[0]
    if t0.ndim == 0:
        aligned = [t.view(1) if t.ndim == 0 else t for t in tensors]
        t0 = aligned[0]
    else:
        aligned = list(tensors) if not isinstance(tensors, list) else tensors

    dim = 0 if t0.ndim == 1 else 1
    return cat(aligned, dim=dim)
