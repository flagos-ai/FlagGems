import logging
from typing import Optional

import torch
import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def clone_func(src):
    return src


def clone(
    inp: torch.Tensor,
    memory_format: Optional[torch.memory_format] = torch.preserve_format,
) -> torch.Tensor:
    logger.debug("GEMS CLONE")
    if memory_format is None:
        memory_format = torch.preserve_format
    out = torch.empty_like(inp, memory_format=memory_format)
    overload = clone_func.instantiate(inp.ndim)
    overload(inp, out0=out)
    return out
