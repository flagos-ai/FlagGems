import logging
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


def nansum(
    inp: torch.Tensor,
    dim: Optional[List[int]] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    logger.debug("GEMS NANSUM")
    inp = torch.nan_to_num(inp, nan=0.0)
    if dim is None:
        return torch.sum(inp, dtype=dtype)
    return torch.sum(inp, dim=dim, keepdim=keepdim, dtype=dtype)


def nansum_out(
    inp: torch.Tensor,
    dim: Optional[List[int]] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
    out: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS NANSUM_OUT")
    result = nansum(inp, dim, keepdim, dtype=dtype)
    out.copy_(result)
    return out
