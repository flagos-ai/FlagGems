import logging

import torch

from flag_gems.ops.sort import sort

logger = logging.getLogger(__name__)


def msort(inp: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS MSORT")
    sorted_values, _ = sort(inp, dim=-1, descending=False)
    return sorted_values
