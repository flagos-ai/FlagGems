import logging

import torch

from flag_gems.ops.scatter_reduce_ import _get_init_value, scatter_reduce_
from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

logger = logging.getLogger(__name__)


def scatter_reduce(self, dim: int, index: torch.Tensor, src: torch.Tensor,
                   reduce: str, *, include_self: bool = True) -> torch.Tensor:
    logger.debug("GEMS SCATTER_REDUCE")
    assert reduce in ("sum", "prod", "mean", "amax", "amin"), \
        f"Invalid reduce op '{reduce}'. Expected one of: sum, prod, mean, amax, amin"
    out = self.clone()
    scatter_reduce_(out, dim, index, src, reduce, include_self=include_self)
    return out


def scatter_reduce_out(self, dim: int, index: torch.Tensor, src: torch.Tensor,
                       reduce: str, out: torch.Tensor, *,
                       include_self: bool = True) -> torch.Tensor:
    logger.debug("GEMS SCATTER_REDUCE_OUT")
    assert reduce in ("sum", "prod", "mean", "amax", "amin"), \
        f"Invalid reduce op '{reduce}'. Expected one of: sum, prod, mean, amax, amin"
    assert (
        has_internal_overlapping(out) != MemOverlap.Yes
    ), "Unsupported: writing to an internally overlapping tensor."
    out.copy_(self)
    scatter_reduce_(out, dim, index, src, reduce, include_self=include_self)
    return out
