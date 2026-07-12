import logging

import torch

from flag_gems.ops.neg import neg_func

logger = logging.getLogger(__name__)


def resolve_neg(A: torch.Tensor):
    logger.debug("GEMS RESOLVE_NEG")
    return neg_func(A) if A.is_neg() else A
