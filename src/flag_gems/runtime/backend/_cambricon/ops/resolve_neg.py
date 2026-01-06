import logging

import torch

from .neg import neg_func

logger = logging.getLogger(__name__)


def resolve_neg(A: torch.Tensor):
    logger.debug("GEMS_CAMBRICON RESOLVE_NEG")
    return neg_func(A) if A.is_neg() else A
