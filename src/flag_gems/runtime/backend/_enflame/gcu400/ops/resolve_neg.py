import logging

import torch

from .neg import neg

logger = logging.getLogger(__name__)


def resolve_neg(A: torch.Tensor):
    logger.debug("GEMS RESOLVE_NEG")
    return neg(A) if A.is_neg() else A
