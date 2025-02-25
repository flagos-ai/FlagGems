import logging

import torch

from .neg import neg_func


def resolve_neg(A: torch.Tensor):
    logging.debug("GEMS_CAMBRICON RESOLVE_NEG")
    return neg_func(A) if A.is_neg() else A
