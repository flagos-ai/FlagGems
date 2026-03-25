import logging

import torch

logger = logging.getLogger(__name__)


def conj(A: torch.Tensor):
    logger.debug("GEMS CONJ")
    if not A.is_complex():
        return A
    return torch.complex(A.real, -A.imag)
