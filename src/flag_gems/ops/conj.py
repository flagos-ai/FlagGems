import logging

import torch

from flag_gems.ops.resolve_conj import resolve_conj_triton

logger = logging.getLogger(__name__)


def conj(A: torch.Tensor):
    logger.debug("GEMS CONJ")
    if not A.is_complex():
        return A
    if A.dtype == torch.complex64:
        return resolve_conj_triton(A, is_conj=True)
    return torch.complex(A.real, A.imag.neg())
