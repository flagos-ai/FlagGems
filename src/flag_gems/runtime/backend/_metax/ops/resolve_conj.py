import logging

import torch

from flag_gems.ops.neg import neg_func


def resolve_conj(A: torch.Tensor):
    logging.debug("METAX GEMS RESOLVE_CONJ")
    return torch.complex(A.real, neg_func(A.imag)) if A.is_conj() else A
