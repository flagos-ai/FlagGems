import logging

import torch

from flag_gems.ops.neg import neg_func

logger = logging.getLogger("flag_gems." + __name__)


def resolve_conj(A: torch.Tensor):
    logger.debug("METAX GEMS RESOLVE_CONJ")
    return torch.complex(A.real, neg_func(A.imag)) if A.is_conj() else A
