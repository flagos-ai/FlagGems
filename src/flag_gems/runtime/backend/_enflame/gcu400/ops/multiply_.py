import logging

import torch

from .mul import mul_

logger = logging.getLogger(__name__)


def multiply_(A, B):
    """In-place multiply (multiply_), an alias for the enflame gcu400 mul_.

    The generic ``flag_gems.ops.multiply_`` binds the *generic* ``mul_`` at
    import time, so it never reaches this backend's optimized ``mul_``. On GCU
    the generic path then tries ``aten::mul.out.redispatch`` with a
    CompositeExplicitAutograd keyset, which TopsRider torch cannot resolve
    (NotImplementedError: no fallback registered for aten::mul.out).
    Overriding ``multiply_`` here routes it to the gcu400 ``mul_`` instead.
    """
    logger.debug("GEMS_ENFLAME MULTIPLY_")
    if not isinstance(A, torch.Tensor):
        raise ValueError("Unreachable.")
    return mul_(A, B)
