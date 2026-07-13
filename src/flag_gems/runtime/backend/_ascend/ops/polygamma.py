import logging

from flag_gems.ops.polygamma import (
    _polygamma_zeta_args,
    digamma_func,
    polygamma_zeta_func,
    trigamma_func,
)

logger = logging.getLogger(__name__)


def polygamma_(A, n):
    # Ascend override: the generic in-place path aliases input and output
    # (out0=A) through pointwise_dynamic, which leaves a fraction of the
    # elements unwritten on large tensors on this backend. Compute
    # out-of-place and copy back instead, matching the Ascend pow_ override.
    logger.debug("GEMS_ASCEND POLYGAMMA_")
    if n < 0:
        raise RuntimeError("polygamma(n, x) does not support negative n.")
    if n == 0:
        out = digamma_func(A)
    elif n == 1:
        out = trigamma_func(A)
    else:
        s, scale = _polygamma_zeta_args(n, A.device)
        out = polygamma_zeta_func(A, s, scale)
    A.copy_(out)
    return A
