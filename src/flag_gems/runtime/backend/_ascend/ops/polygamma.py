import logging

from flag_gems.ops.polygamma import polygamma

logger = logging.getLogger(__name__)


def polygamma_(A, n):
    # Ascend override: for n >= 2 the generic in-place path writes through
    # pointwise_dynamic with out0 aliasing the input, which leaves a fraction of
    # the elements unwritten on large tensors on this backend. Compute
    # out-of-place -- the functional op needs no override here -- and copy back,
    # matching the Ascend pow_ override's pattern.
    logger.debug("GEMS_ASCEND POLYGAMMA_")
    A.copy_(polygamma(n, A))
    return A
