import logging

from flag_gems.ops.polygamma import polygamma

logger = logging.getLogger(__name__)


def polygamma_(A, n):
    # Ascend override: the generic in-place path can write through
    # pointwise_dynamic with out0=A (aliased input and output), which leaves a
    # fraction of the elements unwritten on large tensors on this backend.
    # Delegate to the out-of-place functional op -- which always writes a fresh
    # buffer, including the raw-kernel fast paths for n = 0 and n = 1 -- then
    # copy back, matching the Ascend pow_ override's pattern.
    logger.debug("GEMS_ASCEND POLYGAMMA_")
    out = polygamma(n, A)
    A.copy_(out)
    return A
