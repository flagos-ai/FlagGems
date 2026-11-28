import logging

from .addmv import _launch_addmv, addmv

logger = logging.getLogger(__name__)


def addmv_(self, mat, vec, *, beta=1, alpha=1):
    logger.debug("GEMS_ENFLAME ADDMV_")
    # When self maps one-to-one elementwise with the output, write the kernel directly into self, saving addmv's
    # temporary out and copy_.
    # In-place writeback is safe: the value written, out_block, depends on the loaded bias, so the compiler cannot hoist
    # the store
    # ahead of the bias load; the row blocks are disjoint, and there is no cross-row dependency between programs.
    if self.shape == (mat.shape[0],) and self.stride(0) == 1:
        return _launch_addmv(self, mat, vec, self, beta, alpha)
    result = addmv(self, mat, vec, beta=beta, alpha=alpha)
    return self.copy_(result)
