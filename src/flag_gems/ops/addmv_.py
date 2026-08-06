import logging

logger = logging.getLogger(__name__)

from flag_gems.ops.addmv import addmv  # noqa: E402


def addmv_(self, mat, vec, *, beta=1, alpha=1):
    logger.debug("GEMS ADDMV_")
    result = addmv(self, mat, vec, beta=beta, alpha=alpha)
    return self.copy_(result)
