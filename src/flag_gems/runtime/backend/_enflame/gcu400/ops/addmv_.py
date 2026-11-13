import logging

from .addmv import addmv

logger = logging.getLogger(__name__)


def addmv_(self, mat, vec, *, beta=1, alpha=1):
    logger.debug("GEMS ADDMV_")
    result = addmv(self, mat, vec, beta=beta, alpha=alpha)
    return self.copy_(result)
