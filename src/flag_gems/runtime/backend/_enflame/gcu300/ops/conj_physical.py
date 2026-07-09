import logging

import torch

logger = logging.getLogger(__name__)


def conj_physical(A):
    logger.debug("GEMS CONJ_PHYSICAL")
    return A.clone()
