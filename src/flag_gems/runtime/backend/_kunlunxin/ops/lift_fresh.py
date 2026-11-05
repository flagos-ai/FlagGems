import logging

import torch

logger = logging.getLogger(__name__)


def lift_fresh(x: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN LIFT_FRESH")
    return x
