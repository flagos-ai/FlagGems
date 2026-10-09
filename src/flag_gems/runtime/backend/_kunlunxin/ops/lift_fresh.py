import logging

import torch

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def lift_fresh(x: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN LIFT_FRESH")
    return x
