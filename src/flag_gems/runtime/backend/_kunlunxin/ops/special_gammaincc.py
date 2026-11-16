import logging

import torch

from .igammac import igammac

logger = logging.getLogger(__name__)


def special_gammaincc(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Regularized upper incomplete gamma function Q(a, x) on Kunlunxin XPU."""
    logger.debug("GEMS_KUNLUNXIN SPECIAL_GAMMAINCC")
    return igammac(self, other)
