import logging

import torch

from ._conj import _conj

logger = logging.getLogger(__name__)


def conj(input: torch.Tensor) -> torch.Tensor:
    # aten::conj -> aten::_conj; delegate to the kunlunxin _conj (native path).
    # See _conj for why the generic triton kernel is bypassed on XPU.
    logger.debug("GEMS_KUNLUNXIN CONJ")
    return _conj(input)
