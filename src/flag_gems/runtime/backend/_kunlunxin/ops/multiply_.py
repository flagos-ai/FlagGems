import logging

import torch

from .mul import mul_

logger = logging.getLogger(__name__)


def multiply_(A, B):
    """In-place multiply (multiply_), an alias for the kunlunxin mul_.

    The generic `flag_gems.ops.multiply_` binds the *generic* `mul_` at import
    time, so it never reaches this backend's optimized `mul_` (contiguous
    pointwise kernel). Overriding `multiply_` here routes it to the fast
    kunlunxin `mul_`, matching `a.mul_(b)` (sp ~0.75-0.99 vs ~0.04 generic).
    """
    logger.debug("GEMS_KUNLUNXIN MULTIPLY_")
    if not isinstance(A, torch.Tensor):
        raise ValueError("Unreachable.")
    return mul_(A, B)
