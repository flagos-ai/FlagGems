import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def relu6_kernel(x):
    return tl.minimum(tl.maximum(x, 0), 6)


def relu6(*args, **kwargs):
    logger.debug("GEMS RELU6")
    x = (
        args[0]
        if len(args) > 0
        else kwargs.get("input", kwargs.get("self", kwargs.get("x")))
    )
    if x is None:
        raise TypeError(
            "relu6 expects a tensor as the first positional argument or keyword 'input'/'self'/'x'."
        )
    return relu6_kernel(x)
