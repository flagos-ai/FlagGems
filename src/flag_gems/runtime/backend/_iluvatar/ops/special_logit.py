import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)

log = tl_extra_shim.log
abs = tl_extra_shim.abs


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def logit_kernel(x, eps):
    # clamp x to [eps, 1-eps]
    lo = eps
    hi = 1.0 - eps
    x_clamped = tl.minimum(tl.maximum(x, lo), hi)
    # logit = log(x / (1 - x))
    return log(x_clamped / (1.0 - x_clamped))


def logit(input: torch.Tensor, eps=None):
    logger.debug("GEMS_ILUVATAR LOGIT")
    if eps is None:
        eps = 0.0
    else:
        eps = float(eps)
        if not (0.0 <= eps <= 0.5):
            raise ValueError("eps must be in the range [0.0, 0.5].")

    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if not input.is_floating_point():
        raise TypeError("logit expected a floating point tensor as input")

    if eps > 0.0:
        return logit_kernel(input, eps)
    else:
        # No clamp needed when eps is 0
        # Use eps=0.0 to make the kernel work
        return logit_kernel(input, 0.0)
