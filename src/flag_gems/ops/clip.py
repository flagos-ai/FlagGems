import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def clip_func(x, mini, maxi):
    return tl.minimum(maxi, tl.maximum(mini, x.to(tl.float32)))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clip_func_min(x, mini):
    return tl.maximum(mini, x.to(tl.float32))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clip_func_max(x, maxi):
    return tl.minimum(maxi, x.to(tl.float32))


def clip(A, mini=None, maxi=None):
    logger.debug("GEMS CLIP")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clip_func_max(A, maxi)
    elif maxi is None:
        return clip_func_min(A, mini)
    else:
        return clip_func(A, mini, maxi)


def clip_(A, mini=None, maxi=None):
    logger.debug("GEMS CLIP_")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clip_func_max(A, maxi, out0=A)
    elif maxi is None:
        return clip_func_min(A, mini, out0=A)
    else:
        return clip_func(A, mini, maxi, out0=A)
