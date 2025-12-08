import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(
    is_tensor=[True, True, True, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def clamp_func_tensor(x, mini, maxi, inplace):
    return tl.minimum(maxi, tl.maximum(mini, x.to(tl.float32)))


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clamp_func_min_tensor(x, mini, inplace):
    return tl.maximum(mini, x.to(tl.float32))


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clamp_func_max_tensor(x, maxi, inplace):
    return tl.minimum(maxi, x.to(tl.float32))


def clamp_tensor(A, mini=None, maxi=None):
    logger.debug("GEMS_CAMBRICON CLAMP TENSOR")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clamp_func_max_tensor(A, maxi, False)
    elif maxi is None:
        return clamp_func_min_tensor(A, mini, False)
    else:
        return clamp_func_tensor(A, mini, maxi, False)


def clamp_tensor_(A, mini=None, maxi=None):
    logger.debug("GEMS_CAMBRICON CLAMP_ TENSOR")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clamp_func_max_tensor(A, maxi, True, out0=A)
    elif maxi is None:
        return clamp_func_min_tensor(A, mini, True, out0=A)
    else:
        return clamp_func_tensor(A, mini, maxi, True, out0=A)


@pointwise_dynamic(
    is_tensor=[True, False, False, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def clamp_func(x, mini, maxi, inplace):
    return tl.minimum(maxi, tl.maximum(mini, x.to(tl.float32)))


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def clamp_func_min(x, mini, inplace):
    return tl.maximum(mini, x.to(tl.float32))


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def clamp_func_max(x, maxi, inplace):
    return tl.minimum(maxi, x.to(tl.float32))


def clamp_min(A, mini):
    logger.debug("GEMS_CAMBRICON CLAMP MIN")
    if mini is None:
        raise ValueError("Mini must not be None")
    return clamp_func_min(A, mini, False)


def clamp_min_(A, mini):
    logger.debug("GEMS_CAMBRICON CLAMP_ MIN")
    if mini is None:
        raise ValueError("Mini must not be None")
    return clamp_func_min(A, mini, True, out0=A)


def clamp(A, mini=None, maxi=None):
    logger.debug("GEMS_CAMBRICON CLAMP")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clamp_func_max(A, maxi, False)
    elif maxi is None:
        return clamp_func_min(A, mini, False)
    else:
        return clamp_func(A, mini, maxi, False)


def clamp_(A, mini=None, maxi=None):
    logger.debug("GEMS_CAMBRICON CLAMP_")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clamp_func_max(A, maxi, True, out0=A)
    elif maxi is None:
        return clamp_func_min(A, mini, True, out0=A)
    else:
        return clamp_func(A, mini, maxi, True, out0=A)
