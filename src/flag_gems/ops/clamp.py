import triton
import triton.language as tl
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def clamp_func_tensor(x, mini, maxi):
    return tl.minimum(maxi, tl.maximum(mini, x.to(tl.float32)))


@pointwise_dynamic
@triton.jit
def clamp_func_min_tensor(x, mini):
    return tl.maximum(mini, x.to(tl.float32))


@pointwise_dynamic
@triton.jit
def clamp_func_max_tensor(x, maxi):
    return tl.minimum(maxi, x.to(tl.float32))


def clamp_tensor(A, mini=None, maxi=None):
    logging.debug("GEMS CLAMP TENSOR")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        O = clamp_func_max_tensor(A, maxi)
    elif maxi is None:
        O = clamp_func_min_tensor(A, mini)
    else:
        O = clamp_func_tensor(A, mini, maxi)
    return O


@pointwise_dynamic(is_tensor=[True, False, False])
@triton.jit
def clamp_func(x, mini, maxi):
    return tl.minimum(maxi, tl.maximum(mini, x.to(tl.float32)))


@pointwise_dynamic(is_tensor=[True, False])
@triton.jit
def clamp_func_min(x, mini):
    return tl.maximum(mini, x.to(tl.float32))


@pointwise_dynamic(is_tensor=[True, False])
@triton.jit
def clamp_func_max(x, maxi):
    return tl.minimum(maxi, x.to(tl.float32))


def clamp(A, mini=None, maxi=None):
    logging.debug("GEMS CLAMP")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        O = clamp_func_max(A, maxi)
    elif maxi is None:
        O = clamp_func_min(A, mini)
    else:
        O = clamp_func(A, mini, maxi)
    return O
