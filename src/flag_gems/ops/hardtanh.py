import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, False, False], promotion_methods=[(0, 1, 2, "DEFAULT")])
@triton.jit
def hardtanh_func(x, min_val, max_val):
    xf = x.to(tl.float32)
    return tl.minimum(max_val, tl.maximum(min_val, xf))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def hardtanh_func_min(x, min_val):
    xf = x.to(tl.float32)
    return tl.maximum(min_val, xf)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def hardtanh_func_max(x, max_val):
    xf = x.to(tl.float32)
    return tl.minimum(max_val, xf.to(tl.float32))


@pointwise_dynamic(is_tensor=[True, False, False])
@triton.jit
def hardtanh_backward_func(x, grad_output, min_val, max_val):
    xf = x.to(tl.float32)
    mask = (xf >= min_val) & (xf <= max_val)
    return tl.where(mask, grad_output.to(tl.float32), 0.0)


def hardtanh(x, min_val=-1.0, max_val=1.0):
    logger.debug("GEMS HARDTANH")
    if min_val >= max_val:
        raise ValueError(f"min_val ({min_val}) must be less than max_val ({max_val})")
    return hardtanh_func(x, min_val, max_val)


def hardtanh_(x, min_val=-1.0, max_val=1.0):
    logger.debug("GEMS HARDTANH_")
    if min_val >= max_val:
        raise ValueError(f"min_val ({min_val}) must be less than max_val ({max_val})")
    return hardtanh_func(x, min_val, max_val, out0=x)


def hardtanh_backward(grad_output, x, min_val, max_val):
    logger.debug("GEMS HARDTANH_BACKWARD")
    return hardtanh_backward_func(x, grad_output, min_val, max_val)
