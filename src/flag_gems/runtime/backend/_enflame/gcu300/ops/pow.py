import logging

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

try:
    from triton.language.extra.cuda.libdevice import pow as _pow
except ImportError:
    try:
        from triton.language.math import pow as _pow
    except ImportError:
        from triton.language.libdevice import pow as _pow


logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_tensor_tensor(A, exponent):
    logger.debug("GEMS POW_TENSOR_TENSOR")
    if exponent.dtype == torch.int64:
        exponent = exponent.to(torch.int32)
    if A.dtype == torch.int64:
        A = A.to(torch.int32)
    return pow_func(A, exponent)


def pow_tensor_tensor_(A, exponent):
    logger.debug("GEMS POW_TENSOR_TENSOR_")
    if exponent.dtype == torch.int64:
        exponent = exponent.to(torch.int32)
    if A.dtype == torch.int64:
        A = A.to(torch.int32)
    return pow_func(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_tensor_scalar(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_tensor_scalar(A, exponent):
    logger.debug("GEMS POW_TENSOR_SCALAR")
    return pow_func_tensor_scalar(A, exponent)


def pow_tensor_scalar_(A, exponent):
    logger.debug("GEMS POW_TENSOR_SCALAR_")
    return pow_func_tensor_scalar(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_scalar_tensor(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_scalar(A, exponent):
    logger.debug("GEMS POW_SCALAR")
    return pow_func_scalar_tensor(A, exponent)
