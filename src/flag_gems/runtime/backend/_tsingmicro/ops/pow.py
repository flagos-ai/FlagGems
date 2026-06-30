import logging
import os

import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

_pow = tl_extra_shim.pow
logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_tensor_tensor(A, exponent):
    logger.debug("GEMS_TSINGMICRO POW_TENSOR_TENSOR")
    original_precision_priority = os.environ.get("PRECISION_PRIORITY", None)
    os.environ["PRECISION_PRIORITY"] = "0"  # force to high-perf mode
    try:
        return pow_func(A, exponent)
    finally:
        if original_precision_priority is not None:
            os.environ["PRECISION_PRIORITY"] = original_precision_priority
        else:
            os.environ.pop("PRECISION_PRIORITY", None)


def pow_tensor_tensor_(A, exponent):
    logger.debug("GEMS_TSINGMICRO POW_TENSOR_TENSOR_")
    original_precision_priority = os.environ.get("PRECISION_PRIORITY", None)
    os.environ["PRECISION_PRIORITY"] = "0"  # force to high-perf mode
    try:
        return pow_func(A, exponent, out0=A)
    finally:
        if original_precision_priority is not None:
            os.environ["PRECISION_PRIORITY"] = original_precision_priority
        else:
            os.environ.pop("PRECISION_PRIORITY", None)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_tensor_scalar(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_tensor_scalar(A, exponent):
    logger.debug("GEMS_TSINGMICRO POW_TENSOR_SCALAR")
    original_precision_priority = os.environ.get("PRECISION_PRIORITY", None)
    os.environ["PRECISION_PRIORITY"] = "0"  # force to high-perf mode
    try:
        return pow_func_tensor_scalar(A, exponent)
    finally:
        if original_precision_priority is not None:
            os.environ["PRECISION_PRIORITY"] = original_precision_priority
        else:
            os.environ.pop("PRECISION_PRIORITY", None)


def pow_tensor_scalar_(A, exponent):
    logger.debug("GEMS_TSINGMICRO POW_TENSOR_SCALAR_")
    original_precision_priority = os.environ.get("PRECISION_PRIORITY", None)
    os.environ["PRECISION_PRIORITY"] = "0"  # force to high-perf mode
    try:
        return pow_func_tensor_scalar(A, exponent, out0=A)
    finally:
        if original_precision_priority is not None:
            os.environ["PRECISION_PRIORITY"] = original_precision_priority
        else:
            os.environ.pop("PRECISION_PRIORITY", None)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_scalar_tensor(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_scalar(A, exponent):
    logger.debug("GEMS_TSINGMICRO POW_SCALAR")
    original_precision_priority = os.environ.get("PRECISION_PRIORITY", None)
    # force set to high-perf mode
    os.environ["PRECISION_PRIORITY"] = "0"  # force to high-perf mode

    try:
        return pow_func_scalar_tensor(A, exponent)
    finally:
        # restore original precision priority
        if original_precision_priority is not None:
            os.environ["PRECISION_PRIORITY"] = original_precision_priority
        else:
            os.environ.pop("PRECISION_PRIORITY", None)
