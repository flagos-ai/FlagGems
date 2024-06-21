import logging

import triton
import triton.language as tl
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._prims_common.wrappers import elementwise_type_promotion_wrapper

from ..utils import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def pow_func(x, exponent):
    return tl.math.pow(x.to(tl.float32), exponent)


@elementwise_type_promotion_wrapper(
    type_promoting_args=("A"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG,
)
def pow_tensor_tensor(A, exponent):
    logging.debug("GEMS POW_TENSOR_TENSOR")
    return pow_func(A, exponent)


@pointwise_dynamic(is_tensor=[True, False])
@triton.jit
def pow_func_tensor_scalar(x, exponent):
    return tl.math.pow(x.to(tl.float32), exponent)


@elementwise_type_promotion_wrapper(
    type_promoting_args=("A"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG,
)
def pow_tensor_scalar(A, exponent):
    logging.debug("GEMS POW_TENSOR_SCALAR")
    return pow_func_tensor_scalar(A, exponent)


@pointwise_dynamic(is_tensor=[False, True])
@triton.jit
def pow_func_scalar_tensor(x, exponent):
    return tl.math.pow(x.to(tl.float32), exponent)


@elementwise_type_promotion_wrapper(
    type_promoting_args=("A"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG,
)
def pow_scalar(A, exponent):
    logging.debug("GEMS POW_SCALAR")
    return pow_func_scalar_tensor(A, exponent)
