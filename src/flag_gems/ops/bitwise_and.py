import logging

import triton
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._prims_common.wrappers import elementwise_type_promotion_wrapper

from ..utils import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def bitwise_and_func(x, y):
    return x & y


@elementwise_type_promotion_wrapper(
    type_promoting_args=("A", "B"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def bitwise_and_tensor(A, B):
    logging.debug("GEMS BITWISE AND")
    return bitwise_and_func(A, B)


@pointwise_dynamic(is_tensor=[True, False])
@triton.jit
def bitwise_and_func_scalar(x, y):
    return x & y


@elementwise_type_promotion_wrapper(
    type_promoting_args=("A", "B"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def bitwise_and_scalar(A, B):
    logging.debug("GEMS BITWISE AND SCALAR")
    return bitwise_and_func_scalar(A, B)


@elementwise_type_promotion_wrapper(
    type_promoting_args=("A", "B"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def bitwise_and_scalar_tensor(A, B):
    logging.debug("GEMS BITWISE AND SCALAR TENSOR")
    return bitwise_and_func_scalar(B, A)
