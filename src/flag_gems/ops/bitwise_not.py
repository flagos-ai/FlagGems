import logging

import triton
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._prims_common.wrappers import elementwise_type_promotion_wrapper

from ..utils import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def bitwise_not_func(x):
    return ~x


@elementwise_type_promotion_wrapper(
    type_promoting_args=("A"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def bitwise_not(A):
    logging.debug("GEMS BITWISE NOT")
    return bitwise_not_func(A)
