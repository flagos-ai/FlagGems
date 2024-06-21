import logging

import triton
import triton.language as tl
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._prims_common.wrappers import elementwise_type_promotion_wrapper

from ..utils import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def exp_func(x):
    return tl.exp(x.to(tl.float32))


@elementwise_type_promotion_wrapper(
    type_promoting_args=("A"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def exp(A):
    logging.debug("GEMS EXP")
    return exp_func(A)
