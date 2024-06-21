import logging

import torch
import triton
import triton.language as tl
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._prims_common.wrappers import elementwise_type_promotion_wrapper

from ..utils import pointwise_dynamic


@pointwise_dynamic(output_dtypes=[torch.bool])
@triton.jit
def lt_func(x, y):
    return x.to(tl.float32) < y


@elementwise_type_promotion_wrapper(
    type_promoting_args=("A", "B"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)
def lt(A, B):
    logging.debug("GEMS LT")
    return lt_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], output_dtypes=[torch.bool])
@triton.jit
def lt_func_scalar(x, y):
    return x.to(tl.float32) < y


@elementwise_type_promotion_wrapper(
    type_promoting_args=("A", "B"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)
def lt_scalar(A, B):
    logging.debug("GEMS LT SCALAR")
    return lt_func_scalar(A, B)
