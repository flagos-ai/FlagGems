import logging

import triton
import triton.language as tl
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._prims_common.wrappers import elementwise_type_promotion_wrapper

from ..utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, True, True])
@triton.jit
def where_self_func(self, condition, other):
    return tl.where(condition, self, other)


@elementwise_type_promotion_wrapper(
    type_promoting_args=("self", "other"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
)
def where_self(condition, self, other):
    logging.debug("GEMS WHERE_SELF")
    return where_self_func(self, condition, other)


@pointwise_dynamic(is_tensor=[True, True, False])
@triton.jit
def where_scalar_self_func(other, condition, self):
    return tl.where(condition, self, other)


@elementwise_type_promotion_wrapper(
    type_promoting_args=("self", "other"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
)
def where_scalar_self(condition, self, other):
    logging.debug("GEMS WHERE_SCALAR_SELF")
    return where_scalar_self_func(other, condition, self)


@pointwise_dynamic(is_tensor=[True, True, False])
@triton.jit
def where_scalar_other_func(self, condition, other):
    return tl.where(condition, self, other)


@elementwise_type_promotion_wrapper(
    type_promoting_args=("self", "other"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
)
def where_scalar_other(condition, self, other):
    logging.debug("GEMS WHERE_SCALAR_OTHER")
    return where_scalar_other_func(self, condition, other)
