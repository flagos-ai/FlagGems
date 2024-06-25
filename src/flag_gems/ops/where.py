import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(
    is_tensor=[True, True, True], promotion_methods=[[0, 1, "NO_OPMATH"]]
)
@triton.jit
def where_self_func(condition, self, other):
    return tl.where(condition, self, other)


def where_self(condition, self, other):
    logging.debug("GEMS WHERE_SELF")
    return where_self_func(condition, self, other)


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[[0, 1, "NO_OPMATH"]]
)
@triton.jit
def where_scalar_self_func(condition, self, other):
    return tl.where(condition, self, other)


def where_scalar_self(condition, self, other):
    logging.debug("GEMS WHERE_SCALAR_SELF")
    return where_scalar_self_func(condition, self, other)


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[[0, 1, "NO_OPMATH"]]
)
@triton.jit
def where_scalar_other_func(condition, self, other):
    return tl.where(condition, self, other)


def where_scalar_other(condition, self, other):
    logging.debug("GEMS WHERE_SCALAR_OTHER")
    return where_scalar_other_func(condition, self, other)
