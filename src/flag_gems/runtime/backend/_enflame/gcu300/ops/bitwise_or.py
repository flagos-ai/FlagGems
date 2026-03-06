import logging

import torch
import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_or_func(x, y):
    return x | y


def bitwise_or_tensor(A, B):
    logger.debug("GEMS BITWISE OR")
    if A.dtype == torch.int64:
        A = A.to(torch.int32)
    if B.dtype == torch.int64:
        B = B.to(torch.int32)
    return bitwise_or_func(A, B)


def bitwise_or_tensor_(A, B):
    logger.debug("GEMS BITWISE OR_")
    if A.dtype == torch.int64:
        A = A.to(torch.int32)
    if B.dtype == torch.int64:
        B = B.to(torch.int32)
    return bitwise_or_func(A, B, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_or_func_scalar(x, y):
    return x | y


def bitwise_or_scalar(A, B):
    logger.debug("GEMS BITWISE OR SCALAR")
    return bitwise_or_func_scalar(A, B)


def bitwise_or_scalar_(A, B):
    logger.debug("GEMS BITWISE OR_ SCALAR")
    return bitwise_or_func_scalar(A, B, out0=A)


def bitwise_or_scalar_tensor(A, B):
    logger.debug("GEMS BITWISE OR SCALAR TENSOR")
    return bitwise_or_func_scalar(B, A)
