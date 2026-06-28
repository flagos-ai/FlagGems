import logging

import torch
import triton
import triton.language as tl  # noqa: F401

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def square_func(x):
    return x * x


def square(A):
    logger.debug("GEMS_KUNLUNXIN SQUARE")
    if A.dtype == torch.bfloat16:
        out = square_func(A.float())
        return out.bfloat16()
    return square_func(A)


def square_out(A, *, out=None):
    logger.debug("GEMS_KUNLUNXIN SQUARE_OUT")
    if out is None:
        return square_func(A)
    square_func(A, out0=out)
    return out


def square_(A):
    logger.debug("GEMS_KUNLUNXIN SQUARE_")
    if A.dtype == torch.bfloat16:
        square_func(A.float(), out0=A)
        A = A.bfloat16()
        return A
    square_func(A, out0=A)
    return A
