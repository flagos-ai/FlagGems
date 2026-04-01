import logging

import torch
import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def bitwise_not_func(x):
    return ~x


def bitwise_not(A):
    logger.debug("GEMS_ASCEND BITWISE NOT")
    if A.dtype == torch.bfloat16:
        return bitwise_not_func(A.view(torch.int16)).view(torch.bfloat16)
    return bitwise_not_func(A)


def bitwise_not_(A):
    logger.debug("GEMS_ASCEND BITWISE NOT_")
    if A.dtype == torch.bfloat16:
        bitwise_not_func(A.view(torch.int16), out0=A.view(torch.int16))
        return A
    bitwise_not_func(A, out0=A)
    return A
