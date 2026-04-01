import logging

import torch
import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_or_func(x, y):
    return x | y


def bitwise_or_tensor(A, B):
    logger.debug("GEMS_ASCEND BITWISE OR")
    if A.dtype == torch.bfloat16:
        return bitwise_or_func(
            A.view(torch.int16), B.view(torch.int16)
        ).view(torch.bfloat16)
    return bitwise_or_func(A, B)


def bitwise_or_tensor_(A, B):
    logger.debug("GEMS_ASCEND BITWISE OR_")
    if A.dtype == torch.bfloat16:
        bitwise_or_func(A.view(torch.int16), B.view(torch.int16), out0=A.view(torch.int16))
        return A
    return bitwise_or_func(A, B, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_or_func_scalar(x, y):
    return x | y


def bitwise_or_scalar(A, B):
    logger.debug("GEMS_ASCEND BITWISE OR SCALAR")
    if A.dtype == torch.bfloat16:
        return bitwise_or_func_scalar(
            A.view(torch.int16), int(B)
        ).view(torch.bfloat16)
    return bitwise_or_func_scalar(A, B)


def bitwise_or_scalar_(A, B):
    logger.debug("GEMS_ASCEND BITWISE OR_ SCALAR")
    if A.dtype == torch.bfloat16:
        bitwise_or_func_scalar(A.view(torch.int16), int(B), out0=A.view(torch.int16))
        return A
    return bitwise_or_func_scalar(A, B, out0=A)


def bitwise_or_scalar_tensor(A, B):
    logger.debug("GEMS_ASCEND BITWISE OR SCALAR TENSOR")
    if B.dtype == torch.bfloat16:
        return bitwise_or_func_scalar(
            B.view(torch.int16), int(A)
        ).view(torch.bfloat16)
    return bitwise_or_func_scalar(B, A)
