import logging

import triton
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def bitwise_or_func(x, y):
    return x | y


def bitwise_or_tensor(A, B):
    logger.debug("GEMS BITWISE OR")
    return bitwise_or_func(A, B)


def bitwise_or_tensor_(A, B):
    logger.debug("GEMS BITWISE OR_")
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
