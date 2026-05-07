import logging

import triton

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.pointwise_dynamic import CodeGenConfig

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


MAX_GRID_SIZES = (65535, 65535, 65535)
config = CodeGenConfig(
    max_tile_size=2048,
    max_grid_size=MAX_GRID_SIZES,
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=True,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config)
@triton.jit
def bitwise_and_func(x, y):
    return x & y


def bitwise_and_tensor(A, B):
    logger.debug("GEMS BITWISE AND")
    return bitwise_and_func(A, B)


def bitwise_and_tensor_(A, B):
    logger.debug("GEMS BITWISE AND_")
    return bitwise_and_func(A, B, out0=A)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config
)
@triton.jit
def bitwise_and_func_scalar(x, y):
    return x & y


def bitwise_and_scalar(A, B):
    logger.debug("GEMS BITWISE AND SCALAR")
    return bitwise_and_func_scalar(A, B)


def bitwise_and_scalar_(A, B):
    logger.debug("GEMS BITWISE AND_ SCALAR")
    return bitwise_and_func_scalar(A, B, out0=A)


def bitwise_and_scalar_tensor(A, B):
    logger.debug("GEMS BITWISE AND SCALAR TENSOR")
    return bitwise_and_func_scalar(B, A)
