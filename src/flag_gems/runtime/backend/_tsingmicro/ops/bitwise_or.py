import logging

import triton

from flag_gems.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils.pointwise_dynamic import pointwise_dynamic

my_config = CodeGenConfig(
    max_tile_size= 65536,
    max_grid_size=(16, 16, 16),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
)

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=my_config)
@triton.jit
def bitwise_or_func(x, y):
    return x | y


def bitwise_or_tensor(A, B):
    logger.debug("GEMS BITWISE OR")
    return bitwise_or_func(A, B)


def bitwise_or_tensor_(A, B):
    logger.debug("GEMS BITWISE OR_")
    return bitwise_or_func(A, B, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")], config=my_config)
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
