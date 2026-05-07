import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.pointwise_dynamic import CodeGenConfig

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

MAX_GRID_SIZES = (65535, 65535, 65535)
config = CodeGenConfig(
    max_tile_size=512,
    max_grid_size=MAX_GRID_SIZES,
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=True,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")], config=config)
@triton.jit
def ge_func(x, y):
    return x.to(tl.float32) >= y


def ge(A, B):
    logger.debug("GEMS GE")
    return ge_func(A, B)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")], config=config
)
@triton.jit
def ge_func_scalar(x, y):
    return x.to(tl.float32) >= y


def ge_scalar(A, B):
    logger.debug("GEMS GE SCALAR")
    return ge_func_scalar(A, B)
