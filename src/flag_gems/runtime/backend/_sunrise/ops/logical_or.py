import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.pointwise_dynamic import CodeGenConfig

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

MAX_GRID_SIZES = (65535, 65535, 65535)
config = CodeGenConfig(
    max_tile_size=1024,
    max_grid_size=MAX_GRID_SIZES,
    max_num_warps_per_cta=8,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")], config=config)
@triton.jit
def logical_or_func(x, y):
    return x.to(tl.int1).logical_or(y.to(tl.int1))


def logical_or(A, B):
    logger.debug("GEMS LOGICAL_OR")
    return logical_or_func(A, B)


def logical_or_(A, B):
    logger.debug("GEMS LOGICAL_OR_")
    logical_or_func(A, B, out0=A)
    return A
