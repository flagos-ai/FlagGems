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


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config)
@triton.jit
def cos_func(x):
    return tl.cos(x.to(tl.float32))


def cos(A):
    logger.debug("GEMS COS")
    return cos_func(A)


def cos_(A):
    logger.debug("GEMS COS_")
    cos_func(A, out0=A)
    return A
