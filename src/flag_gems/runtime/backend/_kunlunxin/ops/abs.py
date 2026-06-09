import logging

import triton
import triton.language as tl

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

# Custom config with memory async enabled for better memory throughput
_abs_config = CodeGenConfig(
    max_tile_size=512,
    max_grid_size=(65536, 65536, 65536),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,  # Enable memory async for better overlap
)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")], config=_abs_config)
@triton.jit
def abs_func(x):
    return tl.abs(x)


def abs(A):
    logger.debug("GEMS_KUNLUNXIN ABS")
    return abs_func(A)


def abs_(A):
    logger.debug("GEMS_KUNLUNXIN ABS_")
    abs_func(A, out0=A)
    return A
