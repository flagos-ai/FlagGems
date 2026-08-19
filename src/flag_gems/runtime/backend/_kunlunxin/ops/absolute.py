import logging

import triton
import triton.language as tl

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# absolute is an alias of abs; reuse abs's tuned recipe (memory async enabled).
_absolute_config = CodeGenConfig(
    max_tile_size=512,
    max_grid_size=(65536, 65536, 65536),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,  # Enable memory async for better overlap
)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")], config=_absolute_config)
@triton.jit
def absolute_func(x):
    return tl.abs(x)


def absolute(A):
    logger.debug("GEMS_KUNLUNXIN ABSOLUTE")
    return absolute_func(A)
