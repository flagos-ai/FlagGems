import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim
from flag_gems.utils.pointwise_dynamic import CodeGenConfig

_isnan = tl_extra_shim.isnan

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

MAX_GRID_SIZES = (65535, 65535, 65535)
config = CodeGenConfig(
    max_tile_size=1024,
    max_grid_size=MAX_GRID_SIZES,
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=True,
)


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")], config=config)
@triton.jit
def isnan_func(x):
    return _isnan(x.to(tl.float32))


def isnan(A):
    logger.debug("GEMS ISNAN")
    return isnan_func(A)
