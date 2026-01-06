import logging

import triton
import triton.language as tl

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


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=my_config)
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
