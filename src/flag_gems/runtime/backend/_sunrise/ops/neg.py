import logging

import triton

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig

logger = logging.getLogger(__name__)

config = CodeGenConfig(
    max_tile_size=4096,
    max_grid_size=(65535, 65535, 65535),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
    # num_warps=8,
)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config)
@triton.jit
def neg_func(x):
    return -x


def neg(A):
    logger.debug("GEMS_SUNRISE NEG")
    return neg_func(A)


def neg_(A):
    logger.debug("GEMS_SUNRISE NEG_")
    return neg_func(A, out0=A)
