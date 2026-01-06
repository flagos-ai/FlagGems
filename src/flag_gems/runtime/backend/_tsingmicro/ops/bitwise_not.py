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


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=my_config)
@triton.jit
def bitwise_not_func(x):
    return ~x


def bitwise_not(A):
    logger.debug("GEMS BITWISE NOT")
    return bitwise_not_func(A)


def bitwise_not_(A):
    logger.debug("GEMS BITWISE NOT_")
    bitwise_not_func(A, out0=A)
    return A
