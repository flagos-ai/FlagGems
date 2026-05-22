import logging

import triton

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

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
    logger.debug("GEMS NEG")
    return neg_func(A)


def neg_(A):
    logger.debug("GEMS NEG_")
    return neg_func(A, out0=A)
