import logging

import triton
import triton.language as tl

from flag_gems.runtime import device
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig

my_config = CodeGenConfig(
    max_tile_size= 65536,
    max_grid_size=(16, 16, 16),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
)

logger = logging.getLogger(__name__)
device = device.name


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")], config=my_config)
@triton.jit
def maximum_kernel(X, Y):
    if X.dtype == tl.bfloat16:
        X = X.to(tl.float32)
        Y = Y.to(tl.float32)

    return tl.maximum(X, Y)


def maximum(X, Y):
    logger.debug("GEMS MAXIMUM")
    assert X.device.type == device and Y.device.type == device
    return maximum_kernel(X, Y)
