import logging

import triton
import triton.language as tl

from flag_gems.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic
from flag_gems.utils import tl_extra_shim

my_config = CodeGenConfig(
    max_tile_size= 65536,
    max_grid_size=(16, 16, 16),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
)

try:
    import torch_npu  # noqa: F401
except:  # noqa: E722
    _isnan = tl_extra_shim.isnan

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")], config=my_config)
@triton.jit
def isnan_func(x):
    return _isnan(x.to(tl.float32))


def isnan(A):
    logger.debug("GEMS ISNAN")
    return isnan_func(A)
