import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, "DEFAULT")],
    config=config_,
)
@triton.jit
def _masked_scale_kernel(input, mask, scale):
    return tl.where(mask != 0, input * scale, 0.0)


def _masked_scale(input, mask, scale):
    logger.debug("GEMS_KUNLUNXIN _MASKED_SCALE")
    if not input.is_floating_point():
        raise ValueError(f"Only floating-point dtype is supported, got {input.dtype}")
    return _masked_scale_kernel(input, mask, scale)
