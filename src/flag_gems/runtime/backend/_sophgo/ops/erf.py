import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig, get_codegen_config

logger = logging.getLogger(__name__)

_base = get_codegen_config()
_erf_config = CodeGenConfig(
    max_tile_size=_base.max_tile_size,
    max_grid_size=(65536, 1, 1),
    max_num_warps_per_cta=_base.max_num_warps_per_cta,
    prefer_block_pointer=_base.prefer_block_pointer,
    prefer_1d_tile=_base.prefer_1d_tile,
)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=_erf_config)
@triton.jit
def erf_func(x):
    x_fp32 = x.to(tl.float32)
    abs_x = tl.abs(x_fp32)
    t = 1.0 / (1.0 + 0.3275911 * abs_x)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    poly = (
        0.254829592 * t
        + (-0.284496736) * t2
        + 1.421413741 * t3
        + (-1.453152027) * t4
        + 1.061405429 * t5
    )
    result = 1.0 - poly * tl.exp(-abs_x * abs_x)
    return tl.where(x_fp32 >= 0.0, result, -result)


def erf(x):
    logger.debug("GEMS ERF")
    return erf_func(x)


def erf_(x):
    logger.debug("GEMS ERF_")
    return erf_func(x, out0=x)