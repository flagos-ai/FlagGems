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
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@triton.jit
def _digamma_pos(xr):
    s = tl.zeros_like(xr)
    y = xr
    m0 = y < 8.0
    s = s - tl.where(m0, 1.0 / y, 0.0)
    y = tl.where(m0, y + 1.0, y)
    m1 = y < 8.0
    s = s - tl.where(m1, 1.0 / y, 0.0)
    y = tl.where(m1, y + 1.0, y)
    m2 = y < 8.0
    s = s - tl.where(m2, 1.0 / y, 0.0)
    y = tl.where(m2, y + 1.0, y)
    m3 = y < 8.0
    s = s - tl.where(m3, 1.0 / y, 0.0)
    y = tl.where(m3, y + 1.0, y)
    m4 = y < 8.0
    s = s - tl.where(m4, 1.0 / y, 0.0)
    y = tl.where(m4, y + 1.0, y)
    m5 = y < 8.0
    s = s - tl.where(m5, 1.0 / y, 0.0)
    y = tl.where(m5, y + 1.0, y)
    m6 = y < 8.0
    s = s - tl.where(m6, 1.0 / y, 0.0)
    y = tl.where(m6, y + 1.0, y)
    m7 = y < 8.0
    s = s - tl.where(m7, 1.0 / y, 0.0)
    y = tl.where(m7, y + 1.0, y)
    r = 1.0 / y
    r2 = r * r
    t4 = r2 * r2
    t6 = t4 * r2
    t8 = t4 * t4
    series = (
        -0.5 * r
        + (-1.0 / 12.0) * r2
        + (1.0 / 120.0) * t4
        + (-1.0 / 252.0) * t6
        + (1.0 / 240.0) * t8
    )
    return tl.log(y) + s + series


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def special_digamma_func(x):
    pi = 3.141592653589793
    xf = x.to(tl.float32)
    reflect = xf < 0.5
    xr = tl.where(reflect, 1.0 - xf, xf)
    psi_pos = _digamma_pos(xr)
    rr = xf - tl.floor(xf + 0.5)
    arg = pi * rr
    cot = tl.cos(arg) / tl.sin(arg)
    return tl.where(reflect, psi_pos - pi * cot, psi_pos)


def special_digamma(A):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_DIGAMMA")
    return special_digamma_func(A)
