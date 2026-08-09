# Copyright 2026 FlagOS Contributors
#
# Kunlunxin (XPU) override of special_digamma.
#
# Root cause: generic `flag_gems/ops/special_digamma.py` aliases the raw-pointer
# `digamma` kernel from `flag_gems/ops/digamma_.py`, whose `for _ in range(8)`
# recurrence + reflection cot term crashes the TritonXPUVectorize pass on bf16.
#
# Fix: full-domain digamma via `pointwise_dynamic` (isCloseVectorization=True
# avoids the vectorizer crash). special_digamma is exercised across x>=1,
# small positive (0.05,0.45), mid (0.5,1.0) and negative (-4.9,-0.1) inputs, so
# both the recurrence and reflection paths are needed:
#   psi(x) = psi(1-x) - pi*cot(pi*x)  for x < 0.5.
# The cot argument is period-reduced to [-0.5, 0.5] (cot(pi*x)=cot(pi*r),
# r = x - round(x)) so XPU sin/cos stays accurate for large |x|.
import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import pointwise_dynamic

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
    # xr >= 0.5: shift up to y >= 8 via recurrence psi(x) = psi(x+1) - 1/x,
    # then asymptotic expansion. 8 unconditional steps (tl.where masked).
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
    # cot(pi*xf) = cot(pi*r), r = xf - round(xf) in [-0.5, 0.5] keeps the
    # trig argument bounded (accurate on XPU for large |xf|).
    rr = xf - tl.floor(xf + 0.5)
    arg = pi * rr
    cot = tl.cos(arg) / tl.sin(arg)
    return tl.where(reflect, psi_pos - pi * cot, psi_pos)


def special_digamma(A):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_DIGAMMA")
    return special_digamma_func(A)
