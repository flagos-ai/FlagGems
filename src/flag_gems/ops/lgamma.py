import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def lgamma_func(x):
    # lgamma(x) = log(|Gamma(x)|)
    # Stirling's series for large x, recurrence for small x, reflection for x < 0.5
    xp = x.to(tl.float32)

    # For x < 0.5, use reflection: lgamma(x) = log(pi / sin(pi*x)) - lgamma(1-x)
    pi = 3.14159265358979323846
    reflect = xp < 0.5
    xr = tl.where(reflect, 1.0 - xp, xp)

    # Recurrence: shift xr up until xr >= 7 for Stirling accuracy
    s = tl.zeros_like(xr)
    y = xr
    for _ in range(7):
        m = y < 7.0
        s = s + tl.where(m, tl.log(y), 0.0)
        y = tl.where(m, y + 1.0, y)

    # Stirling's series for lgamma(y) where y >= 7
    half_log_2pi = 0.9189385332046727  # 0.5 * log(2*pi)
    r = 1.0 / y
    r2 = r * r
    stirling = (
        (y - 0.5) * tl.log(y)
        - y
        + half_log_2pi
        + r * (1.0 / 12.0 + r2 * (-1.0 / 360.0 + r2 * (1.0 / 1260.0)))
    )
    result = stirling - s

    # Apply reflection: lgamma(x) = log(pi) - log(|sin(pi*x)|) - lgamma(1-x)
    sin_pix = tl.sin(pi * xp)
    abs_sin = tl.where(sin_pix == 0.0, 1e-30, tl.abs(sin_pix))
    reflected = tl.log(pi) - tl.log(abs_sin) - result
    result = tl.where(reflect, reflected, result)

    return result


def lgamma(A):
    logger.debug("GEMS LGAMMA")
    return lgamma_func(A)


def lgamma_(A):
    logger.debug("GEMS LGAMMA_")
    lgamma_func(A, out0=A)
    return A
