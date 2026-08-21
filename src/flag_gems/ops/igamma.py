import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def igamma_func(a, x):
    # igamma(a, x) = P(a, x) = lower regularized incomplete gamma function
    # Series: P(a,x) = exp(-x + a*log(x) - lgamma(a)) * sum_{n=0}^inf x^n/((a)(a+1)...(a+n))
    af = a.to(tl.float32)
    xf = x.to(tl.float32)

    # Boundary: x <= 0 => 0
    valid = xf > 0.0

    # Shift af up to >= 7 for Stirling lgamma (recurrence: lgamma(a) = lgamma(a+1) - log(a))
    log_prod = tl.zeros_like(af)
    ay = af
    for _ in range(7):
        m = ay < 7.0
        log_prod = log_prod + tl.where(m, tl.log(ay), 0.0)
        ay = tl.where(m, ay + 1.0, ay)

    half_log_2pi = 0.9189385332046727
    ry = 1.0 / ay
    ry2 = ry * ry
    lgamma_ay = (
        (ay - 0.5) * tl.log(ay)
        - ay
        + half_log_2pi
        + ry * (1.0 / 12.0 + ry2 * (-1.0 / 360.0 + ry2 * (1.0 / 1260.0)))
    )
    lgamma_a = lgamma_ay - log_prod

    # Series: P(a,x) = exp(-x + a*log(x) - lgamma(a)) * S
    # S = 1/a + x/(a*(a+1)) + x^2/(a*(a+1)*(a+2)) + ...
    ap = af
    term = 1.0 / af
    s = term
    for _ in range(60):
        ap = ap + 1.0
        term = term * xf / ap
        s = s + term

    log_prefix = -xf + af * tl.log(tl.where(xf > 0.0, xf, 1.0)) - lgamma_a
    result = tl.exp(log_prefix) * s

    return tl.where(valid, result, 0.0)


def igamma(a, x):
    logger.debug("GEMS IGAMMA")
    return igamma_func(a, x)
