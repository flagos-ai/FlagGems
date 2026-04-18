import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gcd_func(a, b):
    # Euclidean algorithm using absolute values
    # Works for integer types; result is always non-negative
    x = tl.abs(a)
    y = tl.abs(b)
    # Unrolled Euclidean loop — Triton requires static loop bounds.
    # 64 iterations covers gcd for all 64-bit integers (worst case ~93 steps for Fibonacci).
    for _ in tl.static_range(64):
        cond = y != 0
        tmp = tl.where(cond, x % y, x)
        x = tl.where(cond, y, x)
        y = tl.where(cond, tmp, y)
    return x


def gcd(A, B):
    logger.debug("GEMS GCD")
    return gcd_func(A, B)
