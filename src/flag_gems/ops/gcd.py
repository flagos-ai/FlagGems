import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

# Worst-case Euclidean steps for int32: 46 (consecutive Fibonacci numbers).
# 48 covers this with margin, and keeps large-tensor workloads memory-bound
# rather than compute-bound, yielding competitive performance vs PyTorch.
_GCD_ITERS = 48


@triton.jit
def _gcd_impl(x, y):
    # Work on absolute values; gcd(a, b) == gcd(|a|, |b|).
    x = tl.where(x < 0, -x, x)
    y = tl.where(y < 0, -y, y)
    # Fixed-iteration Euclidean algorithm (branchless / warp-divergence-free).
    for _ in tl.static_range(_GCD_ITERS):
        nonzero = y != 0
        safe_y = tl.where(nonzero, y, 1)  # guard against division by zero
        r = x % safe_y
        x = tl.where(nonzero, y, x)
        y = tl.where(nonzero, r, y)
    return x


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gcd_func(x, y):
    return _gcd_impl(x, y)


def gcd(A, B):
    logger.debug("GEMS GCD")
    return gcd_func(A, B)


def gcd_(A, B):
    logger.debug("GEMS GCD_")
    return gcd_func(A, B, out0=A)


def gcd_out(A, B, *, out):
    logger.debug("GEMS GCD OUT")
    return gcd_func(A, B, out0=out)
