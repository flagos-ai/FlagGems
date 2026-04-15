import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@triton.jit
def _gcd_block(a, b, steps: tl.constexpr):
    """Do `steps` euclidean iterations with early-exit."""
    nonzero = b != 0
    any_nonzero = tl.sum(nonzero.to(tl.int32)) > 0

    if any_nonzero:
        for _ in range(steps):
            safe_b = tl.where(nonzero, b, 1)
            r = a % safe_b
            a = tl.where(nonzero, b, a)
            b = tl.where(nonzero, r, b)
            nonzero = b != 0

    return a, b


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gcd_func(x, y):
    a = tl.abs(x)
    b = tl.abs(y)

    if x.dtype != y.dtype:
        if x.dtype == tl.int64 or y.dtype == tl.int64:
            a = a.to(tl.int64)
            b = b.to(tl.int64)
            dtype = tl.int64
        elif x.dtype == tl.int32 or y.dtype == tl.int32:
            a = a.to(tl.int32)
            b = b.to(tl.int32)
            dtype = tl.int32
        elif x.dtype == tl.int16 or y.dtype == tl.int16:
            a = a.to(tl.int16)
            b = b.to(tl.int16)
            dtype = tl.int16
        else:
            a = a.to(tl.int8)
            b = b.to(tl.int8)
            dtype = tl.int8
    else:
        dtype = x.dtype

    # Worst-case iterations ≈ log_φ(max_value):
    #   int8 → ~9,  int16 → ~22,  int32 → ~44,  int64 → ~90
    # Each _gcd_block has early-exit, so extra blocks are nearly free.
    if dtype == tl.int64:
        # 90 iterations (10 blocks × 10 steps)
        a, b = _gcd_block(a, b, 10)
        a, b = _gcd_block(a, b, 10)
        a, b = _gcd_block(a, b, 10)
        a, b = _gcd_block(a, b, 10)
        a, b = _gcd_block(a, b, 10)
        a, b = _gcd_block(a, b, 10)
        a, b = _gcd_block(a, b, 10)
        a, b = _gcd_block(a, b, 10)
        a, b = _gcd_block(a, b, 10)

    elif dtype == tl.int32:
        # 44 iterations (8 blocks × 5 steps + 1 block × 4 steps)
        a, b = _gcd_block(a, b, 5)
        a, b = _gcd_block(a, b, 5)
        a, b = _gcd_block(a, b, 5)
        a, b = _gcd_block(a, b, 5)
        a, b = _gcd_block(a, b, 5)
        a, b = _gcd_block(a, b, 5)
        a, b = _gcd_block(a, b, 5)
        a, b = _gcd_block(a, b, 5)
        a, b = _gcd_block(a, b, 4)

    elif dtype == tl.int16:
        # 22 iterations (4 blocks × 5 steps + 1 block × 2 steps)
        a, b = _gcd_block(a, b, 5)
        a, b = _gcd_block(a, b, 5)
        a, b = _gcd_block(a, b, 5)
        a, b = _gcd_block(a, b, 5)
        a, b = _gcd_block(a, b, 2)

    else:  # int8
        # 9 iterations (3 blocks × 3 steps)
        a, b = _gcd_block(a, b, 3)
        a, b = _gcd_block(a, b, 3)
        a, b = _gcd_block(a, b, 3)

    return a


def gcd(A, B):
    logger.debug("GEMS GCD")
    return gcd_func(A, B)


def gcd_(A, B):
    logger.debug("GEMS GCD_")
    return gcd_func(A, B, out0=A)