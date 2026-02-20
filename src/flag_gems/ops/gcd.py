import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Worst-case Euclidean steps for int32: 46 (consecutive Fibonacci numbers).
# 48 covers this with margin.
_GCD_ITERS = 48

_BLOCK_SIZE = 1024


@libentry()
@triton.jit
def gcd_kernel(
    A,
    B,
    OUT,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(A + offsets, mask=mask, other=0)
    b = tl.load(B + offsets, mask=mask, other=0)

    # Work with absolute values — cast to int32 to avoid signed overflow.
    a = tl.abs(a.to(tl.int32))
    b = tl.abs(b.to(tl.int32))

    # Euclidean algorithm with fixed iterations (warp-divergence-free).
    for _ in range(_GCD_ITERS):
        safe_b = tl.where(b != 0, b, 1)
        r = a % safe_b
        a = tl.where(b != 0, b, a)
        b = tl.where(b != 0, r, b)

    tl.store(OUT + offsets, a.to(OUT.dtype.element_ty), mask=mask)


def _gcd_launch(A, B, out):
    A, B = torch.broadcast_tensors(A, B)
    A = A.contiguous()
    B = B.contiguous()
    n = A.numel()
    if n == 0:
        return out
    grid = (triton.cdiv(n, _BLOCK_SIZE),)
    gcd_kernel[grid](A, B, out, n, BLOCK_SIZE=_BLOCK_SIZE)
    return out


def gcd(A, B):
    logger.debug("GEMS GCD")
    A, B = torch.broadcast_tensors(A, B)
    out = torch.empty_like(A)
    return _gcd_launch(A, B, out)


def gcd_(A, B):
    logger.debug("GEMS GCD_")
    return _gcd_launch(A, B, A)


def gcd_out(A, B, *, out):
    logger.debug("GEMS GCD OUT")
    return _gcd_launch(A, B, out)
