import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def gcd_kernel(
    A,
    B,
    OUT,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise GCD using Euclidean algorithm with manual kernel."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(A + offsets, mask=mask, other=0)
    b = tl.load(B + offsets, mask=mask, other=0)

    # Work with absolute values
    a = tl.abs(a)
    b = tl.abs(b)

    # Ensure a >= b
    swap = a < b
    tmp = tl.where(swap, a, b)
    b = tl.where(swap, b, a)
    a = tmp

    # Euclidean algorithm — 32 iterations
    for _ in range(32):
        safe_b = tl.where(b != 0, b, 1)
        r = a % safe_b
        a = tl.where(b != 0, b, a)
        b = tl.where(b != 0, r, b)

    tl.store(OUT + offsets, a, mask=mask)


def gcd(self, other):
    logger.debug("GEMS GCD")
    assert self.shape == other.shape or self.shape == () or other.shape == ()

    # Handle broadcasting
    self, other = torch.broadcast_tensors(self, other)
    self = self.contiguous()
    other = other.contiguous()

    output = torch.empty_like(self)
    n = self.numel()

    if n == 0:
        return output

    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    gcd_kernel[grid](self, other, output, n, BLOCK_SIZE=BLOCK_SIZE)

    return output
