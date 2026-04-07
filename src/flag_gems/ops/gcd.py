import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# Binary GCD algorithm: avoids expensive modulo operations.
# Uses only subtraction, comparison, and division by 2 (bit shift).
# Much faster than Euclidean algorithm in GPU kernels where
# integer modulo has high latency.
@libentry()
@triton.jit
def gcd_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0).to(tl.int64)
    b = tl.load(b_ptr + offsets, mask=mask, other=0).to(tl.int64)
    a = tl.abs(a)
    b = tl.abs(b)

    # Handle zeros: gcd(0, x) = x, gcd(x, 0) = x
    a_zero = a == 0
    b_zero = b == 0
    a = tl.where(a_zero, b, a)
    b = tl.where(b_zero, a, b)

    # Binary GCD: repeatedly subtract the smaller from the larger
    for _ in range(64):
        # Ensure a >= b
        swap = a < b
        tmp = tl.where(swap, a, b)
        a = tl.where(swap, b, a)
        b = tmp

        # a = a - b (when b != 0)
        cond = b != 0
        a = tl.where(cond, a - b, a)

        # Divide a by 2 while even (approximate: just one step per iteration)
        a_even = (a & 1) == 0
        a = tl.where(a_even & (a > 0), a >> 1, a)

    # Result is the non-zero value
    result = tl.where(a != 0, a, b)
    tl.store(out_ptr + offsets, result, mask=mask)


def gcd(A, B):
    logger.debug("GEMS GCD")
    A = A.contiguous()
    B = B.contiguous()
    out = torch.empty_like(A)
    n_elements = A.numel()
    if n_elements == 0:
        return out
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(A.device):
        gcd_kernel[grid](A, B, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
