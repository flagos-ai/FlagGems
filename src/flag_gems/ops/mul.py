import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x * y, mask=mask)


def _mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.contiguous().float()
    y = y.contiguous().float()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out.to(orig_dtype)
    BLOCK_SIZE = triton.next_power_of_2(min(n, 4096))
    num_warps = 2 if BLOCK_SIZE <= 256 else 4 if BLOCK_SIZE <= 1024 else 8
    mul_kernel[(triton.cdiv(n, BLOCK_SIZE),)](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return out.to(orig_dtype)


def mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS MUL")
    return _mul(x, y)


def mul_(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS MUL_")
    result = _mul(x, y)
    x.copy_(result)
    return x
