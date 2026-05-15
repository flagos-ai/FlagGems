import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


def _relu(x):
    orig = x.dtype
    x = x.contiguous().float()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out.to(orig)
    BS = triton.next_power_of_2(min(n, 4096))
    relu_kernel[(triton.cdiv(n, BS),)](x, out, n, BLOCK_SIZE=BS, num_warps=4)
    return out.to(orig)


def relu(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS RELU")
    return _relu(x)


def relu_(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS RELU_")
    r = _relu(x)
    x.copy_(r)
    return x
