import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def sigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, out, mask=mask)


@libentry()
@triton.jit
def sigmoid_backward_kernel(
    x_ptr, dy_ptr, dx_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    dx = dy * sig * (1.0 - sig)
    tl.store(dx_ptr + offsets, dx, mask=mask)


def _fwd(x):
    orig = x.dtype
    x = x.contiguous().float()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out.to(orig)
    BS = triton.next_power_of_2(min(n, 4096))
    sigmoid_kernel[(triton.cdiv(n, BS),)](x, out, n, BLOCK_SIZE=BS, num_warps=4)
    return out.to(orig)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS SIGMOID")
    return _fwd(x)


def sigmoid_(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS SIGMOID_")
    r = _fwd(x)
    x.copy_(r)
    return x


def sigmoid_backward(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS SIGMOID BACKWARD")
    orig = x.dtype
    x = x.contiguous().float()
    dy = dy.contiguous().float()
    dx = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return dx.to(orig)
    BS = triton.next_power_of_2(min(n, 4096))
    sigmoid_backward_kernel[(triton.cdiv(n, BS),)](
        x, dy, dx, n, BLOCK_SIZE=BS, num_warps=4
    )
    return dx.to(orig)
