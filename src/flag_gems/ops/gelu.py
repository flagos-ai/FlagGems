import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    out = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(out_ptr + offsets, out, mask=mask)


@libentry()
@triton.jit
def gelu_backward_kernel(x_ptr, dy_ptr, dx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    tanh_inner = tl.math.tanh(inner)
    sech2 = 1.0 - tanh_inner * tanh_inner
    dtanh = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)
    dx = dy * (0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * dtanh)
    tl.store(dx_ptr + offsets, dx, mask=mask)


def _fwd(x):
    orig = x.dtype; x = x.contiguous().float()
    out = torch.empty_like(x); n = x.numel()
    if n == 0: return out.to(orig)
    BS = triton.next_power_of_2(min(n, 4096))
    gelu_kernel[(triton.cdiv(n, BS),)](x, out, n, BLOCK_SIZE=BS, num_warps=4)
    return out.to(orig)


def gelu(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS GELU"); return _fwd(x)

def gelu_(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS GELU_"); r = _fwd(x); x.copy_(r); return x

def gelu_backward(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS GELU BACKWARD")
    orig = x.dtype; x = x.contiguous().float(); dy = dy.contiguous().float()
    dx = torch.empty_like(x); n = x.numel()
    if n == 0: return dx.to(orig)
    BS = triton.next_power_of_2(min(n, 4096))
    gelu_backward_kernel[(triton.cdiv(n, BS),)](x, dy, dx, n, BLOCK_SIZE=BS, num_warps=4)
    return dx.to(orig)
