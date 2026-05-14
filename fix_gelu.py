"""Run from inside FlagGems folder"""
content = """import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
GELU_COEFF = 0.044715


@libentry()
@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x3 = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)
    out = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(out_ptr + offsets, out, mask=mask)


@libentry()
@triton.jit
def gelu_backward_kernel(
    x_ptr,
    dy_ptr,
    dx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x  = tl.load(x_ptr  + offsets, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)
    x3 = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)
    tanh_inner = tl.math.tanh(inner)
    sech2 = 1.0 - tanh_inner * tanh_inner
    dtanh = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)
    dx = dy * (0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * dtanh)
    tl.store(dx_ptr + offsets, dx, mask=mask)


def _gelu_fwd(x: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.contiguous().float()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out.to(orig_dtype)
    BLOCK_SIZE = triton.next_power_of_2(min(n, 4096))
    num_warps = 2 if BLOCK_SIZE <= 256 else 4 if BLOCK_SIZE <= 1024 else 8
    gelu_kernel[(triton.cdiv(n, BLOCK_SIZE),)](x, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return out.to(orig_dtype)


def gelu(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS GELU")
    return _gelu_fwd(x)


def gelu_(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS GELU_")
    result = _gelu_fwd(x)
    x.copy_(result)
    return x


def gelu_backward(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS GELU BACKWARD")
    orig_dtype = x.dtype
    x  = x.contiguous().float()
    dy = dy.contiguous().float()
    dx = torch.empty_like(x)
    n  = x.numel()
    if n == 0:
        return dx.to(orig_dtype)
    BLOCK_SIZE = triton.next_power_of_2(min(n, 4096))
    num_warps = 2 if BLOCK_SIZE <= 256 else 4 if BLOCK_SIZE <= 1024 else 8
    gelu_backward_kernel[(triton.cdiv(n, BLOCK_SIZE),)](x, dy, dx, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return dx.to(orig_dtype)
"""
import os, re
path = os.path.join("src","flag_gems","ops","gelu.py")
with open(path,"w",encoding="utf-8") as f: f.write(content)
code = open(path,encoding="utf-8").read()
funcs = [f for f in re.findall(r"^def (\w+)\(",code,re.MULTILINE) if not f.startswith("_")]
print("Functions:",funcs)
print("Has gelu:          ", "gelu"          in funcs)
print("Has gelu_:         ", "gelu_"         in funcs)
print("Has gelu_backward: ", "gelu_backward" in funcs)
print("DONE")
