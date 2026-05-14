"""Run from inside FlagGems folder to fix silu.py with all required functions"""

content = """import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of elements
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute sigmoid of x for SiLU activation
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))

    # SiLU is x multiplied by sigmoid of x
    out = x * sigmoid_x

    # Store result back to output buffer
    tl.store(out_ptr + offsets, out, mask=mask)


@libentry()
@triton.jit
def silu_backward_kernel(
    x_ptr,
    dy_ptr,
    dx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of elements
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)

    # Compute sigmoid and its derivative for backward pass
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))

    # SiLU backward: d/dx(x * sigmoid(x)) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    dx = dy * (sigmoid_x + x * sigmoid_x * (1.0 - sigmoid_x))

    # Store gradient
    tl.store(dx_ptr + offsets, dx, mask=mask)


def _silu_forward(x: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.contiguous().float()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 4096))
    if BLOCK_SIZE <= 256:
        num_warps = 2
    elif BLOCK_SIZE <= 1024:
        num_warps = 4
    else:
        num_warps = 8
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    silu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return out.to(orig_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS SILU")
    return _silu_forward(x)


def silu_(x: torch.Tensor) -> torch.Tensor:
    # In-place SiLU
    logger.debug("GEMS SILU_")
    result = _silu_forward(x)
    x.copy_(result)
    return x


def silu_backward(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    # Backward pass for SiLU
    logger.debug("GEMS SILU BACKWARD")
    orig_dtype = x.dtype
    x = x.contiguous().float()
    dy = dy.contiguous().float()
    dx = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 4096))
    if BLOCK_SIZE <= 256:
        num_warps = 2
    elif BLOCK_SIZE <= 1024:
        num_warps = 4
    else:
        num_warps = 8
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    silu_backward_kernel[grid](
        x, dy, dx, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
    )
    return dx.to(orig_dtype)
"""

import os
path = os.path.join("src", "flag_gems", "ops", "silu.py")
with open(path, "w", encoding="utf-8") as f:
    f.write(content)

code = open(path, encoding="utf-8").read()
import re
funcs = re.findall(r"^def (\w+)\(", code, re.MULTILINE)
print("Functions defined:", funcs)
print("Has silu:         ", "silu" in funcs)
print("Has silu_:        ", "silu_" in funcs)
print("Has silu_backward:", "silu_backward" in funcs)
print("DONE")
