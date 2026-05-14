"""Run from inside FlagGems folder to write correct relu.py"""

content = """import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def relu_kernel(
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

    # ReLU: max(0, x)
    out = tl.maximum(x, 0.0)

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@libentry()
@triton.jit
def relu_backward_kernel(
    x_ptr,
    dy_ptr,
    dx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of elements
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)

    # ReLU backward: dy if x > 0 else 0
    dx = tl.where(x > 0, dy, 0.0)

    tl.store(dx_ptr + offsets, dx, mask=mask)


def _relu_forward(x: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.contiguous().float()
    out = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return x.to(orig_dtype)
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 4096))
    num_warps = 2 if BLOCK_SIZE <= 256 else 4 if BLOCK_SIZE <= 1024 else 8
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return out.to(orig_dtype)


def relu(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS RELU")
    return _relu_forward(x)


def relu_(x: torch.Tensor) -> torch.Tensor:
    # In-place ReLU
    logger.debug("GEMS RELU_")
    result = _relu_forward(x)
    x.copy_(result)
    return x
"""

import os
path = os.path.join("src", "flag_gems", "ops", "relu.py")
with open(path, "w", encoding="utf-8") as f:
    f.write(content)

import re
code = open(path, encoding="utf-8").read()
funcs = [f for f in re.findall(r"^def (\w+)\(", code, re.MULTILINE) if not f.startswith("_")]
print("Public functions:", funcs)
print("Has relu:  ", "relu"  in funcs)
print("Has relu_: ", "relu_" in funcs)
print("DONE")
