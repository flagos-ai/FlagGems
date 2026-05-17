"""Run from inside FlagGems folder on a fresh branch"""

content = """import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def dropout_kernel(
    x_ptr,
    out_ptr,
    mask_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Generate random mask using Triton's random number generator
    random = tl.rand(seed, offsets)
    keep = random > p

    # Apply dropout and scale
    scale = 1.0 / (1.0 - p)
    out = tl.where(keep, x * scale, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)
    tl.store(mask_ptr + offsets, keep.to(tl.int8), mask=mask)


@libentry()
@triton.jit
def dropout_backward_kernel(
    dy_ptr,
    mask_ptr,
    dx_ptr,
    n_elements,
    p,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)
    keep = tl.load(mask_ptr + offsets, mask=mask, other=0).to(tl.int1)

    scale = 1.0 / (1.0 - p)
    dx = tl.where(keep, dy * scale, 0.0)

    tl.store(dx_ptr + offsets, dx, mask=mask)


def dropout(x: torch.Tensor, p: float = 0.5, training: bool = True) -> torch.Tensor:
    logger.debug("GEMS DROPOUT")
    if not training or p == 0.0:
        return x
    if p == 1.0:
        return torch.zeros_like(x)
    if x.numel() == 0:
        return x.clone()

    orig_dtype = x.dtype
    x = x.contiguous().float()
    out = torch.empty_like(x)
    mask = torch.empty(x.numel(), dtype=torch.int8, device=x.device)
    n = x.numel()
    BS = triton.next_power_of_2(min(n, 4096))
    seed = torch.randint(0, 2**31, (1,), dtype=torch.int32).item()
    dropout_kernel[(triton.cdiv(n, BS),)](
        x, out, mask, n, p, seed, BLOCK_SIZE=BS, num_warps=4
    )
    return out.to(orig_dtype)


def dropout_backward(
    dy: torch.Tensor, mask: torch.Tensor, p: float
) -> torch.Tensor:
    logger.debug("GEMS DROPOUT BACKWARD")
    if p == 0.0:
        return dy
    if p == 1.0:
        return torch.zeros_like(dy)
    if dy.numel() == 0:
        return dy.clone()

    orig_dtype = dy.dtype
    dy = dy.contiguous().float()
    dx = torch.empty_like(dy)
    n = dy.numel()
    BS = triton.next_power_of_2(min(n, 4096))
    dropout_backward_kernel[(triton.cdiv(n, BS),)](
        dy, mask, dx, n, p, BLOCK_SIZE=BS, num_warps=4
    )
    return dx.to(orig_dtype)
"""

import os
import re

path = os.path.join("src", "flag_gems", "ops", "dropout.py")
with open(path, "w", encoding="utf-8") as f:
    f.write(content)

code = open(path, encoding="utf-8").read()
funcs = [
    f for f in re.findall(r"^def (\w+)\(", code, re.MULTILINE) if not f.startswith("_")
]
print("Functions:", funcs)
print("Has dropout:          ", "dropout" in funcs)
print("Has dropout_backward: ", "dropout_backward" in funcs)
print("DONE")
