"""Run from inside FlagGems folder to write correct rms_norm.py"""

content = """import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def rms_norm_kernel(
    out_ptr,
    x_ptr,
    weight_ptr,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per row
    row_idx = tl.program_id(axis=0)
    row_start = row_idx * n_elements
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_elements

    # Load row
    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)

    # Compute RMS: sqrt(mean(x^2) + eps)
    x_sq = tl.where(mask, x * x, 0.0)
    mean_sq = tl.sum(x_sq, axis=0) / n_elements
    rrms = 1.0 / tl.sqrt(mean_sq + eps)

    # Normalize and scale
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    out = x * rrms * w

    tl.store(out_ptr + row_start + cols, out, mask=mask)


@libentry()
@triton.jit
def rms_norm_backward_kernel(
    dx_ptr,
    dy_ptr,
    x_ptr,
    weight_ptr,
    rrms_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    row_start = row_idx * n_elements
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_elements

    x    = tl.load(x_ptr    + row_start + cols, mask=mask, other=0.0)
    dy   = tl.load(dy_ptr   + row_start + cols, mask=mask, other=0.0)
    w    = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    rrms = tl.load(rrms_ptr + row_idx)

    wdy  = w * dy
    # dx = rrms * (wdy - x * rrms^2 * sum(wdy * x) / n)
    c    = tl.sum(tl.where(mask, wdy * x, 0.0), axis=0) * (rrms * rrms) / n_elements
    dx   = rrms * (wdy - x * c)

    tl.store(dx_ptr + row_start + cols, dx, mask=mask)


def _forward(x: torch.Tensor, weight: torch.Tensor, eps: float):
    orig_dtype = x.dtype
    x = x.contiguous().float()
    w = weight.contiguous().float() if weight is not None else torch.ones(x.shape[-1], device=x.device)
    n_elements = x.shape[-1]
    num_rows = x.numel() // n_elements
    x2d = x.reshape(num_rows, n_elements)
    out = torch.empty_like(x2d)
    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 65536)
    num_warps = 2 if BLOCK_SIZE <= 256 else 4 if BLOCK_SIZE <= 1024 else 8
    rms_norm_kernel[(num_rows,)](
        out, x2d, w, n_elements, eps,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
    )
    return out.view(x.shape).to(orig_dtype)


def rms_norm(
    x: torch.Tensor,
    normalized_shape: list,
    weight: torch.Tensor = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    logger.debug("GEMS RMS_NORM")
    if x.numel() == 0:
        return x.clone()
    return _forward(x, weight, eps)


def rms_norm_forward(
    x: torch.Tensor,
    normalized_shape: list,
    weight: torch.Tensor = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    logger.debug("GEMS RMS_NORM_FORWARD")
    if x.numel() == 0:
        return x.clone()
    return _forward(x, weight, eps)


def rms_norm_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple:
    logger.debug("GEMS RMS_NORM_BACKWARD")
    if dy.numel() == 0:
        return dy.clone(), None

    orig_dtype = dy.dtype
    dy = dy.contiguous().float()
    x  = x.contiguous().float()
    w  = weight.contiguous().float() if weight is not None else torch.ones(x.shape[-1], device=x.device)

    n_elements = x.shape[-1]
    num_rows = x.numel() // n_elements
    x2d  = x.reshape(num_rows, n_elements)
    dy2d = dy.reshape(num_rows, n_elements)
    dx   = torch.empty_like(x2d)

    # Compute rrms for each row
    mean_sq = (x2d * x2d).mean(dim=1)
    rrms    = 1.0 / torch.sqrt(mean_sq + eps)

    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 65536)
    num_warps  = 2 if BLOCK_SIZE <= 256 else 4 if BLOCK_SIZE <= 1024 else 8

    rms_norm_backward_kernel[(num_rows,)](
        dx, dy2d, x2d, w, rrms, n_elements,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
    )

    dx = dx.view(x.shape).to(orig_dtype)
    dw = (dy2d * x2d * rrms[:, None]).sum(0).to(orig_dtype) if weight is not None else None
    return dx, dw
"""

import os
path = os.path.join("src", "flag_gems", "ops", "rms_norm.py")
with open(path, "w", encoding="utf-8") as f:
    f.write(content)

import re
code = open(path, encoding="utf-8").read()
funcs = [f for f in re.findall(r"^def (\w+)\(", code, re.MULTILINE) if not f.startswith("_")]
print("Functions:", funcs)
print("Has rms_norm:          ", "rms_norm"          in funcs)
print("Has rms_norm_forward:  ", "rms_norm_forward"  in funcs)
print("Has rms_norm_backward: ", "rms_norm_backward" in funcs)
print("DONE")
