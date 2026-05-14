import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def layer_norm_kernel(
    out_ptr,
    x_ptr,
    weight_ptr,
    bias_ptr,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per row (each row is normalized independently)
    row_idx = tl.program_id(axis=0)
    row_start = row_idx * n_elements
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_elements

    # Load row
    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / n_elements

    # Compute variance
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_elements
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize
    x_norm = diff * rstd

    # Apply weight and bias
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    out = x_norm * w + b

    tl.store(out_ptr + row_start + cols, out, mask=mask)


@libentry()
@triton.jit
def layer_norm_backward_kernel(
    dx_ptr,
    dy_ptr,
    x_ptr,
    weight_ptr,
    mean_ptr,
    rstd_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    row_start = row_idx * n_elements
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_elements

    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + row_start + cols, mask=mask, other=0.0)
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    mean = tl.load(mean_ptr + row_idx)
    rstd = tl.load(rstd_ptr + row_idx)

    x_hat = (x - mean) * rstd
    wdy = w * dy

    # Gradient formula for layer norm
    c1 = tl.sum(wdy * x_hat, axis=0) / n_elements
    c2 = tl.sum(wdy, axis=0) / n_elements
    dx = rstd * (wdy - x_hat * c1 - c2)

    tl.store(dx_ptr + row_start + cols, dx, mask=mask)


def layer_norm(
    x: torch.Tensor,
    normalized_shape: list,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    logger.debug("GEMS LAYER_NORM")
    if x.numel() == 0:
        return x.clone()

    orig_dtype = x.dtype
    x = x.contiguous().float()
    if weight is not None:
        weight = weight.contiguous().float()
    if bias is not None:
        bias = bias.contiguous().float()

    # Flatten to 2D: (rows, n_elements)
    n_elements = 1
    for s in normalized_shape:
        n_elements *= s
    num_rows = x.numel() // n_elements
    x2d = x.reshape(num_rows, n_elements)
    out = torch.empty_like(x2d)

    if weight is None:
        weight = torch.ones(n_elements, device=x.device, dtype=torch.float32)
    if bias is None:
        bias = torch.zeros(n_elements, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 65536)
    num_warps = 2 if BLOCK_SIZE <= 256 else 4 if BLOCK_SIZE <= 1024 else 8

    layer_norm_kernel[(num_rows,)](
        out,
        x2d,
        weight,
        bias,
        n_elements,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return out.view(x.shape).to(orig_dtype)


def layer_norm_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    normalized_shape: list,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> tuple:
    logger.debug("GEMS LAYER_NORM BACKWARD")
    if dy.numel() == 0:
        return dy.clone(), None, None

    orig_dtype = dy.dtype
    dy = dy.contiguous().float()
    x = x.contiguous().float()
    w = weight.contiguous().float() if weight is not None else None

    n_elements = 1
    for s in normalized_shape:
        n_elements *= s
    num_rows = x.numel() // n_elements
    x2d = x.reshape(num_rows, n_elements)
    dy2d = dy.reshape(num_rows, n_elements)
    dx = torch.empty_like(x2d)

    # Compute mean and rstd
    mean = x2d.mean(dim=1)
    var = x2d.var(dim=1, unbiased=False)
    rstd = 1.0 / torch.sqrt(var + eps)

    if w is None:
        w = torch.ones(n_elements, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 65536)
    num_warps = 2 if BLOCK_SIZE <= 256 else 4 if BLOCK_SIZE <= 1024 else 8

    layer_norm_backward_kernel[(num_rows,)](
        dx,
        dy2d,
        x2d,
        w,
        mean,
        rstd,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    dx = dx.view(x.shape).to(orig_dtype)

    # Weight and bias gradients
    x_hat = (x2d - mean[:, None]) * rstd[:, None]
    dw = (dy2d * x_hat).sum(0).to(orig_dtype) if weight is not None else None
    db = dy2d.sum(0).to(orig_dtype) if bias is not None else None

    return dx, dw, db
