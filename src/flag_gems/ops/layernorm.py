import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def layer_norm_kernel(
    out_ptr, x_ptr, weight_ptr, bias_ptr, n_elements, eps, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_elements
    x = tl.load(x_ptr + row_idx * n_elements + cols, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_elements
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_elements
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row_idx * n_elements + cols, diff * rstd * w + b, mask=mask)


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    logger.debug("GEMS LAYER_NORM")
    if x.numel() == 0:
        return x.clone()
    orig = x.dtype
    x = x.contiguous().float()
    if weight is not None:
        weight = weight.contiguous().float()
    if bias is not None:
        bias = bias.contiguous().float()
    n = 1
    for s in normalized_shape:
        n *= s
    rows = x.numel() // n
    x2d = x.reshape(rows, n)
    out = torch.empty_like(x2d)
    if weight is None:
        weight = torch.ones(n, device=x.device, dtype=torch.float32)
    if bias is None:
        bias = torch.zeros(n, device=x.device, dtype=torch.float32)
    BS = min(triton.next_power_of_2(n), 65536)
    layer_norm_kernel[(rows,)](
        out, x2d, weight, bias, n, eps, BLOCK_SIZE=BS, num_warps=4
    )
    return out.view(x.shape).to(orig)


def layer_norm_backward(dy, x, normalized_shape, weight, bias, eps=1e-5):
    logger.debug("GEMS LAYER_NORM BACKWARD")
    if dy.numel() == 0:
        return dy.clone(), None, None
    orig = dy.dtype
    dy = dy.contiguous().float()
    x = x.contiguous().float()
    n = 1
    for s in normalized_shape:
        n *= s
    rows = x.numel() // n
    x2d = x.reshape(rows, n)
    dy2d = dy.reshape(rows, n)
    mean = x2d.mean(dim=1)
    var = x2d.var(dim=1, unbiased=False)
    rstd = 1.0 / torch.sqrt(var + eps)
    w = (
        weight.contiguous().float()
        if weight is not None
        else torch.ones(n, device=x.device, dtype=torch.float32)
    )
    x_hat = (x2d - mean[:, None]) * rstd[:, None]
    wdy = w * dy2d
    c1 = (wdy * x_hat).sum(1, keepdim=True) / n
    c2 = wdy.sum(1, keepdim=True) / n
    dx = (rstd[:, None] * (wdy - x_hat * c1 - c2)).view(x.shape).to(orig)
    dw = (dy2d * x_hat).sum(0).to(orig) if weight is not None else None
    db = dy2d.sum(0).to(orig) if bias is not None else None
    return dx, dw, db
