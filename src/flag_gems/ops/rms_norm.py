import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def rms_norm_kernel(
    out_ptr, x_ptr, weight_ptr, n_elements, eps, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_elements
    x = tl.load(x_ptr + row_idx * n_elements + cols, mask=mask, other=0.0)
    x_sq = tl.where(mask, x * x, 0.0)
    rrms = 1.0 / tl.sqrt(tl.sum(x_sq, axis=0) / n_elements + eps)
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    tl.store(out_ptr + row_idx * n_elements + cols, x * rrms * w, mask=mask)


def _fwd(x, weight, eps):
    orig = x.dtype
    x = x.contiguous().float()
    w = (
        weight.contiguous().float()
        if weight is not None
        else torch.ones(x.shape[-1], device=x.device)
    )
    n = x.shape[-1]
    rows = x.numel() // n
    x2d = x.reshape(rows, n)
    out = torch.empty_like(x2d)
    BS = min(triton.next_power_of_2(n), 65536)
    rms_norm_kernel[(rows,)](out, x2d, w, n, eps, BLOCK_SIZE=BS, num_warps=4)
    return out.view(x.shape).to(orig)


def rms_norm(x, normalized_shape, weight=None, eps=1e-6):
    logger.debug("GEMS RMS_NORM")
    if x.numel() == 0:
        return x.clone()
    return _fwd(x, weight, eps)


def rms_norm_forward(x, normalized_shape, weight=None, eps=1e-6):
    logger.debug("GEMS RMS_NORM_FORWARD")
    if x.numel() == 0:
        return x.clone()
    return _fwd(x, weight, eps)


def rms_norm_backward(dy, x, weight, eps=1e-6):
    logger.debug("GEMS RMS_NORM_BACKWARD")
    if dy.numel() == 0:
        return dy.clone(), None
    orig = dy.dtype
    dy = dy.contiguous().float()
    x = x.contiguous().float()
    w = (
        weight.contiguous().float()
        if weight is not None
        else torch.ones(x.shape[-1], device=x.device)
    )
    n = x.shape[-1]
    rows = x.numel() // n
    x2d = x.reshape(rows, n)
    dy2d = dy.reshape(rows, n)
    mean_sq = (x2d * x2d).mean(dim=1)
    rrms = 1.0 / torch.sqrt(mean_sq + eps)
    wdy = w * dy2d
    c = (wdy * x2d).sum(1, keepdim=True) * (rrms**2)[:, None] / n
    dx = (rrms[:, None] * (wdy - x2d * c)).view(x.shape).to(orig)
    dw = (dy2d * x2d * rrms[:, None]).sum(0).to(orig) if weight is not None else None
    return dx, dw
