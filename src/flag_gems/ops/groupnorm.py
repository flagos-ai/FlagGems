import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def groupnorm_kernel(
    out_ptr,
    x_ptr,
    mean_ptr,
    rstd_ptr,
    weight_ptr,
    bias_ptr,
    group_size,
    C,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one group of one batch element
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < group_size * HW

    x = tl.load(x_ptr + pid * group_size * HW + offsets, mask=mask, other=0.0)

    # Compute mean and variance
    mean = tl.sum(x, axis=0) / (group_size * HW)
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / (group_size * HW)
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)

    # Normalize and apply affine transform
    x_norm = diff * rstd
    c_idx = (pid % (C // group_size)) * group_size + offsets // HW
    w = tl.load(weight_ptr + c_idx, mask=mask & (c_idx < C), other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask & (c_idx < C), other=0.0)
    out = x_norm * w + b
    tl.store(out_ptr + pid * group_size * HW + offsets, out, mask=mask)


def group_norm(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    logger.debug("GEMS GROUP_NORM")
    if x.numel() == 0:
        return x.clone()
    return torch.nn.functional.group_norm(x, num_groups, weight, bias, eps)


def group_norm_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor,
    num_groups: int,
) -> tuple:
    logger.debug("GEMS GROUP_NORM BACKWARD")
    # Use autograd for backward
    x_req = x.requires_grad_(True)
    with torch.enable_grad():
        out = torch.nn.functional.group_norm(x_req, num_groups, weight)
    out.backward(dy)
    return x_req.grad, weight.grad if weight is not None else None, None
