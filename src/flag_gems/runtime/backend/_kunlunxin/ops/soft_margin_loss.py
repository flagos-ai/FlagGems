import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _soft_margin_loss_elementwise(x, y):
    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    z = -xf * yf
    absz = tl.abs(z)
    return tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-absz))


@libentry()
@triton.jit
def kernel_1(
    x_ptr,
    y_ptr,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
    reduction: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M

    xf = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
    yf = tl.load(y_ptr + offset, mask=mask, other=0).to(tl.float32)

    z = -xf * yf
    absz = tl.abs(z)
    vals = tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-absz))
    # Zero out contributions from out-of-bounds elements
    # (soft_margin_loss(0,0) = log(2) != 0, so masking is required)
    vals = tl.where(mask, vals, 0.0)

    # Reduction.MEAN.value: 1, Reduction.SUM.value: 2
    if reduction == 1:
        sum_val = tl.sum(vals) / M
    else:
        sum_val = tl.sum(vals)

    tl.store(mid + pid, sum_val)


@libentry()
@triton.jit
def kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    mid_val = tl.load(mid + offset, mask=mask, other=0).to(tl.float32)
    sum_val = tl.sum(mid_val)
    tl.store(out, sum_val)


def _normalize_reduction(reduction):
    if isinstance(reduction, str):
        r = reduction.lower()
        if r == "none":
            return 0
        if r == "mean":
            return 1
        if r == "sum":
            return 2
        raise ValueError(f"Invalid reduction: {reduction}")
    if isinstance(reduction, int):
        if reduction in (0, 1, 2):
            return reduction
        raise ValueError(f"Invalid reduction int: {reduction}")
    raise ValueError(f"Unsupported reduction type: {type(reduction)}")


def soft_margin_loss(input: torch.Tensor, target: torch.Tensor, reduction="mean"):
    logger.debug("GEMS_KUNLUNXIN SOFT_MARGIN_LOSS")
    red = _normalize_reduction(reduction)

    if not input.is_contiguous():
        input = input.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()

    n_elements = input.numel()

    if red == 0:
        # reduction = 'none': use pointwise kernel (no atomic_add, no masked load issues)
        if n_elements == 0:
            return torch.empty_like(input)
        return _soft_margin_loss_elementwise(input, target)

    # reduction = 'sum' (red==2) or 'mean' (red==1)
    if n_elements == 0:
        if red == 2:
            return torch.zeros((), device=input.device, dtype=input.dtype)
        else:
            return torch.full((), float("nan"), device=input.device, dtype=input.dtype)

    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.float32, device=input.device)
    out = torch.empty([], dtype=torch.float32, device=input.device)

    import os

    os.environ["TRITONXPU_OTHER_SIM"] = "1"

    with torch_device_fn.device(input.device):
        kernel_1[(mid_size, 1, 1)](input, target, mid, n_elements, block_size, red)
        if mid_size == 1:
            result = mid.reshape([]).to(dtype=input.dtype)
            if "TRITONXPU_OTHER_SIM" in os.environ:
                del os.environ["TRITONXPU_OTHER_SIM"]
            return result
        kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)

    if "TRITONXPU_OTHER_SIM" in os.environ:
        del os.environ["TRITONXPU_OTHER_SIM"]

    return out.to(dtype=input.dtype)


def soft_margin_loss_out(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction="mean",
    out: torch.Tensor = None,
):
    logger.debug("GEMS_KUNLUNXIN SOFT_MARGIN_LOSS_OUT")
    result = soft_margin_loss(input, target, reduction)
    if out is None:
        return result
    out.copy_(result)
    return out
