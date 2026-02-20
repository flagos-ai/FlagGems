import logging
import math
from enum import Enum

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


# ---------------------------------------------------------------------------
# Forward – elementwise only (reduction='none')
# ---------------------------------------------------------------------------
@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit
def smooth_l1_none_kernel(x, y, beta):
    diff = x - y
    abs_diff = tl.abs(diff)
    return tl.where(abs_diff < beta, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta)


# ---------------------------------------------------------------------------
# Forward – first pass: per-block partial sum (mean or sum)
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def smooth_l1_fwd_kernel(
    inp,
    target,
    mid,
    M,
    beta,
    BLOCK_SIZE: tl.constexpr,
    reduction: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M

    x = tl.load(inp + offset, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(target + offset, mask=mask, other=0.0).to(tl.float32)
    diff = x - y
    abs_diff = tl.abs(diff)
    huber = tl.where(abs_diff < beta, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta)

    if reduction == 1:  # MEAN
        tl.store(mid + pid, tl.sum(huber) / M)
    else:  # SUM
        tl.store(mid + pid, tl.sum(huber))


# ---------------------------------------------------------------------------
# Forward – second pass: sum partial results into scalar output
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def smooth_l1_reduce_kernel(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    val = tl.load(mid + offset, mask=mask, other=0.0).to(tl.float32)
    tl.store(out, tl.sum(val))


# ---------------------------------------------------------------------------
# Backward – elementwise for all reduction modes
#   scale = 1/N  for MEAN,  1.0 for SUM/NONE
# ---------------------------------------------------------------------------
@pointwise_dynamic(
    is_tensor=[True, True, True, False, False],
    promotion_methods=[(1, 2, "DEFAULT")],
)
@triton.jit
def smooth_l1_bwd_kernel(grad_out, x, y, beta, scale):
    diff = x - y
    abs_diff = tl.abs(diff)
    sign_diff = tl.where(diff > 0, 1.0, tl.where(diff < 0, -1.0, 0.0))
    grad_in = tl.where(abs_diff < beta, diff / beta, sign_diff)
    return grad_out * grad_in * scale


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
def smooth_l1_loss(inp, target, reduction=Reduction.MEAN.value, beta=1.0):
    logger.debug("GEMS SMOOTH_L1_LOSS")

    if reduction == Reduction.NONE.value:
        return smooth_l1_none_kernel(inp, target, beta)

    inp = inp.contiguous()
    target = target.contiguous()
    M = inp.numel()
    dtype = inp.dtype

    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.float32, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        smooth_l1_fwd_kernel[(mid_size, 1, 1)](
            inp, target, mid, M, beta, block_size, reduction
        )
        smooth_l1_reduce_kernel[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


def smooth_l1_loss_backward(
    grad_output, inp, target, reduction=Reduction.MEAN.value, beta=1.0
):
    logger.debug("GEMS SMOOTH_L1_LOSS_BACKWARD")
    scale = 1.0 / inp.numel() if reduction == Reduction.MEAN.value else 1.0
    return smooth_l1_bwd_kernel(grad_output, inp, target, beta, scale)
