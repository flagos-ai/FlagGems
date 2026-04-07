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


# Smooth L1 Loss (Huber Loss):
#   loss = 0.5 * (x - y)^2 / beta   if |x - y| < beta
#          |x - y| - 0.5 * beta     otherwise
# Supports reduction modes: none (elementwise), mean, sum.
# Uses two-phase parallel reduction for mean/sum.


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def smooth_l1_loss_none_kernel(inp, target, beta):
    x = inp.to(tl.float32)
    y = target.to(tl.float32)
    diff = tl.abs(x - y)
    return tl.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)


@libentry()
@triton.jit
def smooth_l1_loss_reduce_kernel(
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

    inp_val = tl.load(inp + offset, mask=mask, other=0).to(tl.float32)
    target_val = tl.load(target + offset, mask=mask, other=0).to(tl.float32)
    diff = tl.abs(inp_val - target_val)
    loss = tl.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)

    if reduction == 1:
        sum_val = tl.sum(loss) / M
    else:
        sum_val = tl.sum(loss)

    tl.store(mid + pid, sum_val)


@libentry()
@triton.jit
def reduce_sum_kernel(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    mid_val = tl.load(mid + offset, mask=mask, other=0).to(tl.float32)
    tl.store(out, tl.sum(mid_val))


def smooth_l1_loss(inp, target, reduction=Reduction.MEAN.value, beta=1.0):
    logger.debug("GEMS SMOOTH_L1_LOSS")

    if reduction == Reduction.NONE.value:
        return smooth_l1_loss_none_kernel(inp, target, beta)

    inp = inp.contiguous()
    target = target.contiguous()
    M = inp.numel()
    dtype = inp.dtype

    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.float32, device=inp.device)
    out = torch.empty([], dtype=torch.float32, device=inp.device)

    with torch_device_fn.device(inp.device):
        smooth_l1_loss_reduce_kernel[(mid_size, 1, 1)](
            inp, target, mid, M, beta, block_size, reduction
        )
        reduce_sum_kernel[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out.to(dtype)
