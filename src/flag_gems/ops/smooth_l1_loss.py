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


@libentry()
@triton.jit
def kernel_1(inp, target, mid, M, BLOCK_SIZE: tl.constexpr, reduction: tl.constexpr, beta: tl.constexpr):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    target_ptrs = target + offset
    mask = offset < M

    inp_val = tl.load(inp_ptrs, mask=mask, other=0).to(tl.float32)
    target_val = tl.load(target_ptrs, mask=mask, other=0).to(tl.float32)
    
    diff = inp_val - target_val
    abs_diff = tl.abs(diff)
    
    # smooth_l1_loss: 
    # when |diff| < beta: 0.5 * (diff / beta)^2 * beta
    # when |diff| >= beta: |diff| - 0.5 * beta
    beta_val = tl.cast(beta, tl.float32)
    loss = tl.where(
        abs_diff < beta_val,
        0.5 * diff * diff / beta_val,
        abs_diff - 0.5 * beta_val
    )
    
    if reduction == 1:
        sum_val = tl.sum(loss) / M
    else:
        sum_val = tl.sum(loss)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@libentry()
@triton.jit
def kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(tl.float32)
    sum_val = tl.sum(mid_val)
    tl.store(out, sum_val)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def func(x, y, beta: tl.constexpr):
    diff = x - y
    abs_diff = tl.abs(diff)
    beta_val = tl.cast(beta, x.dtype)
    return tl.where(
        abs_diff < beta_val,
        0.5 * diff * diff / beta_val,
        abs_diff - 0.5 * beta_val
    )


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


def smooth_l1_loss(inp, target, reduction=Reduction.MEAN.value, beta=1.0):
    logger.debug("GEMS SMOOTH_L1_LOSS")
    if reduction == Reduction.NONE.value:
        return func(inp, target, beta=beta)

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
        kernel_1[(mid_size, 1, 1)](inp, target, mid, M, block_size, reduction, beta)
        kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out