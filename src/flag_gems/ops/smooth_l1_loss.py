import logging
from enum import Enum

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

MAX_BLOCK_SIZE = 8192


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


# Keep pointwise_dynamic for non-contiguous / broadcast cases
@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def smooth_l1_loss_none_func(x, y, beta):
    diff = x.to(tl.float32) - y.to(tl.float32)
    ad = tl.abs(diff)
    loss = tl.where(
        beta == 0.0,
        ad,
        tl.where(ad < beta, 0.5 * diff * diff / beta, ad - 0.5 * beta),
    )
    return loss


# Optimized kernel for contiguous none reduction with autotune
@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16),
    ],
    key=["M"],
)
@triton.jit
def smooth_l1_loss_none_kernel(inp, target, out, M, beta, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M

    x = tl.load(inp + offset, mask=mask, other=0).to(tl.float32)
    y = tl.load(target + offset, mask=mask, other=0).to(tl.float32)

    diff = x - y
    ad = tl.abs(diff)
    loss = tl.where(
        beta == 0.0,
        ad,
        tl.where(ad < beta, 0.5 * diff * diff / beta, ad - 0.5 * beta),
    )

    tl.store(out + offset, loss, mask=mask)


# Partial sum kernel for mean/sum reductions
@libentry()
@triton.jit
def smooth_l1_loss_reduce_kernel(
    inp,
    target,
    mid,
    M,
    beta,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M

    inp_val = tl.load(inp + offset, mask=mask, other=0).to(tl.float32)
    target_val = tl.load(target + offset, mask=mask, other=0).to(tl.float32)

    diff = inp_val - target_val
    ad = tl.abs(diff)
    loss = tl.where(
        beta == 0.0,
        ad,
        tl.where(ad < beta, 0.5 * diff * diff / beta, ad - 0.5 * beta),
    )

    sum_val = tl.sum(loss)
    tl.store(mid + pid, sum_val)


@libentry()
@triton.jit
def smooth_l1_loss_final_reduce(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    mid_val = tl.load(mid + offset, mask=mask, other=0).to(tl.float32)
    sum_val = tl.sum(mid_val)
    tl.store(out, sum_val)


def smooth_l1_loss(inp, target, reduction=Reduction.MEAN.value, beta=1.0):
    logger.debug("GEMS SMOOTH_L1_LOSS")

    if reduction == Reduction.NONE.value:
        # Fast path for contiguous same-shape tensors
        if inp.is_contiguous() and target.is_contiguous() and inp.shape == target.shape:
            M = inp.numel()
            out = torch.empty_like(inp)
            if M > 0:
                grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE"]),)
                with torch_device_fn.device(inp.device):
                    smooth_l1_loss_none_kernel[grid](inp, target, out, M, beta)
            return out
        # Fallback for non-contiguous / broadcast
        return smooth_l1_loss_none_func(inp, target, beta)

    inp = inp.contiguous()
    target = target.contiguous()
    M = inp.numel()
    dtype = inp.dtype

    if M == 0:
        if reduction == Reduction.MEAN.value:
            return torch.tensor(float("nan"), dtype=dtype, device=inp.device)
        else:
            return torch.tensor(0.0, dtype=dtype, device=inp.device)

    # Cap block_size to avoid register spilling with complex per-element logic
    block_size = min(
        triton.next_power_of_2(max(triton.cdiv(M, 65535), 1)),
        MAX_BLOCK_SIZE,
    )
    # Ensure block_size is at least 1024
    block_size = max(block_size, 1024)
    mid_size = triton.cdiv(M, block_size)

    mid = torch.empty((mid_size,), dtype=torch.float32, device=inp.device)

    with torch_device_fn.device(inp.device):
        smooth_l1_loss_reduce_kernel[(mid_size, 1, 1)](
            inp, target, mid, M, beta, block_size
        )

    # Final reduction: use Triton kernel for small mid_size, PyTorch for large
    # Keep intermediate sum in float32 to avoid overflow in low-precision dtypes
    if mid_size <= 65536:
        block_mid = triton.next_power_of_2(mid_size)
        out = torch.empty([], dtype=torch.float32, device=inp.device)
        with torch_device_fn.device(inp.device):
            smooth_l1_loss_final_reduce[(1, 1, 1)](mid, out, mid_size, block_mid)
        total = out
    else:
        total = mid.sum()

    if reduction == Reduction.MEAN.value:
        return (total / M).to(dtype)
    else:
        return total.to(dtype)
