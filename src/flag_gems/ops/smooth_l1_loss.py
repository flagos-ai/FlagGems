"""smooth_l1_loss forward + backward — FlagGems operator.

Forward:
    loss_i = 0.5*(x_i - y_i)^2 / beta        if |x_i - y_i| < beta
            |x_i - y_i| - 0.5*beta           otherwise
    Reductions: 0=none, 1=mean, 2=sum.

Backward (registered separately as ``smooth_l1_loss_backward``):
    d/dx loss_i = (x_i - y_i) / beta         if |x_i - y_i| < beta
                  sign(x_i - y_i)            otherwise
    d/dy loss_i = -d/dx loss_i

    For reduction == none:  d_input_i = grad_output_i * d/dx loss_i
    For reduction == mean:  d_input_i = grad_output * d/dx loss_i / N
    For reduction == sum:   d_input_i = grad_output * d/dx loss_i
"""

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
# Forward
# ---------------------------------------------------------------------------
# Kept for completeness — ``pointwise_dynamic`` JIT compiles a polymorphic
# variant that supports broadcasting and odd strides, but carries a
# multi-hundred-microsecond fixed dispatch cost.  The direct kernel below
# is used when the inputs are contiguous (the common case).
@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def smooth_l1_loss_none_kernel(inp, target, beta):
    x = inp.to(tl.float32)
    y = target.to(tl.float32)
    diff = tl.abs(x - y)
    return tl.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)


@libentry()
@triton.jit
def smooth_l1_loss_none_direct_kernel(
    inp_ptr,
    target_ptr,
    out_ptr,
    M,
    beta,
    BLOCK_SIZE: tl.constexpr,
):
    """Direct elementwise kernel for ``reduction='none'`` on contiguous inputs.

    Skips the ``pointwise_dynamic`` polymorphic dispatch (which carries a
    multi-hundred-microsecond fixed cost on Triton 3.6).  Two- to twentyfold
    faster on small (≤ 1k element) inputs where launch overhead dominates.
    """
    pid = tle.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < M
    x = tl.load(inp_ptr + offs, mask=mask, other=0).to(tl.float32)
    y = tl.load(target_ptr + offs, mask=mask, other=0).to(tl.float32)
    diff = tl.abs(x - y)
    loss = tl.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    tl.store(out_ptr + offs, loss.to(out_ptr.dtype.element_ty), mask=mask)


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

    # Apply masked-zero contribution to the per-block partial sum.
    loss = tl.where(mask, loss, 0.0)

    if reduction == 1:  # mean
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


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def smooth_l1_loss_backward_none_kernel(
    grad_output,
    inp,
    target,
    grad_input,
    M,
    beta,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduction == none: grad_output is element-wise; output is per-element."""
    pid = tle.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < M

    go = tl.load(grad_output + offs, mask=mask, other=0).to(tl.float32)
    x = tl.load(inp + offs, mask=mask, other=0).to(tl.float32)
    y = tl.load(target + offs, mask=mask, other=0).to(tl.float32)
    diff = x - y
    abs_diff = tl.abs(diff)
    grad = tl.where(abs_diff < beta, diff / beta, tl.where(diff > 0, 1.0, -1.0))
    out = grad * go
    tl.store(grad_input + offs, out.to(grad_input.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def smooth_l1_loss_backward_reduced_kernel(
    grad_output_scalar_ptr,
    inp,
    target,
    grad_input,
    M,
    beta,
    inv_N,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduction == mean (inv_N=1/M) or sum (inv_N=1.0): grad_output is scalar."""
    pid = tle.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < M

    go = tl.load(grad_output_scalar_ptr).to(tl.float32)
    x = tl.load(inp + offs, mask=mask, other=0).to(tl.float32)
    y = tl.load(target + offs, mask=mask, other=0).to(tl.float32)
    diff = x - y
    abs_diff = tl.abs(diff)
    grad = tl.where(abs_diff < beta, diff / beta, tl.where(diff > 0, 1.0, -1.0))
    out = grad * go * inv_N
    tl.store(grad_input + offs, out.to(grad_input.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------
def _resolve_reduction(reduction):
    """Accept either int (aten enum) or str ('none' | 'mean' | 'sum')."""
    if isinstance(reduction, str):
        return {"none": 0, "mean": 1, "sum": 2}[reduction]
    return int(reduction)


def smooth_l1_loss(inp, target, reduction=Reduction.MEAN.value, beta=1.0):
    logger.debug("GEMS SMOOTH_L1_LOSS")
    reduction = _resolve_reduction(reduction)
    beta = float(beta)
    assert beta > 0, f"beta must be positive, got {beta}"

    if reduction == Reduction.NONE.value:
        # Fast path: direct kernel on contiguous inputs avoids the
        # ``pointwise_dynamic`` dispatch overhead.  Falls back to the
        # polymorphic kernel when the inputs aren't contiguous (rare in
        # production) or shapes differ (broadcasting case).
        if (
            inp.is_contiguous()
            and target.is_contiguous()
            and inp.shape == target.shape
            and inp.is_cuda
            and inp.numel() > 0
        ):
            out = torch.empty_like(inp)
            M = inp.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(M, BLOCK_SIZE),)
            with torch_device_fn.device(inp.device):
                smooth_l1_loss_none_direct_kernel[grid](
                    inp,
                    target,
                    out,
                    M,
                    beta,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
            return out
        return smooth_l1_loss_none_kernel(inp, target, beta)

    inp = inp.contiguous()
    target = target.contiguous()
    M = inp.numel()
    dtype = inp.dtype

    if M == 0:
        # Match torch: mean over empty is NaN (0/0); sum is 0.
        if reduction == Reduction.MEAN.value:
            return torch.full([], float("nan"), dtype=dtype, device=inp.device)
        return torch.zeros([], dtype=dtype, device=inp.device)

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


def smooth_l1_loss_backward(
    grad_output, inp, target, reduction=Reduction.MEAN.value, beta=1.0
):
    """Compute gradient of smooth_l1_loss with respect to ``inp``.

    The gradient w.r.t. ``target`` is the negation of this; PyTorch's
    autograd graph routes that automatically when ``target`` requires grad.
    """
    logger.debug("GEMS SMOOTH_L1_LOSS_BACKWARD")
    reduction = _resolve_reduction(reduction)
    beta = float(beta)

    inp = inp.contiguous()
    target = target.contiguous()
    grad_output = grad_output.contiguous()
    grad_input = torch.empty_like(inp)
    M = inp.numel()
    if M == 0:
        return grad_input

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(M, BLOCK_SIZE),)

    with torch_device_fn.device(inp.device):
        if reduction == Reduction.NONE.value:
            smooth_l1_loss_backward_none_kernel[grid](
                grad_output,
                inp,
                target,
                grad_input,
                M,
                beta,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            inv_N = 1.0 / M if reduction == Reduction.MEAN.value else 1.0
            smooth_l1_loss_backward_reduced_kernel[grid](
                grad_output,
                inp,
                target,
                grad_input,
                M,
                beta,
                inv_N,
                BLOCK_SIZE=BLOCK_SIZE,
            )
    return grad_input
