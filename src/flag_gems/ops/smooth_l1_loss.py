import logging
import math

import torch
import triton
import triton.language as tl

import flag_gems

log = logging.getLogger(__name__)


@triton.jit
def _smooth_l1_loss_none_kernel(
    input_ptr,
    target_ptr,
    out_ptr,
    n_elements,
    beta,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise smooth L1 loss kernel for reduction='none'.

    Computes all intermediate values in float32 for accuracy.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    inp = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

    inp_f = inp.to(tl.float32)
    target_f = target.to(tl.float32)
    diff = inp_f - target_f
    abs_diff = tl.abs(diff)

    half_beta = 0.5 * beta
    loss = tl.where(abs_diff < beta, 0.5 * diff * diff / beta, abs_diff - half_beta)

    tl.store(out_ptr + offsets, loss, mask=mask)


@triton.jit
def _smooth_l1_loss_reduce_kernel(
    input_ptr,
    target_ptr,
    mid_ptr,
    n_elements,
    beta,
    BLOCK_SIZE: tl.constexpr,
):
    """First-stage reduction: each block computes partial sum to mid array."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    inp = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

    inp_f = inp.to(tl.float32)
    target_f = target.to(tl.float32)
    diff = inp_f - target_f
    abs_diff = tl.abs(diff)

    half_beta = 0.5 * beta
    vals = tl.where(abs_diff < beta, 0.5 * diff * diff / beta, abs_diff - half_beta)
    vals = tl.where(mask, vals, 0.0)

    acc = tl.sum(vals, axis=0)
    tl.store(mid_ptr + pid, acc)


@triton.jit
def _smooth_l1_loss_final_kernel(
    mid_ptr,
    out_ptr,
    mid_size,
    total_elements,
    BLOCK_MID: tl.constexpr,
):
    """Second-stage reduction: sum partial sums and apply mean if needed."""
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid_ptr + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(mid_val)
    # total_elements > 0 means mean reduction (divide by total_elements)
    # total_elements == 0 means sum reduction
    if total_elements > 0:
        result = total / total_elements
    else:
        result = total
    tl.store(out_ptr, result)


def _normalize_reduction(reduction):
    """Normalize reduction parameter to integer 0/1/2."""
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


def _check_tensors(input: torch.Tensor, target: torch.Tensor):
    """Validate that input and target tensors are on the same device."""
    if not (input.is_cuda and target.is_cuda):
        raise AssertionError("smooth_l1_loss: input and target must be CUDA tensors.")
    if input.device != target.device:
        raise AssertionError(
            "smooth_l1_loss: input and target must be on the same device."
        )
    if input.numel() != target.numel():
        raise AssertionError(
            "smooth_l1_loss: input and target must have the same number of elements."
        )
    if not input.is_contiguous():
        input = input.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()
    return input, target


def smooth_l1_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: int = 1,
    beta: float = 1.0,
):
    """Compute smooth L1 loss (Huber loss) using Triton kernels.

    Args:
        input: Predicted tensor of any shape.
        target: Target tensor, same shape as input.
        reduction: Specifies the reduction to apply:
            0/'none': no reduction
            1/'mean': mean of all elements
            2/'sum': sum of all elements
        beta: Threshold at which to change between L1 and L2 loss.

    Returns:
        Loss tensor. Shape depends on reduction mode.
    """
    log.debug("GEMS SMOOTH_L1_LOSS")

    # Handle device fallback for non-CUDA tensors
    device = input.device
    if not (isinstance(device, torch.device) and device.type == flag_gems.device):
        return torch.ops.aten.smooth_l1_loss(input, target, reduction, beta)

    input, target = _check_tensors(input, target)
    red = _normalize_reduction(reduction)
    n_elements = input.numel()

    if red == 0:
        # reduction = 'none'
        out = torch.empty_like(input)
        if n_elements == 0:
            return out
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _smooth_l1_loss_none_kernel[grid](
            input, target, out, n_elements, beta, BLOCK_SIZE=BLOCK_SIZE
        )
        return out

    dtype = input.dtype

    if n_elements == 0:
        # Follow PyTorch behavior: sum -> 0, mean -> NaN
        if red == 2:
            return torch.zeros((), device=input.device, dtype=dtype)
        else:
            return torch.full((), float("nan"), device=input.device, dtype=dtype)

    # Two-stage reduction: partial sums -> mid array -> final sum
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.float32, device=input.device)
    out = torch.empty([], dtype=dtype, device=input.device)

    _smooth_l1_loss_reduce_kernel[(mid_size, 1, 1)](
        input, target, mid, n_elements, beta, BLOCK_SIZE=block_size
    )

    _smooth_l1_loss_final_kernel[(1, 1, 1)](
        mid,
        out,
        mid_size,
        n_elements if red == 1 else 0,
        BLOCK_MID=block_mid,
    )

    return out


def smooth_l1_loss_out(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: int = 1,
    beta: float = 1.0,
    *,
    out: torch.Tensor,
):
    log.debug("GEMS SMOOTH_L1_LOSS OUT")
    device = input.device
    if not (isinstance(device, torch.device) and device.type == flag_gems.device):
        result = torch.ops.aten.smooth_l1_loss(input, target, reduction, beta)
        out.copy_(result)
        return out

    input, target = _check_tensors(input, target)
    red = _normalize_reduction(reduction)
    n_elements = input.numel()
    dtype = input.dtype

    if red == 0:
        if n_elements == 0:
            return out
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _smooth_l1_loss_none_kernel[grid](
            input, target, out, n_elements, beta, BLOCK_SIZE=BLOCK_SIZE
        )
        return out

    if n_elements == 0:
        if red == 2:
            out.copy_(torch.zeros((), device=input.device, dtype=dtype))
        else:
            out.copy_(torch.full((), float("nan"), device=input.device, dtype=dtype))
        return out

    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)
    mid = torch.empty((mid_size,), dtype=torch.float32, device=input.device)

    _smooth_l1_loss_reduce_kernel[(mid_size, 1, 1)](
        input, target, mid, n_elements, beta, BLOCK_SIZE=block_size
    )
    _smooth_l1_loss_final_kernel[(1, 1, 1)](
        mid, out, mid_size, n_elements if red == 1 else 0, BLOCK_MID=block_mid
    )
    return out
