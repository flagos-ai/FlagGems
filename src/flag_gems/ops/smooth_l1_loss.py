import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def smooth_l1_loss_forward_kernel(
    input_ptr,
    target_ptr,
    output_ptr,
    n_elements,
    beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    target_val = tl.load(target_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    diff = input_val - target_val
    abs_diff = tl.abs(diff)
    if beta > 0:
        loss = tl.where(
            abs_diff < beta, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta
        )
    else:
        loss = abs_diff
    tl.store(output_ptr + offsets, loss, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def smooth_l1_loss_backward_kernel(
    grad_output_ptr,
    input_ptr,
    target_ptr,
    grad_input_ptr,
    n_elements,
    beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    grad_out = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    target_val = tl.load(target_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    diff = input_val - target_val
    abs_diff = tl.abs(diff)
    if beta > 0:
        grad = tl.where(abs_diff < beta, diff / beta, tl.where(diff > 0, 1.0, -1.0))
    else:
        grad = tl.where(diff > 0, 1.0, tl.where(diff < 0, -1.0, 0.0))
    tl.store(grad_input_ptr + offsets, grad * grad_out, mask=mask)


def smooth_l1_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    beta: float = 1.0,
) -> torch.Tensor:
    logger.debug("GEMS SMOOTH_L1_LOSS FORWARD")
    if beta < 0:
        raise ValueError(f"beta must be non-negative. Got: {beta}")
    if input.shape != target.shape:
        target = torch.broadcast_to(target, input.shape)
    input = input.contiguous()
    target = target.contiguous()
    output = torch.empty_like(input)
    n_elements = input.numel()
    if n_elements == 0:
        if reduction == "none":
            return output
        return torch.tensor(0.0, device=input.device, dtype=input.dtype)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(input.device):
        smooth_l1_loss_forward_kernel[grid](input, target, output, n_elements, beta)
    if reduction == "mean":
        return output.mean()
    elif reduction == "sum":
        return output.sum()
    else:
        return output


def smooth_l1_loss_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    beta: float = 1.0,
) -> torch.Tensor:
    logger.debug("GEMS SMOOTH_L1_LOSS BACKWARD")
    if input.shape != target.shape:
        target = torch.broadcast_to(target, input.shape)
    input = input.contiguous()
    target = target.contiguous()
    grad_input = torch.empty_like(input)
    n_elements = input.numel()
    if n_elements == 0:
        return grad_input
    if grad_output.numel() == 1:
        grad_output_expanded = grad_output.expand_as(input).contiguous()
        if reduction == "mean":
            grad_output_expanded = grad_output_expanded / n_elements
    else:
        grad_output_expanded = grad_output.contiguous()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(input.device):
        smooth_l1_loss_backward_kernel[grid](
            grad_output_expanded, input, target, grad_input, n_elements, beta
        )
    return grad_input
