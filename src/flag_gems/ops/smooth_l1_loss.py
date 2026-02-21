"""Smooth L1 Loss operator implementation."""
import logging

import torch
import triton
import triton.language as tl

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
    """
    Compute smooth L1 loss (Huber loss).

    For each element:
    - If |input - target| < beta: loss = 0.5 * (input - target)^2 / beta
    - Otherwise: loss = |input - target| - 0.5 * beta
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input and target
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    target_val = tl.load(target_ptr + offsets, mask=mask, other=0.0)

    # Compute difference
    diff = input_val - target_val
    abs_diff = tl.abs(diff)

    # Smooth L1 loss formula
    # When beta > 0:
    #   if |diff| < beta: 0.5 * diff^2 / beta
    #   else: |diff| - 0.5 * beta
    # When beta == 0: L1 loss = |diff|
    if beta > 0:
        loss = tl.where(
            abs_diff < beta, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta
        )
    else:
        loss = abs_diff

    # Store result
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
    """
    Compute gradient of smooth L1 loss.

    For each element:
    - If |input - target| < beta: grad = (input - target) / beta
    - Otherwise: grad = sign(input - target)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values
    grad_out = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    target_val = tl.load(target_ptr + offsets, mask=mask, other=0.0)

    # Compute difference
    diff = input_val - target_val
    abs_diff = tl.abs(diff)

    # Gradient formula
    # When beta > 0:
    #   if |diff| < beta: grad = diff / beta
    #   else: grad = sign(diff)
    # When beta == 0: grad = sign(diff) (L1 loss gradient)
    if beta > 0:
        grad = tl.where(abs_diff < beta, diff / beta, tl.where(diff > 0, 1.0, -1.0))
    else:
        grad = tl.where(diff > 0, 1.0, tl.where(diff < 0, -1.0, 0.0))

    # Apply chain rule with grad_output
    grad_input = grad * grad_out

    # Store result
    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)


def smooth_l1_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Compute smooth L1 loss (Huber loss).

    Args:
        input: Input tensor of any shape
        target: Target tensor of same shape as input
        reduction: Specifies the reduction to apply to the output:
            'none': no reduction will be applied
            'mean': the sum of the output will be divided by the number of elements
            'sum': the output will be summed
        beta: Specifies the threshold at which to change between L1 and L2 loss.
              Default: 1.0

    Returns:
        Loss tensor. If reduction is 'none', same shape as input.
        Otherwise, scalar tensor.

    Formula:
        loss(x, y) = 0.5 * (x - y)^2 / beta,  if |x - y| < beta
                     |x - y| - 0.5 * beta,     otherwise
    """
    logger.debug("GEMS SMOOTH_L1_LOSS FORWARD")

    # Validate inputs
    if reduction not in ["none", "mean", "sum"]:
        raise ValueError(
            f"reduction must be 'none', 'mean', or 'sum'. Got: {reduction}"
        )

    if beta < 0:
        raise ValueError(f"beta must be non-negative. Got: {beta}")

    # Broadcast target to match input shape if needed
    if input.shape != target.shape:
        target = torch.broadcast_to(target, input.shape)

    # Ensure contiguous
    input = input.contiguous()
    target = target.contiguous()

    # Allocate output
    output = torch.empty_like(input)

    # Get total number of elements
    n_elements = input.numel()

    if n_elements == 0:
        # Handle empty tensor
        if reduction == "none":
            return output
        else:
            return torch.tensor(0.0, device=input.device, dtype=input.dtype)

    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    smooth_l1_loss_forward_kernel[grid](
        input,
        target,
        output,
        n_elements,
        beta,
    )

    # Apply reduction
    if reduction == "mean":
        return output.mean()
    elif reduction == "sum":
        return output.sum()
    else:  # reduction == "none"
        return output


def smooth_l1_loss_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Compute gradient of smooth L1 loss with respect to input.

    Args:
        grad_output: Gradient of loss with respect to output
        input: Input tensor
        target: Target tensor
        reduction: Reduction mode used in forward pass
        beta: Beta parameter used in forward pass

    Returns:
        Gradient with respect to input
    """
    logger.debug("GEMS SMOOTH_L1_LOSS BACKWARD")

    # Broadcast target to match input shape if needed
    if input.shape != target.shape:
        target = torch.broadcast_to(target, input.shape)

    # Ensure contiguous
    input = input.contiguous()
    target = target.contiguous()

    # Allocate gradient
    grad_input = torch.empty_like(input)

    n_elements = input.numel()

    if n_elements == 0:
        return grad_input

    # Expand grad_output if needed (for mean/sum reduction)
    if grad_output.numel() == 1:
        # Scalar gradient from reduction
        grad_output_expanded = grad_output.expand_as(input).contiguous()

        # Adjust for reduction type
        if reduction == "mean":
            # For mean reduction, gradient is divided by number of elements
            grad_output_expanded = grad_output_expanded / n_elements
    else:
        # No reduction or already expanded
        grad_output_expanded = grad_output.contiguous()

    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    smooth_l1_loss_backward_kernel[grid](
        grad_output_expanded,
        input,
        target,
        grad_input,
        n_elements,
        beta,
    )

    return grad_input


class SmoothL1Loss(torch.autograd.Function):
    """Autograd function for smooth L1 loss."""

    @staticmethod
    def forward(ctx, input, target, reduction, beta):
        """Forward pass."""
        ctx.save_for_backward(input, target)
        ctx.reduction = reduction
        ctx.beta = beta
        return smooth_l1_loss(input, target, reduction, beta)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass."""
        input, target = ctx.saved_tensors
        grad_input = smooth_l1_loss_backward(
            grad_output, input, target, ctx.reduction, ctx.beta
        )
        return grad_input, None, None, None
