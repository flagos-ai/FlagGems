import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Reduction mode constants (matching PyTorch)
_NONE = 0
_MEAN = 1
_SUM = 2


@triton.jit
def _smooth_l1_kernel(
    input_ptr,
    target_ptr,
    out_ptr,
    numel,
    beta,
    BLOCK: tl.constexpr,
):
    """Element-wise smooth L1 (Huber) loss kernel."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(target_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    diff = tl.abs(x - y)
    # Smooth L1: 0.5*diff^2/beta if diff < beta, else diff - 0.5*beta
    loss = tl.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    tl.store(out_ptr + offsets, loss, mask=mask)


def smooth_l1_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: int = _MEAN,
    beta: float = 1.0,
) -> torch.Tensor:
    """Smooth L1 (Huber) loss.

    Args:
        input: Predictions tensor.
        target: Targets tensor, same shape as input.
        reduction: 0=none, 1=mean, 2=sum.
        beta: Threshold for switching between L1 and L2 behaviour.

    Returns:
        Scalar loss (or per-element tensor if reduction=none).
    """
    logger.debug("GEMS SMOOTH_L1_LOSS")
    assert input.shape == target.shape, "input and target must have the same shape"

    inp = input.contiguous()
    tgt = target.contiguous()
    numel = inp.numel()

    out = torch.empty(numel, dtype=torch.float32, device=input.device)

    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)

    with torch_device_fn.device(input.device):
        _smooth_l1_kernel[grid](
            inp.view(-1),
            tgt.view(-1),
            out,
            numel,
            beta,
            BLOCK=BLOCK,
        )

    out = out.view(input.shape).to(input.dtype)

    if reduction == _NONE:
        return out
    elif reduction == _MEAN:
        return out.mean()
    else:  # SUM
        return out.sum()
