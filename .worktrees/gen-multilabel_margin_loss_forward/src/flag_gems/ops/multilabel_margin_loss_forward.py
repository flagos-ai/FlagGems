import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _multilabel_margin_loss_kernel(
    input_ptr,
    target_ptr,
    output_ptr,
    N,
    C,
    stride_in_n,
    stride_in_c,
    stride_tgt_n,
    stride_tgt_c,
    REDUCTION: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Compute multilabel margin loss.

    For each sample n:
    - Find the end_idx (first position where target[n] == 0, or C if none)
    - Valid targets are target[n, 0:end_idx)
    - For each j in [0, end_idx):
        For each i in [0, C), i != target[n,j]:
            loss += max(0, 1 - (input[n, target[n,j]] - input[n, i]))
    """
    pid_n = tl.program_id(0)

    # Find first zero position using a loop
    first_zero_pos = C
    for c in range(C):
        tgt_val = tl.load(target_ptr + pid_n * stride_tgt_n + c * stride_tgt_c)
        first_zero_pos = tl.where((first_zero_pos == C) & (tgt_val == 0), c, first_zero_pos)

    # Compute loss for this sample
    loss = 0.0

    # For each target position j
    for j in range(C):
        is_valid_pos_j = j < first_zero_pos
        target_class_j = tl.load(target_ptr + pid_n * stride_tgt_n + j * stride_tgt_c).to(tl.int32)
        x_yj = tl.load(input_ptr + pid_n * stride_in_n + target_class_j * stride_in_c)

        # For each position i
        for i in range(C):
            is_different_idx = i != target_class_j
            should_add = is_valid_pos_j & is_different_idx
            x_i = tl.load(input_ptr + pid_n * stride_in_n + i * stride_in_c)
            margin = 1.0 - (x_yj - x_i)
            loss = loss + tl.where(should_add, tl.maximum(margin, 0.0), 0.0)

    if REDUCTION == 1:
        loss = loss / C

    if REDUCTION == 0:
        out_ptr = output_ptr + pid_n
        tl.store(out_ptr, loss)
    else:
        if pid_n == 0:
            tl.store(output_ptr, loss)


def multilabel_margin_loss_forward(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: int = 1,
) -> tuple:
    """
    Compute multilabel margin loss forward.

    Args:
        input: Input tensor of shape (N, C) where N is batch size, C is number of classes
        target: Target tensor of shape (N, C) containing class indices in [0, C-1] or 0
        reduction: 0='none', 1='mean', 2='sum'

    Returns:
        output: Loss value (scalar if reduction != 0, else tensor of shape (N,))
        is_target: Tensor of shape (N, C) indicating valid target positions
    """
    logger.debug("GEMS MULTILABEL_MARGIN_LOSS_FORWARD")

    assert input.dim() == 2, f"Expected 2D input, got {input.dim()}D"
    assert target.dim() == 2, f"Expected 2D target, got {target.dim()}D"
    assert input.shape == target.shape, f"Input and target shape mismatch: {input.shape} vs {target.shape}"

    N, C = input.shape

    if reduction == 0:
        output = torch.empty((N,), dtype=input.dtype, device=input.device)
    else:
        output = torch.empty([], dtype=input.dtype, device=input.device)

    # Compute is_target
    is_target = torch.zeros_like(target, dtype=torch.float32)
    for n in range(N):
        found_zero = False
        for c in range(C):
            if not found_zero:
                if target[n, c] == 0:
                    found_zero = True
                is_target[n, c] = 1.0

    BLOCK_C = triton.next_power_of_2(C)
    grid = lambda meta: (N,)

    with torch_device_fn.device(input.device):
        _multilabel_margin_loss_kernel[grid](
            input,
            target,
            output,
            N,
            C,
            input.stride(0),
            input.stride(1),
            target.stride(0),
            target.stride(1),
            REDUCTION=reduction,
            BLOCK_C=BLOCK_C,
        )

    if reduction == 2:
        output = output.sum()
    elif reduction == 1:
        output = output.sum() / N

    return output, is_target