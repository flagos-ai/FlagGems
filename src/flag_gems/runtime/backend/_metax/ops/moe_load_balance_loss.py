import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


@libentry()
@triton.jit
def moe_load_balance_loss_kernel(
    gates_ptr,
    output_ptr,
    num_tokens,
    num_experts,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel to compute MoE load balancing loss.

    Computes: loss = sum(gates^2) * num_experts / num_tokens

    Each program processes a block of tokens.
    """
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_tokens

    # Accumulator for this block - use tl.zeros with tuple shape
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over experts in chunks
    for expert_offset in range(0, num_experts, BLOCK_SIZE):
        expert_offs = expert_offset + tl.arange(0, BLOCK_SIZE)
        expert_mask = expert_offs < num_experts

        # Load gates for this block of tokens and expert chunk
        # gates[token, expert]
        gate_offsets = offs[:, None] * num_experts + expert_offs[None, :]
        gates = tl.load(
            gates_ptr + gate_offsets,
            mask=(mask[:, None] & expert_mask[None, :]),
            other=0.0,
        ).to(tl.float32)

        # Accumulate squares: sum over experts dimension
        acc += tl.sum(gates * gates, axis=1)

    # Store partial sum for this block
    tl.store(output_ptr + offs, acc, mask=mask)


def moe_load_balance_loss(gates: torch.Tensor) -> torch.Tensor:
    """Compute the auxiliary load balancing loss for MoE.

    This loss encourages an even distribution of tokens across experts.
    The formula (from ST-MoE, DeepSeek) is:
        loss = sum(f_i * P_i) * num_experts

    For soft routing with softmax probabilities:
        loss = sum(probs^2) * num_experts / num_tokens

    This is minimized when all experts have equal probability mass.

    Args:
        gates: Tensor of shape (num_tokens, num_experts) containing routing
               probabilities (softmax outputs from router).

    Returns:
        Scalar tensor containing the load balancing loss.
    """
    logger.debug("METAX GEMS MOE_LOAD_BALANCE_LOSS")

    num_tokens, num_experts = gates.shape
    gates = gates.contiguous()

    # For small cases, use simple PyTorch computation for efficiency
    if num_tokens * num_experts <= 4096:
        loss = (gates.float() ** 2).sum() * num_experts / num_tokens
        return loss

    # Use Triton kernel for larger cases
    BLOCK_SIZE = min(triton.next_power_of_2(num_tokens), 64)
    num_blocks = triton.cdiv(num_tokens, BLOCK_SIZE)

    output = torch.empty((num_tokens,), dtype=torch.float32, device=gates.device)

    grid = lambda meta: (num_blocks,)
    with torch_device_fn.device(gates.device):
        moe_load_balance_loss_kernel[grid](
            gates,
            output,
            num_tokens,
            num_experts,
            BLOCK_SIZE,
        )

    # Sum all partial results and scale
    partial_sum = output.sum()
    loss = partial_sum * num_experts / num_tokens
    return loss
