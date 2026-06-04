import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


def _torch_moe_load_balance_loss(gate_logits: torch.Tensor) -> torch.Tensor:
    num_tokens, num_experts = gate_logits.shape
    softmax_probs = torch.softmax(gate_logits, dim=-1)
    expert_loads = torch.sum(softmax_probs, dim=0)
    expert_loads_normalized = expert_loads / num_tokens
    loss = num_experts * torch.sum(expert_loads_normalized**2)
    return loss.to(gate_logits.dtype)


@libentry()
@triton.jit
def MoELoadBalanceLoss_kernel(
    softmax_probs_ptr,
    expert_loads_ptr,
    num_tokens,
    num_experts,
    BLOCK_SIZE: tl.constexpr,
):
    """Accumulate expert loads from softmax probabilities.

    Each program processes a block of tokens and accumulates their
    softmax contributions to each expert via atomic additions.
    """
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_tokens

    for expert_idx in range(num_experts):
        probs = tl.load(
            softmax_probs_ptr + offset * num_experts + expert_idx,
            mask=mask,
            other=0.0,
        )
        expert_sum = tl.sum(probs)
        tl.atomic_add(expert_loads_ptr + expert_idx, expert_sum)


def MoELoadBalanceLoss(gate_logits: torch.Tensor) -> torch.Tensor:
    """Compute the auxiliary load balancing loss for MoE models.

    This loss encourages uniform usage of experts by penalizing
    imbalanced expert loads.

    Args:
        gate_logits: Tensor of shape (num_tokens, num_experts) containing
                     the raw logits from the gating mechanism.

    Returns:
        Scalar tensor containing the load balancing loss.

    Mathematical formula:
        loss = num_experts * sum((expert_load / num_tokens)^2)

        where expert_load[i] = sum_j(softmax(gate_logits)[j, i])
        and softmax is computed along the expert dimension.
    """
    logger.debug("METAX GEMS MoELoadBalanceLoss")

    assert gate_logits.ndim == 2, f"Expected 2D input, got {gate_logits.ndim}D"
    if gate_logits.device.type == "cpu":
        return _torch_moe_load_balance_loss(gate_logits)

    num_tokens, num_experts = gate_logits.shape
    gate_logits = gate_logits.contiguous()

    # Compute softmax probabilities from logits before kernel
    softmax_probs = torch.softmax(gate_logits, dim=-1)

    # Allocate output buffer for expert loads
    expert_loads = torch.zeros(
        num_experts, dtype=torch.float32, device=gate_logits.device
    )

    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(num_tokens, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(gate_logits.device):
        MoELoadBalanceLoss_kernel[grid](
            softmax_probs,
            expert_loads,
            num_tokens,
            num_experts,
            BLOCK_SIZE,
        )

    # Compute final loss: num_experts * sum((expert_load / num_tokens)^2)
    expert_loads_normalized = expert_loads / num_tokens
    loss = num_experts * torch.sum(expert_loads_normalized**2)

    return loss.to(gate_logits.dtype)
