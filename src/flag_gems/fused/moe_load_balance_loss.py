# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Top-K MoE load-balancing auxiliary loss.

This module computes the load-balancing auxiliary loss used to prevent expert
collapse in Mixture-of-Experts (MoE) models. The formula is

.. math::
    L_{ib} = N_e \\sum_i f_i P_i,

where ``f_i`` is expert ``i``'s Top-K assignment frequency and ``P_i`` is
its mean routing probability. Both statistics are computed in fp32 from the
provided router logits so the result is dtype-independent at the API
boundary. When ``attention_mask`` is supplied, zero entries are excluded
from both statistics and the denominator.

The implementation is a two-pass Triton pipeline: the first pass performs
TOP_K sequential argmax selections and accumulates expert counts via
``atomic_add``; the second pass re-reads the logits, computes the softmax
probabilities, and dot-products them against the accumulated counts to
produce the scalar loss.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


# Keep the token tile at one full accelerator vector. Shorter vector atomics
# are padded to 32 lanes on some backends which can otherwise over-count expert
# assignments. A 32-token tile also stays within the UB budget for the practical
# MoE expert counts targeted by this operator.
_BLOCK_TOKENS = 32


@triton.jit
def _topk_count_kernel(
    gate_logits,
    attention_mask,
    expert_counts,
    valid_token_count,
    num_tokens,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    token_offsets = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    expert_offsets = tl.arange(0, BLOCK_EXPERTS)
    valid_tokens = token_offsets < num_tokens

    if HAS_MASK:
        mask_values = tl.load(
            attention_mask + token_offsets,
            mask=valid_tokens,
            other=0,
        )
        valid_tokens = valid_tokens & (mask_values != 0)

    logits_mask = valid_tokens[:, None] & (expert_offsets[None, :] < NUM_EXPERTS)
    logits = tl.load(
        gate_logits + token_offsets[:, None] * NUM_EXPERTS + expert_offsets[None, :],
        mask=logits_mask,
        other=-float("inf"),
    ).to(tl.float32)

    # Softmax preserves ordering, so Top-K(logits) equals Top-K(softmax(logits)).
    for _ in tl.static_range(0, TOP_K):
        selected = tl.argmax(logits, axis=1)
        tl.atomic_add(
            expert_counts + selected,
            1.0,
            mask=valid_tokens,
        )
        logits = tl.where(
            expert_offsets[None, :] == selected[:, None],
            -float("inf"),
            logits,
        )

    local_valid_count = tl.sum(valid_tokens.to(tl.float32), axis=0)
    tl.atomic_add(valid_token_count, local_valid_count)


@triton.jit
def _loss_kernel(
    gate_logits,
    attention_mask,
    expert_counts,
    valid_token_count,
    output,
    num_tokens,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    token_offsets = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    expert_offsets = tl.arange(0, BLOCK_EXPERTS)
    valid_tokens = token_offsets < num_tokens

    if HAS_MASK:
        mask_values = tl.load(
            attention_mask + token_offsets,
            mask=valid_tokens,
            other=0,
        )
        valid_tokens = valid_tokens & (mask_values != 0)

    logits_mask = valid_tokens[:, None] & (expert_offsets[None, :] < NUM_EXPERTS)
    logits = tl.load(
        gate_logits + token_offsets[:, None] * NUM_EXPERTS + expert_offsets[None, :],
        mask=logits_mask,
        other=-float("inf"),
    ).to(tl.float32)
    logits = logits - tl.max(logits, axis=1)[:, None]
    numerators = tl.exp(logits)
    probabilities = numerators / tl.sum(numerators, axis=1)[:, None]

    counts = tl.load(
        expert_counts + expert_offsets,
        mask=expert_offsets < NUM_EXPERTS,
        other=0.0,
    )
    token_contributions = tl.sum(probabilities * counts[None, :], axis=1)

    total_valid = tl.load(valid_token_count)
    denominator = tl.maximum(total_valid * total_valid, 1.0)
    token_contributions = tl.where(
        valid_tokens,
        token_contributions * NUM_EXPERTS / denominator,
        0.0,
    )
    tl.atomic_add(output, tl.sum(token_contributions, axis=0))


def _validate_inputs(
    gate_logits: torch.Tensor,
    top_k: int,
    attention_mask: torch.Tensor | None,
) -> None:
    if not isinstance(gate_logits, torch.Tensor):
        raise TypeError("gate_logits must be a torch.Tensor")
    if gate_logits.ndim != 2:
        raise ValueError(
            "gate_logits must have shape [T, N_e], "
            f"but got {tuple(gate_logits.shape)}"
        )
    if gate_logits.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise TypeError("gate_logits supports only float16, bfloat16, and float32")

    num_tokens, num_experts = gate_logits.shape
    if num_tokens == 0 or num_experts == 0:
        raise ValueError("T and N_e must be greater than zero")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be a Python int")
    if not 1 <= top_k <= num_experts:
        raise ValueError("top_k must satisfy 1 <= top_k <= N_e")

    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError("attention_mask must be a torch.Tensor or None")
        if attention_mask.numel() != num_tokens:
            raise ValueError(
                "attention_mask must contain T elements, "
                f"but got {attention_mask.numel()} for T={num_tokens}"
            )
        if attention_mask.dtype not in (
            torch.bool,
            torch.uint8,
            torch.int32,
            torch.int64,
        ):
            raise TypeError(
                "attention_mask supports only bool, uint8, int32, and int64"
            )
        if attention_mask.device != gate_logits.device:
            raise ValueError(
                "gate_logits and attention_mask must be on the same device"
            )


def moe_load_balance_loss(
    gate_logits: torch.Tensor,
    top_k: int = 2,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Compute the Top-K MoE load-balancing auxiliary loss.

    The operator computes

    .. math::
        L_{ib}=N_e\sum_i f_iP_i,

    where ``f_i`` is expert ``i``'s Top-K assignment frequency and ``P_i`` is
    its mean routing probability. Top-K selection is computed internally from
    ``gate_logits``. If ``attention_mask`` is given, zero entries are excluded
    from both statistics.

    Args:
        gate_logits: Raw router logits with shape ``[T, N_e]`` and dtype
            float16, bfloat16, or float32.
        top_k: Number of experts selected per token.
        attention_mask: Optional mask containing ``T`` elements. Nonzero entries
            denote valid tokens.

    Returns:
        A float32 scalar tensor containing ``L_ib``. An all-zero mask returns
        zero. This implementation provides forward computation only.
    """
    _validate_inputs(gate_logits, top_k, attention_mask)
    logger.debug("GEMS MOE LOAD BALANCE LOSS FORWARD")

    logits = gate_logits.contiguous()
    # When no mask is supplied we still need a dummy pointer for the kernel
    # signature. Reuse ``logits`` so we avoid any extra allocation; the kernel
    # never reads it when ``HAS_MASK`` is False.
    flat_mask = (
        logits if attention_mask is None else attention_mask.reshape(-1).contiguous()
    )
    has_mask = attention_mask is not None
    num_tokens, num_experts = logits.shape

    # Keep atomic targets in separate allocations. Some backends pad vector
    # atomic stores to accelerator lanes, so adjacent slices may alias.
    expert_counts = torch.zeros(
        num_experts,
        dtype=torch.float32,
        device=logits.device,
    )
    valid_token_count = torch.zeros((), dtype=torch.float32, device=logits.device)
    output = torch.zeros((), dtype=torch.float32, device=logits.device)

    block_experts = triton.next_power_of_2(num_experts)
    block_tokens = _BLOCK_TOKENS
    num_warps = 4 if block_experts <= 256 else 8
    grid = (triton.cdiv(num_tokens, block_tokens),)

    with torch_device_fn.device(logits.device):
        _topk_count_kernel[grid](
            logits,
            flat_mask,
            expert_counts,
            valid_token_count,
            num_tokens,
            NUM_EXPERTS=num_experts,
            TOP_K=top_k,
            BLOCK_TOKENS=block_tokens,
            BLOCK_EXPERTS=block_experts,
            HAS_MASK=has_mask,
            num_warps=num_warps,
        )
        _loss_kernel[grid](
            logits,
            flat_mask,
            expert_counts,
            valid_token_count,
            output,
            num_tokens,
            NUM_EXPERTS=num_experts,
            BLOCK_TOKENS=block_tokens,
            BLOCK_EXPERTS=block_experts,
            HAS_MASK=has_mask,
            num_warps=num_warps,
        )

    return output


__all__ = ["moe_load_balance_loss"]
