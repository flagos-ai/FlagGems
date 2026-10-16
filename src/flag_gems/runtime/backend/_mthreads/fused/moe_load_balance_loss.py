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

"""Moore Threads (MUSA) specialization of the Top-K MoE load-balancing loss.

.. math::
    L_{ib} = N_e \\sum_i f_i P_i,

with ``f_i`` the Top-K assignment frequency of expert ``i`` and ``P_i`` its
mean routing probability, both computed in fp32 from the router logits and
restricted to the tokens selected by the optional ``attention_mask``.

The pipeline is the two-pass one (count, then loss) because the loss pass needs
the grid-wide expert histogram produced by the count pass.  Three choices are
MUSA-specific and were each measured on the MTT S5000:

1. **Relaxed atomics.**  ``tl.atomic_add`` defaults to ``sem="acq_rel"``, which
   on MUSA forces device-scope ordering around every atomic.  With one atomic
   per program the ordering dominates: for ``T=32768, N_e=128, K=8`` (fp16) the
   count pass falls from ``271 us`` to ``148 us`` when the two accumulator
   atomics drop to ``sem="relaxed"``.  The accumulators are pure sums whose
   result is only read after kernel completion, so relaxed ordering is correct;
   cross-kernel visibility is provided by the stream dependency between the two
   launches.
2. **Per-program register histogram, no scattered atomics.**  The count pass
   builds the histogram of its token tile in registers and publishes it with a
   single *contiguous* vector atomic, so neighbouring atomic lanes coalesce
   instead of each lane serializing on an unrelated address.  The per-rank picks
   are accumulated with ``tl.histogram`` instead of an explicit
   ``[BLOCK_TOKENS, BLOCK_EXPERTS]`` integer mask, which keeps the tile in
   registers: the same shape drops from ``148 us`` (mask + relaxed) to
   ``86 us``.  Padding rows and masked-out tokens always report expert 0 once
   per rank, so their contribution is subtracted from bin 0 afterwards, which
   keeps the counts bit-identical to the reference.
3. **Tile geometry.**  ``num_warps = 8`` with a 32-token tile is best across the
   measured shapes; a wider tile spills the fp32 logits tile once ``N_e``
   reaches 128.

The Top-K index is recovered with a plain ``tl.max`` + ``tl.min(where(...))``
pair rather than ``tl.argmax`` (ties resolve to the lowest expert index, which
keeps the selection deterministic).  ``exp()/sum(exp())`` is folded into the
expert-weighted numerator so the normalizer is applied once per row instead of
once per element.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger("flag_gems.ops.moe_load_balancing_loss")


_NUM_WARPS = 8
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
    """Sequential Top-K counting with a per-program register histogram.

    Each of the ``TOP_K`` ranks picks the running maximum, the index is
    recovered with a ``tl.max`` + ``tl.min`` pair, and the pick is tallied with
    ``tl.histogram``.  Padding rows and masked-out tokens have an all ``-inf``
    row, so they report expert 0 on every rank; that over-count is removed from
    bin 0 in one shot at the end.
    """
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

    histogram = tl.zeros((BLOCK_EXPERTS,), dtype=tl.float32)

    # Softmax preserves ordering, so Top-K(logits) equals Top-K(softmax(logits)).
    for _ in tl.static_range(0, TOP_K):
        row_max = tl.max(logits, axis=1)
        selected = tl.min(
            tl.where(
                logits == row_max[:, None],
                expert_offsets[None, :],
                BLOCK_EXPERTS,
            ),
            axis=1,
        )
        histogram += tl.histogram(selected, BLOCK_EXPERTS).to(tl.float32)
        logits = tl.where(
            expert_offsets[None, :] == selected[:, None], -float("inf"), logits
        )

    # Every padded / masked row reports expert 0 once per rank; drop that.
    invalid_rows = tl.sum(tl.where(valid_tokens, 0.0, 1.0), axis=0)
    histogram = tl.where(
        expert_offsets == 0, histogram - TOP_K * invalid_rows, histogram
    )

    tl.atomic_add(
        expert_counts + expert_offsets,
        histogram,
        mask=expert_offsets < NUM_EXPERTS,
        sem="relaxed",
    )
    tl.atomic_add(
        valid_token_count,
        tl.sum(valid_tokens.to(tl.float32), axis=0),
        sem="relaxed",
    )


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
    """Accumulate ``N_e * sum_i c_i * p_i / V^2`` over the token tile."""
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

    counts = tl.load(
        expert_counts + expert_offsets,
        mask=expert_offsets < NUM_EXPERTS,
        other=0.0,
    )
    # exp()/sum(exp()) is folded into the expert-weighted numerator so the
    # normalizer is applied once per row instead of once per element.
    weighted = tl.sum(numerators * counts[None, :], axis=1)
    denominator = tl.sum(numerators, axis=1)

    total_valid = tl.load(valid_token_count)
    scale = NUM_EXPERTS / tl.maximum(total_valid * total_valid, 1.0)
    contributions = tl.where(valid_tokens, weighted * scale / denominator, 0.0)
    tl.atomic_add(output, tl.sum(contributions, axis=0), sem="relaxed")


def _validate_inputs(
    gate_logits: torch.Tensor,
    top_k: int,
    attention_mask: torch.Tensor | None,
) -> None:
    """Validate the operator contract."""
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
    r"""Compute the Top-K MoE load-balancing auxiliary loss on Moore Threads.

    ``gate_logits`` is ``[T, N_e]`` in float16/bfloat16/float32, ``top_k`` a
    Python int in ``[1, N_e]``, and the optional ``attention_mask`` holds ``T``
    entries whose zero positions are excluded from both statistics.  The
    returned value is a float32 scalar (forward only).

    Args:
        gate_logits: Raw router logits with shape ``[T, N_e]``.
        top_k: Number of experts selected per token.
        attention_mask: Optional mask containing ``T`` elements.

    Returns:
        A float32 scalar tensor containing ``L_ib``.
    """
    _validate_inputs(gate_logits, top_k, attention_mask)
    logger.debug("GEMS_MTHREADS MOE_LOAD_BALANCE_LOSS")

    logits = gate_logits.contiguous()
    # The kernel never reads the mask when HAS_MASK is False; reuse the logits
    # pointer so no extra allocation is needed for the dummy argument.
    flat_mask = (
        logits if attention_mask is None else attention_mask.reshape(-1).contiguous()
    )
    has_mask = attention_mask is not None
    num_tokens, num_experts = logits.shape

    # All three atomic targets live in one allocation so a single memset zeroes
    # them (three separate ``torch.zeros`` would cost three device-wide memsets,
    # which dominates the small-router shapes).  They are still disjoint ranges,
    # so a widened vector atomic can never alias a neighbour.
    scratch = torch.zeros(num_experts + 2, dtype=torch.float32, device=logits.device)
    expert_counts = scratch[:num_experts]
    valid_token_count = scratch[num_experts : num_experts + 1]
    output = scratch[num_experts + 1 : num_experts + 2]

    block_experts = triton.next_power_of_2(num_experts)
    grid = (triton.cdiv(num_tokens, _BLOCK_TOKENS),)

    with torch_device_fn.device(logits.device):
        _topk_count_kernel[grid](
            logits,
            flat_mask,
            expert_counts,
            valid_token_count,
            num_tokens,
            NUM_EXPERTS=num_experts,
            TOP_K=top_k,
            BLOCK_TOKENS=_BLOCK_TOKENS,
            BLOCK_EXPERTS=block_experts,
            HAS_MASK=has_mask,
            num_warps=_NUM_WARPS,
        )
        _loss_kernel[grid](
            logits,
            flat_mask,
            expert_counts,
            valid_token_count,
            output,
            num_tokens,
            NUM_EXPERTS=num_experts,
            BLOCK_TOKENS=_BLOCK_TOKENS,
            BLOCK_EXPERTS=block_experts,
            HAS_MASK=has_mask,
            num_warps=_NUM_WARPS,
        )

    return scratch[num_experts + 1]


__all__ = ["moe_load_balance_loss"]
