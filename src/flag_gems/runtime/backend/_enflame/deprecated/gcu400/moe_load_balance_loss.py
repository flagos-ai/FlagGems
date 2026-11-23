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

"""Enflame GCU400/410 specialization of the Top-K MoE load-balancing loss.

.. math::
    L_{ib} = N_e \\sum_i f_i P_i,

with ``f_i`` the Top-K assignment frequency of expert ``i`` and ``P_i`` its
mean routing probability, both computed in fp32 from the router logits.
Tokens zeroed by the optional ``attention_mask`` are excluded from both
statistics and from the denominator.

The public implementation accumulates with cross-CTA ``tl.atomic_add`` and
re-reads the logits in a second loss pass. GCU kernels must not use atomics
(non-deterministic ordering; see knowledge/coding_rules.md), so this port:

1. computes the three sufficient statistics -- the expert Top-K histogram,
   the per-expert softmax probability sums, and the valid token count -- in
   **one** streaming pass over the logits, publishing per-program partial
   rows with disjoint stores (no atomics, fixed reduction order, therefore
   bit-deterministic);
2. reduces the few dozen partial rows with a tiny single-program finalize
   kernel. The loss is a pure function of the aggregates,
   ``L = N_e * sum_i counts_i * probsum_i / max(V^2, 1)``, so the public
   version's per-token second pass (a full T x N_e re-read) is unnecessary;
3. recovers Top-K indices with the ``max`` + tie-break ``where`` + ``min``
   chain (lowest expert index on ties, matching ``tl.argmax``) and tallies
   with one-hot + ``tl.sum(axis=0)``: ``tl.histogram`` is silently wrong on
   GCU (dropped masks / cross-run non-determinism, knowledge/compiler_traps);
4. narrows int64 masks to int32 on the host so the kernel never performs
   8-byte integer loads, independently of TORCH_GCU_ENABLE_INT64_AND_UINT64;
5. drops boundary predicates from the logits load when N_e is a power of two
   and T divides the token tile (every core benchmark shape), and pins
   num_warps to 1 -- measured fastest for every probed configuration on
   GCU400 and the only warp count verified safe for BLOCK_EXPERTS >= 256
   (the P10 reduction-then-broadcast pattern miscompiles with wider warps).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


# Persistent grid: GCU CTA dispatch is expensive, so the program count is
# decoupled from the problem size (one program per SP) and every program
# walks several token tiles.
_NUM_PROGRAMS = 48
# Token tile per BLOCK_EXPERTS, measured on GCU400 (do_bench medians):
# wide expert rows stream fastest with ~8 KiB tiles, narrow rows prefer the
# largest token batch (which also lets small T collapse onto the single
# launch path). num_warps=1 won every probed (BLOCK_EXPERTS, dtype, shape)
# combination -- axis reductions do not amortize wider warps here -- and it
# is also the only warp count verified safe for BLOCK_EXPERTS >= 256 (P10).
_TOKENS_BE16 = 256
_TOKENS_BE64 = 128
_BIG_TILE_BYTES = 8192
_BLOCK_TOKENS_MIN = 16
# K is statically unrolled only for the practical router Top-K values; a
# larger K takes the runtime-bound loop to bound compile time and code size.
_TOPK_STATIC_MAX = 16


def _plan_block_tokens(block_experts, element_size):
    if block_experts <= 16:
        return _TOKENS_BE16
    if block_experts <= 64:
        return _TOKENS_BE64 if element_size < 4 else _TOKENS_BE64 // 2
    return max(_BLOCK_TOKENS_MIN, _BIG_TILE_BYTES // (block_experts * element_size))


@triton.jit
def _topk_round(
    cur,
    hist,
    valid,
    expert_offsets,
    BLOCK_EXPERTS: tl.constexpr,
    MARK: tl.constexpr,
):
    """One Top-K round: pick each row's running max, lowest index on ties."""
    row_max = tl.max(cur, axis=1)
    selected = tl.min(
        tl.where(cur == row_max[:, None], expert_offsets[None, :], BLOCK_EXPERTS),
        axis=1,
    )
    if MARK:
        # Invalid rows report BLOCK_EXPERTS, which matches no one-hot lane.
        selected = tl.where(valid, selected, BLOCK_EXPERTS)
    hist = hist + tl.sum(
        (selected[:, None] == expert_offsets[None, :]).to(tl.float32), axis=0
    )
    cur = tl.where(expert_offsets[None, :] == selected[:, None], -float("inf"), cur)
    return cur, hist


@triton.jit
def _moe_lb_stats_kernel(
    gate_logits,
    attention_mask,
    partials,
    output,
    num_tokens,
    top_k,
    NUM_EXPERTS: tl.constexpr,
    TOP_K_STATIC: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    EVEN_LOAD: tl.constexpr,
    ALL_VALID: tl.constexpr,
    GRID: tl.constexpr,
    FINALIZE: tl.constexpr,
):
    """Single streaming pass producing the per-program loss statistics.

    ``FINALIZE`` (single-program grids) reduces the in-register aggregates to
    the scalar loss directly; otherwise each program writes one disjoint
    ``[counts | prob_sums | valid_count]`` partial row for the reduce kernel.
    """
    pid = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_EXPERTS)
    token_lanes = tl.arange(0, BLOCK_TOKENS)
    num_tiles = tl.cdiv(num_tokens, BLOCK_TOKENS)

    histogram = tl.zeros((BLOCK_EXPERTS,), dtype=tl.float32)
    prob_sums = tl.zeros((BLOCK_EXPERTS,), dtype=tl.float32)
    valid_count = 0.0

    for tile in range(pid, num_tiles, GRID):
        token_offsets = tile * BLOCK_TOKENS + token_lanes
        row_offsets = token_offsets[:, None] * NUM_EXPERTS + expert_offsets[None, :]
        in_range = token_offsets < num_tokens

        if HAS_MASK:
            mask_values = tl.load(
                attention_mask + token_offsets, mask=in_range, other=0
            )
            valid = in_range & (mask_values != 0)
        else:
            valid = in_range

        if EVEN_LOAD:
            logits = tl.load(gate_logits + row_offsets).to(tl.float32)
        else:
            load_mask = valid[:, None] & (expert_offsets[None, :] < NUM_EXPERTS)
            logits = tl.load(
                gate_logits + row_offsets, mask=load_mask, other=-float("inf")
            ).to(tl.float32)

        # Softmax statistics from the pristine tile: padded expert columns
        # give exp(-inf) = 0 naturally, and rows without a valid token are
        # zeroed before entering the per-expert accumulator.
        row_max = tl.max(logits, axis=1)
        numerators = tl.exp(logits - row_max[:, None])
        denominators = tl.sum(numerators, axis=1)
        probabilities = numerators / denominators[:, None]
        if not ALL_VALID:
            probabilities = tl.where(valid[:, None], probabilities, 0.0)
        prob_sums += tl.sum(probabilities, axis=0)

        # Top-K assignment histogram (mutates the tile copy afterwards).
        if TOP_K_STATIC > 0:
            for _ in tl.static_range(TOP_K_STATIC):
                logits, histogram = _topk_round(
                    logits,
                    histogram,
                    valid,
                    expert_offsets,
                    BLOCK_EXPERTS,
                    not ALL_VALID,
                )
        else:
            for _ in range(0, top_k):
                logits, histogram = _topk_round(
                    logits,
                    histogram,
                    valid,
                    expert_offsets,
                    BLOCK_EXPERTS,
                    not ALL_VALID,
                )

        if ALL_VALID:
            valid_count += BLOCK_TOKENS
        else:
            valid_count += tl.sum(valid.to(tl.float32), axis=0)

    if FINALIZE:
        denominator = tl.maximum(valid_count * valid_count, 1.0)
        loss = tl.sum(histogram * prob_sums) * (NUM_EXPERTS / denominator)
        tl.store(output, loss)
    else:
        base = partials + pid * (2 * BLOCK_EXPERTS + 1)
        tl.store(base + expert_offsets, histogram)
        tl.store(base + BLOCK_EXPERTS + expert_offsets, prob_sums)
        tl.store(base + 2 * BLOCK_EXPERTS, valid_count)


@triton.jit
def _moe_lb_reduce_kernel(
    partials,
    output,
    num_programs,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Fixed-order single-program reduction of the partial rows to the loss."""
    expert_offsets = tl.arange(0, BLOCK_EXPERTS)
    program_offsets = tl.arange(0, BLOCK_P)
    stride = 2 * BLOCK_EXPERTS + 1

    counts = tl.zeros((BLOCK_EXPERTS,), dtype=tl.float32)
    prob_sums = tl.zeros((BLOCK_EXPERTS,), dtype=tl.float32)
    for program in range(0, num_programs):
        base = partials + program * stride
        counts += tl.load(base + expert_offsets)
        prob_sums += tl.load(base + BLOCK_EXPERTS + expert_offsets)

    program_mask = program_offsets < num_programs
    valid_count = tl.sum(
        tl.load(
            partials + program_offsets * stride + 2 * BLOCK_EXPERTS,
            mask=program_mask,
            other=0.0,
        )
    )
    denominator = tl.maximum(valid_count * valid_count, 1.0)
    loss = tl.sum(counts * prob_sums) * (NUM_EXPERTS / denominator)
    tl.store(output, loss)


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
    r"""Compute the Top-K MoE load-balancing auxiliary loss on Enflame GCU400.

    ``gate_logits`` is ``[T, N_e]`` in float16/bfloat16/float32, ``top_k`` a
    Python int in ``[1, N_e]``, and the optional ``attention_mask`` holds ``T``
    entries whose zero positions are excluded from both statistics. The
    returned value is a float32 scalar (forward only).

    Args:
        gate_logits: Raw router logits with shape ``[T, N_e]``.
        top_k: Number of experts selected per token.
        attention_mask: Optional mask containing ``T`` elements.

    Returns:
        A float32 scalar tensor containing ``L_ib``. An all-zero mask
        returns zero.
    """
    _validate_inputs(gate_logits, top_k, attention_mask)
    logger.debug("GEMS_ENFLAME GCU400 MOE_LOAD_BALANCE_LOSS")

    logits = gate_logits.contiguous()
    # The kernel never reads the mask when HAS_MASK is False; reuse the logits
    # pointer so no extra allocation is needed for the dummy argument.
    if attention_mask is None:
        flat_mask = logits
    else:
        flat_mask = attention_mask.reshape(-1).contiguous()
        if flat_mask.dtype == torch.int64:
            # ATen handles the fake-int64 layout correctly, Triton loads do
            # not; narrow on the host so the kernel is env-independent.
            flat_mask = flat_mask.to(torch.int32)
    has_mask = attention_mask is not None
    num_tokens, num_experts = logits.shape

    block_experts = triton.next_power_of_2(num_experts)
    block_tokens = _plan_block_tokens(block_experts, logits.element_size())
    num_programs = min(_NUM_PROGRAMS, triton.cdiv(num_tokens, block_tokens))
    single = num_programs == 1
    t_even = num_tokens % block_tokens == 0

    output = torch.empty((), dtype=torch.float32, device=logits.device)
    # Every program overwrites its own disjoint row, so no zero init is
    # needed; the dummy alias keeps the kernel signature stable.
    partials = (
        output
        if single
        else torch.empty(
            num_programs * (2 * block_experts + 1),
            dtype=torch.float32,
            device=logits.device,
        )
    )

    # num_warps=1 won every probed (BLOCK_EXPERTS, dtype, shape) combination
    # on GCU400 and is additionally the only warp count verified safe for
    # BLOCK_EXPERTS >= 256, where the P10 axis-1-reduction-then-broadcast
    # pattern miscompiles with multiple warps.
    num_warps = 1

    with torch_device_fn.device(logits.device):
        _moe_lb_stats_kernel[(num_programs,)](
            logits,
            flat_mask,
            partials,
            output,
            num_tokens,
            top_k,
            NUM_EXPERTS=num_experts,
            TOP_K_STATIC=top_k if top_k <= _TOPK_STATIC_MAX else 0,
            BLOCK_TOKENS=block_tokens,
            BLOCK_EXPERTS=block_experts,
            HAS_MASK=has_mask,
            EVEN_LOAD=(num_experts == block_experts) and t_even,
            ALL_VALID=(not has_mask) and t_even,
            GRID=num_programs,
            FINALIZE=single,
            num_warps=num_warps,
        )
        if not single:
            _moe_lb_reduce_kernel[(1,)](
                partials,
                output,
                num_programs,
                NUM_EXPERTS=num_experts,
                BLOCK_EXPERTS=block_experts,
                BLOCK_P=triton.next_power_of_2(num_programs),
                num_warps=1,
            )

    return output


__all__ = ["moe_load_balance_loss"]
