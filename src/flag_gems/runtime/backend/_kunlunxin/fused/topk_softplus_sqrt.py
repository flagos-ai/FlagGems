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

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["num_elements"])
def _cast_logits_kernel(gating_ptr, scores_ptr, num_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_elements
    logits = tl.load(gating_ptr + offsets, mask=mask, other=0.0)
    tl.store(scores_ptr + offsets, logits.to(tl.float32), mask=mask)


@triton.jit(do_not_specialize=["num_elements"])
def _exp_inplace_kernel(values_ptr, num_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_elements
    values = tl.load(values_ptr + offsets, mask=mask, other=0.0)
    tl.store(values_ptr + offsets, tl.exp(values), mask=mask)


@triton.jit(do_not_specialize=["num_tokens", "num_experts"])
def _prepare_scores_kernel(
    scores_ptr,
    correction_bias_ptr,
    num_tokens,
    num_experts,
    HAS_BIAS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    token_idx = tl.program_id(0)
    expert_offsets = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)
    expert_mask = expert_offsets < num_experts
    row_offset = token_idx * num_experts
    logits = tl.load(
        scores_ptr + row_offset + expert_offsets,
        mask=expert_mask,
        other=0.0,
    )
    scores = tl.sqrt(tl.log(1.0 + logits))
    if HAS_BIAS:
        bias = tl.load(
            correction_bias_ptr + expert_offsets,
            mask=expert_mask,
            other=0.0,
        ).to(tl.float32)
        scores += bias
    tl.store(scores_ptr + row_offset + expert_offsets, scores, mask=expert_mask)


@triton.jit(
    do_not_specialize=["num_tokens", "num_experts", "topk", "topk_offset"]
)
def _select_one_kernel(
    scores_ptr,
    correction_bias_ptr,
    topk_weights_ptr,
    topk_indices_ptr,
    token_expert_indices_ptr,
    num_tokens,
    num_experts,
    topk,
    topk_offset,
    HAS_BIAS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    token_idx = tl.program_id(0)
    if token_idx >= num_tokens:
        return

    expert_offsets = tl.arange(0, BLOCK_E)
    expert_mask = expert_offsets < num_experts
    row_offset = token_idx * num_experts
    scores = tl.load(
        scores_ptr + row_offset + expert_offsets,
        mask=expert_mask,
        other=-float("inf"),
    )
    max_score = tl.max(scores, axis=0)
    is_max = scores == max_score
    expert_idx = tl.min(
        tl.where(is_max, expert_offsets, num_experts), axis=0
    ).to(tl.int32)

    weight = max_score
    if HAS_BIAS:
        weight -= tl.load(correction_bias_ptr + expert_idx).to(tl.float32)

    output_offset = token_idx * topk + topk_offset
    tl.store(topk_weights_ptr + output_offset, weight)
    tl.store(topk_indices_ptr + output_offset, expert_idx)
    tl.store(token_expert_indices_ptr + output_offset, output_offset)

    tl.store(scores_ptr + row_offset + expert_idx, -float("inf"))


@triton.jit(
    do_not_specialize=["num_tokens", "topk", "routed_scaling_factor"]
)
def _finalize_weights_kernel(
    topk_weights_ptr,
    num_tokens,
    topk,
    routed_scaling_factor,
    APPLY_LOG1P_SQRT: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token_idx = tl.program_id(0)
    if token_idx >= num_tokens:
        return

    topk_offsets = tl.arange(0, BLOCK_K)
    topk_mask = topk_offsets < topk
    row_offset = token_idx * topk
    weights = tl.load(
        topk_weights_ptr + row_offset + topk_offsets,
        mask=topk_mask,
        other=0.0,
    )
    if APPLY_LOG1P_SQRT:
        weights = tl.sqrt(tl.log(1.0 + weights))

    scale = routed_scaling_factor
    if RENORMALIZE:
        weight_sum = tl.sum(tl.where(topk_mask, weights, 0.0), axis=0)
        scale = routed_scaling_factor / tl.where(weight_sum > 0.0, weight_sum, 1.0)
    tl.store(
        topk_weights_ptr + row_offset + topk_offsets,
        weights * scale,
        mask=topk_mask,
    )


@triton.jit(do_not_specialize=["num_tokens", "num_experts", "topk"])
def _hash_gather_kernel(
    gating_ptr,
    topk_weights_ptr,
    topk_indices_ptr,
    token_expert_indices_ptr,
    input_ids_ptr,
    tid2eid_ptr,
    num_tokens,
    num_experts,
    topk,
    BLOCK_K: tl.constexpr,
):
    token_idx = tl.program_id(0)
    if token_idx >= num_tokens:
        return

    topk_offsets = tl.arange(0, BLOCK_K)
    topk_mask = topk_offsets < topk
    token_id = tl.load(input_ids_ptr + token_idx).to(tl.int64)
    expert_indices = tl.load(
        tid2eid_ptr + token_id * topk + topk_offsets,
        mask=topk_mask,
        other=0,
    ).to(tl.int32)
    logits = tl.load(
        gating_ptr + token_idx * num_experts + expert_indices,
        mask=topk_mask,
        other=0.0,
    )

    output_offset = token_idx * topk + topk_offsets
    tl.store(topk_weights_ptr + output_offset, logits.to(tl.float32), mask=topk_mask)
    tl.store(topk_indices_ptr + output_offset, expert_indices, mask=topk_mask)
    tl.store(token_expert_indices_ptr + output_offset, output_offset, mask=topk_mask)


def topk_softplus_sqrt(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    renormalize,
    routed_scaling_factor,
    correction_bias=None,
    input_ids=None,
    tid2eid=None,
):
    logger.debug("GEMS KUNLUNXIN TOPK_SOFTPLUS_SQRT")
    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.shape[1]
    if num_tokens == 0:
        return

    launch_kwargs = {
        "isCloseVectorization": True,
        "buffer_size_limit": 2048,
        "num_warps": 1,
        "num_stages": 1,
    }
    math_launch_kwargs = {"num_warps": 1, "num_stages": 1}
    token_grid = (num_tokens,)
    block_k = triton.next_power_of_2(topk)

    if input_ids is not None and tid2eid is not None:
        _hash_gather_kernel[token_grid](
            gating_output,
            topk_weights,
            topk_indices,
            token_expert_indices,
            input_ids,
            tid2eid,
            num_tokens,
            num_experts,
            topk,
            BLOCK_K=block_k,
            **launch_kwargs,
        )
        hash_elements = num_tokens * topk
        _exp_inplace_kernel[(triton.cdiv(hash_elements, 256),)](
            topk_weights,
            hash_elements,
            BLOCK=256,
            **math_launch_kwargs,
        )
        _finalize_weights_kernel[token_grid](
            topk_weights,
            num_tokens,
            topk,
            routed_scaling_factor,
            APPLY_LOG1P_SQRT=True,
            RENORMALIZE=renormalize,
            BLOCK_K=block_k,
            **math_launch_kwargs,
        )
        return

    scores = torch.empty(
        (num_tokens, num_experts), dtype=torch.float32, device=gating_output.device
    )
    num_elements = num_tokens * num_experts
    cast_block = 256
    _cast_logits_kernel[(triton.cdiv(num_elements, cast_block),)](
        gating_output,
        scores,
        num_elements,
        BLOCK=cast_block,
        **launch_kwargs,
    )
    _exp_inplace_kernel[(triton.cdiv(num_elements, cast_block),)](
        scores,
        num_elements,
        BLOCK=cast_block,
        **math_launch_kwargs,
    )

    bias_ptr = correction_bias if correction_bias is not None else gating_output
    block_e = triton.next_power_of_2(num_experts)
    prepare_block = 128
    prepare_grid = (num_tokens, triton.cdiv(num_experts, prepare_block))
    _prepare_scores_kernel[prepare_grid](
        scores,
        bias_ptr,
        num_tokens,
        num_experts,
        HAS_BIAS=correction_bias is not None,
        BLOCK_E=prepare_block,
        **math_launch_kwargs,
    )
    for topk_offset in range(topk):
        _select_one_kernel[token_grid](
            scores,
            bias_ptr,
            topk_weights,
            topk_indices,
            token_expert_indices,
            num_tokens,
            num_experts,
            topk,
            topk_offset,
            HAS_BIAS=correction_bias is not None,
            BLOCK_E=block_e,
            **launch_kwargs,
        )
    _finalize_weights_kernel[token_grid](
        topk_weights,
        num_tokens,
        topk,
        routed_scaling_factor,
        APPLY_LOG1P_SQRT=False,
        RENORMALIZE=renormalize,
        BLOCK_K=block_k,
        **math_launch_kwargs,
    )
