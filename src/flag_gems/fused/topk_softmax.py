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


@triton.jit
def topk_gating_softmax_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    source_rows_ptr,
    num_rows,
    num_experts,
    renormalize: tl.constexpr,
    INDEX_TY: tl.constexpr,
    K: tl.constexpr,
    KB: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < num_rows
    cols = tl.arange(0, BLOCK_E)
    col_mask = cols < num_experts

    offs = rows[:, None] * num_experts + cols[None, :]
    logits = tl.load(
        input_ptr + offs,
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    logits = tl.where(col_mask[None, :], logits, -float("inf"))
    logits = tl.where(row_mask[:, None], logits, 0.0)

    k_ar = tl.arange(0, KB)
    k_valid = k_ar < K

    if renormalize:
        # The global softmax denominator cancels in renormalization: select
        # top-k directly in the logits domain and only exp/normalize the K
        # selected logits.
        weights = tl.full([BLOCK_ROWS, KB], -float("inf"), dtype=tl.float32)
        work = logits
        for ki in tl.static_range(K):
            val, idx = tl.max(work, axis=1, return_indices=True)
            sel = k_ar[None, :] == ki
            weights = tl.where(sel, val[:, None], weights)
            tl.store(indices_ptr + rows * K + ki, idx.to(INDEX_TY), mask=row_mask)
            tl.store(
                source_rows_ptr + rows * K + ki,
                (ki * num_rows + rows).to(tl.int32),
                mask=row_mask,
            )
            work = tl.where(cols[None, :] == idx[:, None], -float("inf"), work)

        top1 = tl.max(weights, axis=1)
        e = tl.exp(weights - top1[:, None])
        wsum = tl.sum(e, axis=1) + 1e-8
        # fast div: one reciprocal + multiplies instead of per-element division
        inv = tl.math.fdiv(1.0, wsum)
        weights = e * inv[:, None]

        out_offs = rows[:, None] * K + k_ar[None, :]
        out_mask = row_mask[:, None] & k_valid[None, :]
        tl.store(output_ptr + out_offs, weights, mask=out_mask)
    else:
        # The output is the softmax probability itself, so the full softmax
        # denominator is required.
        row_max = tl.max(logits, axis=1)
        exp_vals = tl.exp(logits - row_max[:, None])
        denom = tl.sum(exp_vals, axis=1) + 1e-8
        # fast div: one reciprocal + multiplies instead of per-element division
        inv = tl.math.fdiv(1.0, denom)
        probs = exp_vals * inv[:, None]

        work = probs
        for ki in tl.static_range(K):
            val, idx = tl.max(work, axis=1, return_indices=True)
            tl.store(output_ptr + rows * K + ki, val, mask=row_mask)
            tl.store(indices_ptr + rows * K + ki, idx.to(INDEX_TY), mask=row_mask)
            tl.store(
                source_rows_ptr + rows * K + ki,
                (ki * num_rows + rows).to(tl.int32),
                mask=row_mask,
            )
            work = tl.where(cols[None, :] == idx[:, None], -float("inf"), work)


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> None:
    logger.debug("GEMS TOPK_SOFTMAX")
    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.size(-1)
    assert topk <= 32

    if topk_indices.dtype == torch.int32:
        index_ty = tl.int32
    elif topk_indices.dtype == torch.uint32:
        index_ty = tl.uint32
    elif topk_indices.dtype == torch.int64:
        index_ty = tl.int64
    else:
        raise TypeError("topk_indices must be int32/int64/uint32")

    max_total_threads = 1024
    block_e = triton.next_power_of_2(num_experts)
    block_e = min(block_e, max_total_threads)
    block_rows = max(1, min(16, max_total_threads // block_e))
    num_warps = 4 if block_e <= 256 else 8
    kb = triton.next_power_of_2(topk)

    grid = (triton.cdiv(num_tokens, block_rows),)
    topk_gating_softmax_kernel[grid](
        input_ptr=gating_output,
        output_ptr=topk_weights,
        indices_ptr=topk_indices,
        source_rows_ptr=token_expert_indices,
        num_rows=num_tokens,
        num_experts=num_experts,
        renormalize=renormalize,
        INDEX_TY=index_ty,
        K=topk,
        KB=kb,
        BLOCK_ROWS=block_rows,
        BLOCK_E=block_e,
        num_warps=num_warps,
    )
