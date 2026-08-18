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

import math

import pytest
import torch

import flag_gems
from flag_gems.fused import bf16_paged_mqa_logits

device = flag_gems.device

# Default model parameters (hardcoded to match kernel specializations)
HEAD_DIM = 128  # hardcoded in kernel specializations
BLOCK_SIZE = 64  # hardcoded KV cache page size

# Test shapes: (batch_size, next_n, num_heads, context_len)
# Cover both H=32 and H=64 specializations with various context lengths
TEST_SHAPES = [
    (1, 1, 32, 64),
    (1, 1, 32, 128),
    (1, 1, 32, 256),
    (1, 1, 32, 1024),
    (1, 1, 32, 2048),
    (4, 1, 32, 256),
    (4, 1, 32, 1024),
    (4, 1, 32, 2048),
    (8, 1, 32, 1024),
    (4, 2, 32, 1024),
    (1, 1, 64, 1024),
    (4, 1, 64, 2048),
]


def _ceil_div(a, b):
    return (a + b - 1) // b


def _make_inputs(B, next_n, H, ctx_len):
    """Generate test inputs for bf16 paged MQA logits."""
    torch.manual_seed(42)
    q = torch.randn(B, next_n, H, HEAD_DIM, device=device, dtype=torch.bfloat16)
    weights = torch.randn(B * next_n, H, device=device, dtype=torch.float32).abs()
    context_lens = torch.full((B, next_n), ctx_len, device=device, dtype=torch.int32)

    num_blocks_per_seq = _ceil_div(ctx_len, BLOCK_SIZE)
    total_blocks = B * num_blocks_per_seq + 10
    kv_cache = torch.randn(
        total_blocks, BLOCK_SIZE, 1, HEAD_DIM, device=device, dtype=torch.bfloat16
    )

    block_table = torch.zeros(B, num_blocks_per_seq, device=device, dtype=torch.int32)
    for b in range(B):
        for i in range(num_blocks_per_seq):
            block_table[b, i] = b * num_blocks_per_seq + i

    schedule_meta = torch.zeros(2, 2, device=device, dtype=torch.int32)
    return q, kv_cache, weights, context_lens, block_table, schedule_meta


def _reference_bf16_paged_mqa_logits(
    q, kv_cache, weights, context_lens, block_table, max_context_len
):
    """Pure PyTorch reference implementation.

    logits[row, pos] = sum_h( relu(q[b,n,h,:] · K[pos,:]) * w[row, h] )
    """
    B, next_n, H, D = q.shape
    total_tokens = B * next_n
    logits = torch.zeros(
        total_tokens, max_context_len, device=q.device, dtype=torch.float32
    )

    for b in range(B):
        for n in range(next_n):
            row = b * next_n + n
            ctx_len = int(context_lens[b, n].item())
            q_row = q[b, n]  # [H, D]

            for blk_idx in range(_ceil_div(ctx_len, BLOCK_SIZE)):
                phys_blk = int(block_table[b, blk_idx].item())
                start_pos = blk_idx * BLOCK_SIZE
                end_pos = min(start_pos + BLOCK_SIZE, ctx_len)
                num_pos = end_pos - start_pos

                # K: [num_pos, D]
                k_block = kv_cache[phys_blk, :num_pos, 0, :]

                # scores: [num_pos, H]
                scores = torch.mm(k_block.float(), q_row.float().t())

                # ReLU + weighted sum
                scores = torch.relu(scores)
                w = weights[row]  # [H]
                logits_val = (scores * w.unsqueeze(0)).sum(dim=1)

                logits[row, start_pos:end_pos] = logits_val

    return logits


@pytest.mark.bf16_paged_mqa_logits
@pytest.mark.parametrize(
    "B, next_n, H, ctx_len",
    TEST_SHAPES,
    ids=[f"B{b}_N{n}_H{h}_L{l}" for b, n, h, l in TEST_SHAPES],
)
def test_bf16_paged_mqa_logits(B, next_n, H, ctx_len):
    q, kv_cache, weights, context_lens, block_table, schedule_meta = _make_inputs(
        B, next_n, H, ctx_len
    )

    # Run Triton kernel
    triton_out = bf16_paged_mqa_logits(
        q=q,
        kv_cache=kv_cache,
        weights=weights,
        context_lens=context_lens,
        block_table=block_table,
        schedule_metadata=schedule_meta,
        max_context_len=ctx_len,
        clean_logits=False,
    )

    # Run PyTorch reference
    ref_out = _reference_bf16_paged_mqa_logits(
        q, kv_cache, weights, context_lens, block_table, ctx_len
    )

    # Compare valid positions
    # Tolerance scaled by head_dim * sqrt(num_heads) due to FP accumulation
    atol = 1e-4 * HEAD_DIM * math.sqrt(H)
    rtol = 1.6e-2

    total_tokens = B * next_n
    ctx_flat = context_lens.reshape(-1)[:total_tokens]
    for row in range(total_tokens):
        cl = int(ctx_flat[row].item())
        if cl == 0:
            continue
        torch.testing.assert_close(
            triton_out[row, :cl],
            ref_out[row, :cl],
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.bf16_paged_mqa_logits
def test_bf16_paged_mqa_logits_variable_ctx():
    """Test with variable context lengths across batch elements."""
    B, next_n, H = 4, 1, 32
    ctx_list = [64, 128, 256, 512]
    max_ctx = max(ctx_list)

    torch.manual_seed(123)
    q = torch.randn(B, next_n, H, HEAD_DIM, device=device, dtype=torch.bfloat16)
    weights = torch.randn(B * next_n, H, device=device, dtype=torch.float32).abs()
    context_lens = torch.tensor(
        [[c] for c in ctx_list], device=device, dtype=torch.int32
    )

    max_blocks = _ceil_div(max_ctx, BLOCK_SIZE)
    total_blocks = B * max_blocks + 5
    kv_cache = torch.randn(
        total_blocks, BLOCK_SIZE, 1, HEAD_DIM, device=device, dtype=torch.bfloat16
    )

    block_table = torch.zeros(B, max_blocks, device=device, dtype=torch.int32)
    c = 0
    for b in range(B):
        nb = _ceil_div(ctx_list[b], BLOCK_SIZE)
        for i in range(nb):
            block_table[b, i] = c
            c += 1

    schedule_meta = torch.zeros(2, 2, device=device, dtype=torch.int32)

    triton_out = bf16_paged_mqa_logits(
        q=q,
        kv_cache=kv_cache,
        weights=weights,
        context_lens=context_lens,
        block_table=block_table,
        schedule_metadata=schedule_meta,
        max_context_len=max_ctx,
        clean_logits=False,
    )

    ref_out = _reference_bf16_paged_mqa_logits(
        q, kv_cache, weights, context_lens, block_table, max_ctx
    )

    atol = 1e-4 * HEAD_DIM * math.sqrt(H)
    rtol = 1.6e-2

    for b in range(B):
        cl = ctx_list[b]
        torch.testing.assert_close(
            triton_out[b, :cl],
            ref_out[b, :cl],
            rtol=rtol,
            atol=atol,
        )
