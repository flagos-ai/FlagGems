"""Accuracy tests for unified_attention (vLLM paged attention kernel).

Tests the KernelGen v10 optimized Triton kernel against a pure-PyTorch
reference implementation of paged attention. The reference unpacks the
paged KV cache and computes attention with torch.einsum, providing a
bit-exact correctness baseline.

Test shapes cover decode, prefill, GQA, and large-head-dim scenarios.
"""

import pytest
import torch

import flag_gems
from flag_gems.fused import unified_attention

from . import conftest as cfg

device = flag_gems.device

# Default: bfloat16 for paged attention (matches vLLM default)
DTYPE = torch.bfloat16
BLOCK_SIZE = 16
RTOL = 1e-2
ATOL = 1e-2

# Test shapes: (num_seqs, query_len, kv_len, n_q_heads, n_kv_heads, head_size)
if cfg.QUICK_MODE:
    TEST_SHAPES = [
        (1, 1, 1024, 32, 8, 128),  # decode B1
        (4, 128, 1024, 32, 8, 128),  # prefill
    ]
else:
    TEST_SHAPES = [
        # decode
        (1, 1, 1024, 32, 8, 128),
        (8, 1, 4096, 32, 8, 128),
        (4, 1, 4096, 32, 4, 128),  # GQA: 4 KV heads
        # prefill
        (4, 128, 1024, 32, 8, 128),
        (2, 256, 4096, 32, 8, 128),
        (1, 128, 4096, 16, 8, 128),
        # large head dim
        (1, 1, 1024, 32, 8, 256),
        (4, 1, 4096, 32, 8, 256),
    ]


def _build_inputs(num_seqs, query_len, kv_len, n_q_heads, n_kv_heads, head_size):
    """Build paged attention inputs with contiguous block tables."""
    total_q = num_seqs * query_len
    blocks_needed = num_seqs * ((kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
    num_blocks = max(blocks_needed + 8, 256)

    q = torch.randn(total_q, n_q_heads, head_size, dtype=DTYPE, device=device)
    k_cache = torch.randn(
        num_blocks, BLOCK_SIZE, n_kv_heads, head_size, dtype=DTYPE, device=device
    )
    v_cache = torch.randn_like(k_cache)

    # Build block table: each sequence gets contiguous blocks
    max_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.zeros(num_seqs, max_blocks, dtype=torch.int32, device=device)
    offset = 0
    for s in range(num_seqs):
        n_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_tables[s, :n_blocks] = torch.arange(
            offset, offset + n_blocks, device=device
        )
        offset += n_blocks

    seqlens_k = torch.full((num_seqs,), kv_len, dtype=torch.int32, device=device)
    cu = torch.tensor(
        [0] + [query_len] * num_seqs, dtype=torch.int32, device=device
    ).cumsum(0)
    scale = head_size**-0.5
    return q, k_cache, v_cache, block_tables, seqlens_k, cu, scale


def _ref_paged_attention(
    q,
    k_cache,
    v_cache,
    block_tables,
    seqlens_k,
    cu_seqlens_q,
    softmax_scale,
    causal=True,
):
    """Pure-PyTorch reference: unpacks paged KV cache and computes attention.

    This is the golden reference for correctness. It unpacks the paged KV
    cache into dense tensors and uses standard torch.einsum + softmax.
    """
    num_seqs = len(seqlens_k)
    _, block_size, num_kv_heads, head_size = k_cache.shape
    num_query_heads = q.shape[1]
    outputs: list[torch.Tensor] = []

    for i in range(num_seqs):
        q_start = cu_seqlens_q[i].item()
        q_end = cu_seqlens_q[i + 1].item()
        kv_len = seqlens_k[i].item()
        query_i = q[q_start:q_end]
        query_len = q_end - q_start

        # Unpack paged KV cache to dense [kv_len, heads, head_size]
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]
        k = k_cache[block_indices].reshape(-1, num_kv_heads, head_size)[:kv_len]
        v = v_cache[block_indices].reshape(-1, num_kv_heads, head_size)[:kv_len]

        # GQA: repeat KV heads to match query heads
        if num_query_heads != num_kv_heads:
            k = k.repeat_interleave(num_query_heads // num_kv_heads, dim=1)
            v = v.repeat_interleave(num_query_heads // num_kv_heads, dim=1)

        # Attention: Q·K^T, causal mask, softmax, weighted V
        attn = torch.einsum("qhd,khd->hqk", query_i * softmax_scale, k).float()
        if causal:
            mask = torch.triu(
                torch.ones(query_len, kv_len, device=q.device),
                diagonal=kv_len - query_len + 1,
            ).bool()
            attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        outputs.append(torch.einsum("hqk,khd->qhd", attn, v))

    return torch.cat(outputs, dim=0)


@pytest.mark.unified_attention
@pytest.mark.parametrize(
    "num_seqs, query_len, kv_len, n_q_heads, n_kv_heads, head_size",
    TEST_SHAPES,
    ids=[
        f"B{s}_Q{ql}_KV{kvl}_QH{qh}_KVH{kh}_D{d}"
        for s, ql, kvl, qh, kh, d in TEST_SHAPES
    ],
)
def test_unified_attention(
    num_seqs, query_len, kv_len, n_q_heads, n_kv_heads, head_size
):
    """Test unified_attention against PyTorch reference for correctness."""
    torch.manual_seed(0)

    q, kc, vc, bt, slens, cu, scale = _build_inputs(
        num_seqs, query_len, kv_len, n_q_heads, n_kv_heads, head_size
    )

    # Compute reference output (PyTorch eager, considered golden)
    out_ref = _ref_paged_attention(q, kc, vc, bt, slens, cu, scale, causal=True)

    # Compute Triton output
    out_triton = unified_attention(
        q,
        kc,
        vc,
        bt,
        slens,
        cu,
        query_len,
        softmax_scale=scale,
        causal=True,
    )

    # Compare: both should match within numerical tolerance for bf16 attention
    torch.testing.assert_close(
        out_triton.float(),
        out_ref.float(),
        rtol=RTOL,
        atol=ATOL,
    )
