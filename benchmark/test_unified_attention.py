"""Benchmark for unified_attention (vLLM paged attention).

Production shapes match vLLM inference scenarios on H20 GPU:
  - decode: B=1/8/64, Q=1, KV=1K/4K, GQA and D=256 variants
  - prefill: Q=128/256/512, KV=1K/4K
  - Baselines against PyTorch reference (paged attention in eager mode).

The reference used for speedup calculation is the pure-PyTorch paged
attention implementation (v0-level correctness baseline, matching the
original vLLM Triton kernel output).
"""

import pytest
import torch

import flag_gems
from flag_gems.fused import unified_attention

from . import base, consts

device = flag_gems.device

BLOCK_SIZE = 16

# Production benchmark shapes:
# (name, num_seqs, query_len, kv_len, n_q_heads, n_kv_heads, head_size)
BENCH_SHAPES: list[tuple[str, int, int, int, int, int, int]] = [
    ("decode-B1-KV1K", 1, 1, 1024, 32, 8, 128),
    ("decode-B8-KV4K", 8, 1, 4096, 32, 8, 128),
    ("decode-B64-KV4K", 64, 1, 4096, 40, 8, 128),
    ("decode-D256-KV4K", 8, 1, 4096, 32, 8, 256),
    ("prefill-Q128-KV1K", 4, 128, 1024, 32, 8, 128),
    ("prefill-Q256-KV4K", 2, 256, 4096, 32, 8, 128),
    ("prefill-Q512-KV4K", 1, 512, 4096, 32, 8, 128),
    ("decode-GQA8-KV4K", 8, 1, 4096, 32, 4, 128),
]


def _build_inputs(num_seqs, query_len, kv_len, n_q_heads, n_kv_heads, head_size, dtype):
    """Build paged attention inputs with contiguous block tables."""
    total_q = num_seqs * query_len
    blocks_needed = num_seqs * ((kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
    num_blocks = max(blocks_needed + 16, 512)

    q = torch.randn(total_q, n_q_heads, head_size, dtype=dtype, device=device)
    k_cache = torch.randn(
        num_blocks, BLOCK_SIZE, n_kv_heads, head_size, dtype=dtype, device=device
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
    return q, k_cache, v_cache, block_tables, seqlens_k, cu, scale, query_len


def _torch_ref_fn(
    q,
    k_cache,
    v_cache,
    block_tables,
    seqlens_k,
    cu_seqlens_q,
    softmax_scale,
    query_len,
    causal=True,
):
    """Pure-PyTorch reference: unpacks paged KV cache and computes attention."""
    num_seqs = len(seqlens_k)
    _, block_size, num_kv_heads, head_size = k_cache.shape
    num_query_heads = q.shape[1]
    outputs: list[torch.Tensor] = []

    for i in range(num_seqs):
        q_start = cu_seqlens_q[i].item()
        q_end = cu_seqlens_q[i + 1].item()
        kv_len = seqlens_k[i].item()
        query_i = q[q_start:q_end]

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]
        k = k_cache[block_indices].reshape(-1, num_kv_heads, head_size)[:kv_len]
        v = v_cache[block_indices].reshape(-1, num_kv_heads, head_size)[:kv_len]

        if num_query_heads != num_kv_heads:
            k = k.repeat_interleave(num_query_heads // num_kv_heads, dim=1)
            v = v.repeat_interleave(num_query_heads // num_kv_heads, dim=1)

        attn = torch.einsum("qhd,khd->hqk", query_i * softmax_scale, k).float()
        if causal:
            mask = torch.triu(
                torch.ones(q_end - q_start, kv_len, device=q.device),
                diagonal=kv_len - (q_end - q_start) + 1,
            ).bool()
            attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        outputs.append(torch.einsum("hqk,khd->qhd", attn, v))

    return torch.cat(outputs, dim=0)


def _gems_op(q, kc, vc, bt, slens, cu, scale, max_qlen):
    """FlagGems Triton kernel wrapper matching get_input_iter signature."""
    return unified_attention(
        q,
        kc,
        vc,
        bt,
        slens,
        cu,
        max_qlen,
        softmax_scale=scale,
        causal=True,
    )


def _torch_op(q, kc, vc, bt, slens, cu, scale, max_qlen):
    """PyTorch reference wrapper matching get_input_iter signature."""
    return _torch_ref_fn(
        q,
        kc,
        vc,
        bt,
        slens,
        cu,
        scale,
        max_qlen,
        causal=True,
    )


class UnifiedAttentionBenchmark(base.Benchmark):
    """Benchmark for unified paged attention kernel."""

    DEFAULT_SHAPE_DESC = "num_seqs, query_len, kv_len, n_q_heads, n_kv_heads, head_size"

    def set_shapes(self, shape_file_path=None):
        self.shapes = BENCH_SHAPES

    def get_input_iter(self, dtype):
        for name, num_seqs, qlen, kvlen, n_qh, n_kvh, hd in self.shapes:
            torch.manual_seed(0)
            q, kc, vc, bt, slens, cu, scale, max_qlen = _build_inputs(
                num_seqs, qlen, kvlen, n_qh, n_kvh, hd, dtype
            )
            yield (q, kc, vc, bt, slens, cu, scale, max_qlen)


@pytest.mark.unified_attention
def test_unified_attention():
    bench = UnifiedAttentionBenchmark(
        op_name="unified_attention",
        torch_op=_torch_op,
        gems_op=_gems_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
