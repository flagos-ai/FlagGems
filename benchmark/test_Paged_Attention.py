import pytest
import torch

import flag_gems

from . import base


class PagedAttentionBenchmark(base.GenericBenchmark):
    """
    benchmark for Paged_Attention
    """

    def set_shapes(self, shape_file_path=None):
        # Override shapes directly
        self.shapes = [
            # (batch_size, num_query_heads, num_kv_heads, query_len, kv_len, head_dim, block_size, num_blocks)
            (1, 4, 4, 512, 512, 128, 16, 64),
            (1, 8, 8, 1024, 1024, 128, 16, 128),
            (1, 16, 16, 2048, 2048, 128, 16, 256),
            (4, 4, 4, 512, 512, 128, 16, 64),
            (4, 8, 8, 1024, 1024, 128, 16, 128),
        ]
        self.shape_desc = (
            "batch_size, num_query_heads, num_kv_heads, query_len, kv_len, "
            "head_dim, block_size, num_blocks"
        )

    def set_more_shapes(self):
        return None


def torch_paged_attention_ref(
    query, key_cache, value_cache, query_lens, kv_lens, block_tables, scale
):
    """Reference paged attention using torch operations."""
    num_seqs = len(query_lens)
    block_size = key_cache.shape[1]
    num_kv_heads = key_cache.shape[2]
    head_size = key_cache.shape[3]

    outputs = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len].clone()
        q = q * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks].cpu().numpy()

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k)
        empty_mask = torch.ones(query_len, kv_len, device=q.device)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        attn.masked_fill_(mask, float("-inf"))

        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def paged_attention_input_fn(shape, dtype, device):
    (
        batch_size,
        num_query_heads,
        num_kv_heads,
        query_len,
        kv_len,
        head_dim,
        block_size,
        num_blocks,
    ) = shape
    query_lens = [query_len // batch_size] * batch_size
    kv_lens = [kv_len // batch_size] * batch_size
    query = torch.randn(
        sum(query_lens), num_query_heads, head_dim, device=device, dtype=dtype
    )
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    blocks_per_seq = [(kv_len + block_size - 1) // block_size for kv_len in kv_lens]
    max_num_blocks_per_seq = max(blocks_per_seq)
    block_tables = torch.zeros(
        batch_size, max_num_blocks_per_seq, dtype=torch.int32, device=device
    )
    if sum(blocks_per_seq) <= num_blocks:
        next_block = 0
        for seq_idx, num_seq_blocks in enumerate(blocks_per_seq):
            block_tables[seq_idx, :num_seq_blocks] = torch.arange(
                next_block,
                next_block + num_seq_blocks,
                dtype=torch.int32,
                device=device,
            )
            next_block += num_seq_blocks
    else:
        block_tables = torch.randint(
            0,
            num_blocks,
            (batch_size, max_num_blocks_per_seq),
            dtype=torch.int32,
            device=device,
        )
    scale = head_dim**-0.5

    yield query, key_cache, value_cache, query_lens, kv_lens, block_tables, scale


@pytest.mark.Paged_Attention
def test_Paged_Attention():
    """
    Benchmark Paged_Attention through the shared benchmark framework.
    """
    bench = PagedAttentionBenchmark(
        op_name="Paged_Attention",
        input_fn=paged_attention_input_fn,
        torch_op=torch_paged_attention_ref,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_gems(flag_gems.Paged_Attention)
    bench.run()
