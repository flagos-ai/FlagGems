import os
from typing import List, Optional, Tuple

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def ref_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens: List[int],
    block_tables: torch.Tensor,
    scale: float,
    attn_bias: torch.Tensor = None,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    """Reference implementation for paged attention."""
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len].clone()
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

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
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))

        if attn_bias is not None:
            attn = attn + attn_bias[i, :, :query_len, :kv_len]

        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def _make_paged_attention_block_tables(
    kv_lens: List[int],
    block_size: int,
    num_blocks: int,
    device: str,
) -> torch.Tensor:
    blocks_per_seq = [(kv_len + block_size - 1) // block_size for kv_len in kv_lens]
    max_blocks_per_seq = max(blocks_per_seq)
    if sum(blocks_per_seq) <= num_blocks:
        block_tables = torch.zeros(
            len(kv_lens), max_blocks_per_seq, dtype=torch.int32, device=flag_gems.device
        )
        next_block = 0
        for seq_idx, num_seq_blocks in enumerate(blocks_per_seq):
            block_tables[seq_idx, :num_seq_blocks] = torch.arange(
                next_block,
                next_block + num_seq_blocks,
                dtype=torch.int32,
                device=flag_gems.device,
            )
            next_block += num_seq_blocks
        return block_tables

    return torch.randint(
        0,
        num_blocks,
        (len(kv_lens), max_blocks_per_seq),
        dtype=torch.int32,
        device=flag_gems.device,
    )


@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RESULT TODOFIX")
@pytest.mark.Paged_Attention
@pytest.mark.parametrize("seq_lens", [[(1, 1328), (5, 18), (129, 463)]])
@pytest.mark.parametrize("num_heads", [(4, 4), (8, 2)])
@pytest.mark.parametrize("head_size", [128, 256])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None, 10.0])
@pytest.mark.parametrize("softmax_scale", [None, "explicit"])
@pytest.mark.parametrize("num_blocks", [2048])
@torch.inference_mode()
def test_Paged_Attention(
    seq_lens: List[Tuple[int, int]],
    num_heads: Tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    softmax_scale: Optional[str],
    num_blocks: int,
) -> None:
    if flag_gems.vendor_name == "mthreads":
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    with torch.device(flag_gems.device):
        utils.init_seed(1234567890)
        query_lens = [x[0] for x in seq_lens]
        kv_lens = [x[1] for x in seq_lens]
        num_query_heads = num_heads[0]
        num_kv_heads = num_heads[1]
        assert num_query_heads % num_kv_heads == 0
        scale = head_size**-0.5

        query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
        key_cache = torch.randn(
            num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
        )
        value_cache = torch.randn_like(key_cache)

        block_tables = _make_paged_attention_block_tables(
            kv_lens, block_size, num_blocks, flag_gems.device
        )

        # Reference implementation
        ref_output = ref_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=block_tables,
            scale=scale,
            soft_cap=soft_cap,
        )

        # GEMS implementation
        with flag_gems.use_gems():
            gems_output = flag_gems.Paged_Attention(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                query_lens=query_lens,
                kv_lens=kv_lens,
                block_tables=block_tables,
                softmax_scale=scale if softmax_scale == "explicit" else None,
                softcap=soft_cap if soft_cap is not None else 0.0,
            )

        torch.testing.assert_close(
            gems_output, ref_output, atol=2e-2, rtol=1e-2
        ), f"{torch.max(torch.abs(gems_output - ref_output))}"

    if flag_gems.vendor_name == "mthreads":
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.Paged_Attention
def test_Paged_Attention_invalid_arguments() -> None:
    with torch.device(flag_gems.device):
        query = torch.randn(2, 4, 128, dtype=torch.float16)
        key_cache = torch.randn(4, 16, 4, 128, dtype=torch.float16)
        value_cache = torch.randn_like(key_cache)
        block_tables = torch.zeros(1, 1, dtype=torch.int64)

        with pytest.raises(TypeError, match="block_tables"):
            flag_gems.Paged_Attention(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                query_lens=[2],
                kv_lens=[16],
                block_tables=block_tables,
            )

        with pytest.raises(ValueError, match="sum\\(query_lens\\)"):
            flag_gems.Paged_Attention(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                query_lens=[1],
                kv_lens=[16],
                block_tables=block_tables.to(torch.int32),
            )
