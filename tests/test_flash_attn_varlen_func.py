from typing import List, Optional, Tuple

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

device = flag_gems.device
vendor_name = flag_gems.vendor_name


# Following varlen and paged attn tests are copied from
# https://github.com/vllm-project/flash-attention/blob/main/tests/test_vllm_flash_attn.py
def attn_bias_from_alibi_slopes(slopes, seqlen_q, seqlen_k, causal=False):
    device = slopes.device
    slopes = slopes.unsqueeze(-1).unsqueeze(-1)

    if causal:
        v = torch.arange(-seqlen_k + 1, 1, device=device, dtype=torch.float32)
        return v * slopes

    row_idx = torch.arange(seqlen_q, device=device, dtype=torch.long).unsqueeze(-1)
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    relative_pos = torch.abs(row_idx + seqlen_k - seqlen_q - col_idx)

    return -slopes * relative_pos.to(dtype=slopes.dtype)


def ref_paged_attn(
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
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        # clone to avoid clobbering the query tensor
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


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="#2815: Not supported")
@pytest.mark.skipif(vendor_name == "hygon", reason="#2816: Not working")
@pytest.mark.parametrize("seq_lens", [[(1, 1328), (5, 18), (129, 463)]])
@pytest.mark.parametrize("num_heads", [(4, 4), (8, 2), (16, 2)])
@pytest.mark.parametrize("head_size", [128, 192, 256])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("sliding_window", [None])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("num_blocks", [32768, 2048])
@pytest.mark.parametrize("optimize_init", [False, True])
@torch.inference_mode()
def test_flash_attn_varlen_func(
    monkeypatch,
    seq_lens: List[Tuple[int, int]],
    num_heads: Tuple[int, int],
    head_size: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    block_size: int,
    alibi: bool,
    soft_cap: Optional[float],
    num_blocks: int,
    optimize_init: bool,
) -> None:
    if vendor_name == "mthreads":
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    # (Issue) numerical stability concern
    if alibi is True and soft_cap is not None:
        return

    with torch.device(flag_gems.device):
        utils.init_seed(1234567890)

        if vendor_name == "cambricon":
            torch.manual_seed(123456)
            torch.mlu.manual_seed_all(123456)

        num_seqs = len(seq_lens)
        query_lens = [x[0] for x in seq_lens]
        kv_lens = [x[1] for x in seq_lens]
        num_query_heads = num_heads[0]
        num_kv_heads = num_heads[1]
        assert num_query_heads % num_kv_heads == 0
        max_query_len = max(query_lens)
        max_kv_len = max(kv_lens)
        window_size = (
            (sliding_window, sliding_window) if sliding_window is not None else (-1, -1)
        )
        scale = head_size**-0.5
        query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
        key_cache = torch.randn(
            num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
        )
        value_cache = torch.randn_like(key_cache)
        cu_query_lens = torch.tensor(
            [0] + query_lens, dtype=torch.int32, device=device
        ).cumsum(dim=0, dtype=torch.int32)
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(
            0,
            num_blocks,
            (num_seqs, max_num_blocks_per_seq),
            dtype=torch.int32,
            device=device,
        )

        causal = True

        if alibi:
            # alibi_slopes = torch.rand(num_seqs, num_query_heads, device=device, dtype=torch.float32) * 0.3
            alibi_slopes = (
                torch.ones(
                    num_seqs, num_query_heads, device=device, dtype=torch.float32
                )
                * 0.3
            )
            attn_bias = attn_bias_from_alibi_slopes(
                alibi_slopes, max_query_len, max_kv_len, causal=causal
            )
        else:
            alibi_slopes, attn_bias = None, None

        if vendor_name == "cambricon":
            output = flag_gems.flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=cu_query_lens,
                seqused_k=seqused_k,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_kv_len,
                softmax_scale=scale,
                causal=causal,
                window_size=window_size,
                block_table=block_tables,
                softcap=soft_cap if soft_cap is not None else 0,
                alibi_slopes=alibi_slopes,
                fa_version=2,
            )
        else:
            if optimize_init:
                output = flag_gems.ops.flash_attn_varlen_opt_func(
                    q=query,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=cu_query_lens,
                    seqused_k=seqused_k,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_kv_len,
                    softmax_scale=scale,
                    causal=causal,
                    window_size=window_size,
                    block_table=block_tables,
                    softcap=soft_cap if soft_cap is not None else 0,
                    alibi_slopes=alibi_slopes,
                    fa_version=2,
                )
            else:
                output = flag_gems.ops.flash_attn_varlen_func(
                    q=query,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=cu_query_lens,
                    seqused_k=seqused_k,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_kv_len,
                    softmax_scale=scale,
                    causal=causal,
                    window_size=window_size,
                    block_table=block_tables,
                    softcap=soft_cap if soft_cap is not None else 0,
                    alibi_slopes=alibi_slopes,
                    fa_version=2,
                )

        ref_output = ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=block_tables,
            scale=scale,
            attn_bias=attn_bias,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
        )

        msg = f"{torch.max(torch.abs(output - ref_output))}"
        torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2, msg=msg)


@pytest.mark.skipif(vendor_name == "kunlunxin", reason="#2815: Not working")
@pytest.mark.skipif(vendor_name == "hygon", reason="#2816: Not working")
@pytest.mark.flash_attn_varlen_func
@pytest.mark.parametrize("seq_lens", [[(1, 1328), (1, 18), (1, 463)]])
@pytest.mark.parametrize("num_heads", [(8, 2)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("sliding_window", [None])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None, 10.0])
@pytest.mark.parametrize("num_blocks", [2048])
@torch.inference_mode()
def test_flash_attn_varlen_func_swap_qg(
    monkeypatch,
    seq_lens: List[Tuple[int, int]],
    num_heads: Tuple[int, int],
    head_size: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
) -> None:
    if vendor_name == "mthreads":
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    with torch.device(flag_gems.device):
        utils.init_seed(1234567890)
        num_seqs = len(seq_lens)
        query_lens = [x[0] for x in seq_lens]
        kv_lens = [x[1] for x in seq_lens]
        num_query_heads = num_heads[0]
        num_kv_heads = num_heads[1]
        assert num_query_heads % num_kv_heads == 0
        max_query_len = max(query_lens)
        max_kv_len = max(kv_lens)
        window_size = (
            (sliding_window, sliding_window) if sliding_window is not None else (-1, -1)
        )
        scale = head_size**-0.5
        query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
        key_cache = torch.randn(
            num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
        )
        value_cache = torch.randn_like(key_cache)
        cu_query_lens = torch.tensor(
            [0] + query_lens, dtype=torch.int32, device=device
        ).cumsum(dim=0, dtype=torch.int32)
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(
            0,
            num_blocks,
            (num_seqs, max_num_blocks_per_seq),
            dtype=torch.int32,
            device=device,
        )

        if vendor_name == "cambricon":
            output = flag_gems.flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=cu_query_lens,
                seqused_k=seqused_k,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_kv_len,
                softmax_scale=scale,
                causal=True,
                window_size=window_size,
                block_table=block_tables,
                softcap=soft_cap if soft_cap is not None else 0,
                fa_version=2,
            )
        else:
            output = flag_gems.ops.flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=cu_query_lens,
                seqused_k=seqused_k,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_kv_len,
                softmax_scale=scale,
                causal=True,
                window_size=window_size,
                block_table=block_tables,
                softcap=soft_cap if soft_cap is not None else 0,
                fa_version=2,
            )

        ref_output = ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=block_tables,
            scale=scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
        )

        torch.testing.assert_close(
            output, ref_output, atol=2e-2, rtol=1e-2
        ), f"{torch.max(torch.abs(output - ref_output))}"
