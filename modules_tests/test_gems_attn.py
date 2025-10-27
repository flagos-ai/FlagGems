from typing import List, Optional, Tuple

import pytest
import torch

import flag_gems
from modules_tests.module_test_util import init_seed

try:
    import vllm.vllm_flash_attn.flash_attn_interface as _vllm  # noqa: F401

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

device = flag_gems.device


def attn_bias_from_alibi_slopes(slopes, seqlen_q, seqlen_k, causal=False):
    batch, nheads = slopes.shape
    device = slopes.device
    slopes = slopes.unsqueeze(-1).unsqueeze(-1)
    if causal:
        return (
            torch.arange(-seqlen_k + 1, 1, device=device, dtype=torch.float32) * slopes
        )

    row_idx = torch.arange(seqlen_q, device=device, dtype=torch.long).unsqueeze(-1)
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    relative_pos = torch.abs(row_idx + seqlen_k - seqlen_q - col_idx)
    return -slopes * relative_pos.to(dtype=slopes.dtype)


@pytest.mark.skipif(flag_gems.vendor_name == "mthreads", reason="RESULT TODOFIX")
@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RESULT TODOFIX")
@pytest.mark.flash_attn_varlen_func
@pytest.mark.parametrize("seq_lens", [[(1, 1328), (5, 18), (129, 463)]])
@pytest.mark.parametrize("num_heads", [(4, 4), (8, 2), (16, 2)])
@pytest.mark.parametrize("head_size", [128, 192, 256])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("sliding_window", [None])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("num_blocks", [32768, 2048])
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is not installed")
@torch.inference_mode()
def test_flash_attn_varlen_func(
    seq_lens: List[Tuple[int, int]],
    num_heads: Tuple[int, int],
    head_size: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    block_size: int,
    alibi: bool,
    soft_cap: Optional[float],
    num_blocks: int,
) -> None:
    # (Issue) numerical stability concern
    if alibi is True and soft_cap is not None:
        return

    with torch.device(flag_gems.device):
        init_seed(1234567890)
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
        cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
            dim=0, dtype=torch.int32
        )
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(
            0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
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
            alibi_slopes, attn_bias = None, None  # noqa: F841

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

    if HAS_VLLM:
        import vllm.vllm_flash_attn.flash_attn_interface as vllm_

        vllm_window = (
            None if sliding_window is None else [sliding_window, sliding_window]
        )
        if vllm_window is None:
            vllm_window = [-1, -1]

        vllm_out = vllm_.flash_attn_varlen_func(
            q=query,  # [sum_q, num_q_heads, head_size]
            k=key_cache,  # paged KV cache: [num_blocks, block_size, num_kv_heads, head_size]
            v=value_cache,  # same key cache
            max_seqlen_q=max_query_len,  # int
            cu_seqlens_q=cu_query_lens,  # [num_seqs+1]
            max_seqlen_k=max_kv_len,  # int
            cu_seqlens_k=None,  # paged mode not used
            seqused_k=torch.tensor(kv_lens, dtype=torch.int32, device=query.device),
            q_v=None,  # unused
            dropout_p=0.0,  # unused
            softmax_scale=scale,  # same
            causal=causal,
            window_size=vllm_window,  # List[int]
            softcap=soft_cap if soft_cap is not None else 0.0,
            alibi_slopes=alibi_slopes,
            deterministic=False,
            return_attn_probs=False,
            block_table=block_tables,  # paged KV
            return_softmax_lse=False,
            out=None,
            scheduler_metadata=None,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            fa_version=2,
        )

        # 数值对比
        torch.testing.assert_close(
            output, vllm_out, atol=2e-2, rtol=1e-2
        ), f"max abs diff vs vLLM: {torch.max(torch.abs(output - vllm_out))}"
