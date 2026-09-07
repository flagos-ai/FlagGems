from typing import List, Tuple

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

device = flag_gems.device
vendor_name = flag_gems.vendor_name


def _is_hopper():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


if QUICK_MODE:
    NUM_HEADS = [(8, 2)]
    HEAD_SIZES = [128]
    FLOAT_DTYPES = [torch.float16]
    NUM_BLOCKS = [2048]
else:
    NUM_HEADS = [(4, 4), (8, 2), (16, 2)]
    HEAD_SIZES = [64, 128, 256]
    FLOAT_DTYPES = [torch.float16, torch.bfloat16]
    NUM_BLOCKS = [32768, 2048]


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens: List[int],
    block_tables: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Pure-PyTorch reference for paged causal attention."""
    block_tables_np = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(len(query_lens)):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len].clone() * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_idx = block_tables_np[i, :num_kv_blocks]
        k = key_cache[block_idx].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_idx].view(-1, num_kv_heads, head_size)[:kv_len]

        if q.shape[1] != k.shape[1]:
            ratio = q.shape[1] // k.shape[1]
            k = torch.repeat_interleave(k, ratio, dim=1)
            v = torch.repeat_interleave(v, ratio, dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k)
        ones = torch.ones(query_len, kv_len, device=q.device)
        mask = torch.triu(ones, diagonal=kv_len - query_len + 1).bool()
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)
        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(not _is_hopper(), reason="FA3 requires Hopper (sm_90+)")
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="Not supported")
@pytest.mark.skipif(vendor_name == "hygon", reason="Not working")
@pytest.mark.parametrize("seq_lens", [[(1, 1328), (5, 18), (129, 463)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
def test_flash_attn_varlen_fa3_func(
    seq_lens: List[Tuple[int, int]],
    num_heads: Tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    num_blocks: int,
) -> None:
    with torch.device(device):
        utils.init_seed(1234567890)

        num_seqs = len(seq_lens)
        query_lens = [x[0] for x in seq_lens]
        kv_lens = [x[1] for x in seq_lens]
        num_query_heads, num_kv_heads = num_heads
        assert num_query_heads % num_kv_heads == 0

        max_query_len = max(query_lens)
        max_kv_len = max(kv_lens)
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
            window_size=(-1, -1),
            block_table=block_tables,
            softcap=0,
            fa_version=3,
        )

        ref_output = ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=block_tables,
            scale=scale,
        )

        msg = f"max_diff={torch.max(torch.abs(output - ref_output))}"
        torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2, msg=msg)


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(not _is_hopper(), reason="FA3 requires Hopper (sm_90+)")
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="Not supported")
@pytest.mark.skipif(vendor_name == "hygon", reason="Not working")
@pytest.mark.parametrize("seq_lens", [[(1, 512)] * 32, [(1, 2048)] * 16])
@pytest.mark.parametrize("num_heads", [(16, 8), (8, 2)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_blocks", [2048])
def test_flash_attn_varlen_fa3_func_decode(
    seq_lens: List[Tuple[int, int]],
    num_heads: Tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    num_blocks: int,
) -> None:
    with torch.device(device):
        utils.init_seed(42)

        num_seqs = len(seq_lens)
        query_lens = [x[0] for x in seq_lens]
        kv_lens = [x[1] for x in seq_lens]
        num_query_heads, num_kv_heads = num_heads
        max_query_len = max(query_lens)
        max_kv_len = max(kv_lens)
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
            window_size=(-1, -1),
            block_table=block_tables,
            softcap=0,
            fa_version=3,
        )

        ref_output = ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=block_tables,
            scale=scale,
        )

        msg = f"max_diff={torch.max(torch.abs(output - ref_output))}"
        torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2, msg=msg)


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(not _is_hopper(), reason="FA3 requires Hopper (sm_90+)")
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="Not supported")
@pytest.mark.skipif(vendor_name == "hygon", reason="Not working")
@pytest.mark.parametrize("seq_lens", [[(512, 512), (256, 256), (128, 128)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_flash_attn_varlen_fa3_func_non_paged(
    seq_lens: List[Tuple[int, int]],
    num_heads: Tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
) -> None:
    """non-paged path: k/v are flat [total_kv, nhead, d] tensors."""
    with torch.device(device):
        utils.init_seed(1234567890)

        query_lens = [x[0] for x in seq_lens]
        kv_lens = [x[1] for x in seq_lens]
        num_query_heads, num_kv_heads = num_heads
        assert num_query_heads % num_kv_heads == 0

        max_query_len = max(query_lens)
        max_kv_len = max(kv_lens)
        scale = head_size**-0.5

        query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
        key = torch.randn(sum(kv_lens), num_kv_heads, head_size, dtype=dtype)
        value = torch.randn_like(key)

        cu_query_lens = torch.tensor(
            [0] + query_lens, dtype=torch.int32, device=device
        ).cumsum(dim=0, dtype=torch.int32)
        cu_kv_lens = torch.tensor(
            [0] + kv_lens, dtype=torch.int32, device=device
        ).cumsum(dim=0, dtype=torch.int32)

        output = flag_gems.flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_query_lens,
            cu_seqlens_k=cu_kv_lens,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            softcap=0,
            fa_version=3,
        )

        outputs_ref = []
        q_off, k_off = 0, 0
        for ql, kl in zip(query_lens, kv_lens):
            q_i = query[q_off : q_off + ql].clone() * scale
            k_i = key[k_off : k_off + kl]
            v_i = value[k_off : k_off + kl]
            if num_query_heads != num_kv_heads:
                ratio = num_query_heads // num_kv_heads
                k_i = torch.repeat_interleave(k_i, ratio, dim=1)
                v_i = torch.repeat_interleave(v_i, ratio, dim=1)
            attn = torch.einsum("qhd,khd->hqk", q_i, k_i)
            ones = torch.ones(ql, kl, device=device)
            mask = torch.triu(ones, diagonal=kl - ql + 1).bool()
            attn.masked_fill_(mask, float("-inf"))
            attn = torch.softmax(attn, dim=-1).to(v_i.dtype)
            outputs_ref.append(torch.einsum("hqk,khd->qhd", attn, v_i))
            q_off += ql
            k_off += kl

        ref_output = torch.cat(outputs_ref, dim=0)
        msg = f"max_diff={torch.max(torch.abs(output - ref_output))}"
        torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2, msg=msg)
