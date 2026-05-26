import pytest
import torch

import flag_gems.testing as fg_testing
from flag_gems.fused.deepseek_v4_attention_dequantize_and_gather_k_cache import (
    dequantize_and_gather_k_cache,
)
from flag_gems.utils.device_info import get_device_capability

try:
    from vllm.v1.attention.ops.deepseek_v4_ops import (
        dequantize_and_gather_k_cache as vllm_dequantize_and_gather_k_cache,
    )

    _HAS_VLLM_DEQUANTIZE_AND_GATHER_K_CACHE = True
except Exception:
    vllm_dequantize_and_gather_k_cache = None
    _HAS_VLLM_DEQUANTIZE_AND_GATHER_K_CACHE = False


def is_support_fp8e4nv():
    major, minor = get_device_capability()
    return major * 10 + minor >= 89


def _fill_cache(k_cache, expected_rows, block_size, nope_dim, rope_dim, scale_slots):
    token_data_size = nope_dim + rope_dim * 2
    for slot, row in expected_rows.items():
        block = slot // block_size
        pos = slot % block_size
        base = pos * token_data_size
        x = (
            torch.arange(nope_dim, device=k_cache.device, dtype=torch.float32) / 32.0
            + slot / 8.0
        ).to(torch.float8_e4m3fn)
        rope = (
            torch.arange(rope_dim, device=k_cache.device, dtype=torch.float32) / 16.0
            + slot
        ).to(torch.bfloat16)
        k_cache[block, base : base + nope_dim].copy_(x.view(torch.uint8))
        k_cache[block, base + nope_dim : base + nope_dim + rope_dim * 2].copy_(
            rope.view(torch.uint8)
        )
        scale_base = block_size * token_data_size + pos * scale_slots
        k_cache[block, scale_base : scale_base + scale_slots] = 127
        row[..., :nope_dim] = x.to(torch.float32).to(torch.bfloat16)
        row[..., nope_dim : nope_dim + rope_dim] = rope


@pytest.mark.parametrize(
    ("batch", "seq_len", "gather_len", "block_size", "nope_dim", "rope_dim"),
    [
        (1, 6, 3, 4, 64, 16),
        (2, 12, 5, 64, 448, 64),
    ],
)
@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_support_fp8e4nv(),
    reason="requires cuda with fp8e4nv support (capability >= 89)",
)
def test_dequantize_and_gather_k_cache_accuracy(
    batch, seq_len, gather_len, block_size, nope_dim, rope_dim
):
    device = "cuda"
    scale_slots = (nope_dim + 63) // 64 + (1 if nope_dim % 64 == 0 else 0)
    output_dim = nope_dim + rope_dim
    token_data_size = nope_dim + rope_dim * 2
    block_stride = block_size * token_data_size + block_size * scale_slots
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = batch * blocks_per_seq
    k_cache = torch.zeros((num_blocks, block_stride), device=device, dtype=torch.uint8)
    out = torch.empty(
        (batch, gather_len, output_dim), device=device, dtype=torch.bfloat16
    )
    expected = torch.empty_like(out)
    rows = {}
    for req in range(batch):
        start_pos = seq_len - gather_len
        for local_i in range(gather_len):
            pos = start_pos + local_i
            physical_block = req * blocks_per_seq + pos // block_size
            slot = physical_block * block_size + pos % block_size
            rows[slot] = expected[req : req + 1, local_i : local_i + 1, :]
    _fill_cache(k_cache, rows, block_size, nope_dim, rope_dim, scale_slots)

    seq_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    gather_lens = torch.full((batch,), gather_len, device=device, dtype=torch.int32)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(
        batch, blocks_per_seq
    )
    dequantize_and_gather_k_cache(
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        rope_dim=rope_dim,
        nope_dim=nope_dim,
        scale_slots=scale_slots,
    )

    fg_testing.assert_close(out, expected, dtype=torch.bfloat16, equal_nan=True)


@pytest.mark.skipif(
    (not torch.cuda.is_available())
    or (not is_support_fp8e4nv())
    or (not _HAS_VLLM_DEQUANTIZE_AND_GATHER_K_CACHE),
    reason="requires cuda with fp8e4nv support and vllm deepseek_v4_ops.dequantize_and_gather_k_cache",
)
def test_dequantize_and_gather_k_cache_vllm_accuracy():
    device = "cuda"
    batch = 2
    seq_len = 12
    gather_len = 5
    block_size = 64
    nope_dim = 448
    rope_dim = 64
    scale_slots = 8
    output_dim = nope_dim + rope_dim
    token_data_size = nope_dim + rope_dim * 2
    block_stride = block_size * token_data_size + block_size * scale_slots
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = batch * blocks_per_seq
    k_cache = torch.zeros((num_blocks, block_stride), device=device, dtype=torch.uint8)
    expected_rows = torch.empty(
        (batch, gather_len, output_dim), device=device, dtype=torch.bfloat16
    )
    rows = {}
    for req in range(batch):
        start_pos = seq_len - gather_len
        for local_i in range(gather_len):
            pos = start_pos + local_i
            physical_block = req * blocks_per_seq + pos // block_size
            slot = physical_block * block_size + pos % block_size
            rows[slot] = expected_rows[req : req + 1, local_i : local_i + 1, :]
    _fill_cache(k_cache, rows, block_size, nope_dim, rope_dim, scale_slots)

    actual = torch.empty_like(expected_rows)
    expected = torch.empty_like(expected_rows)
    seq_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    gather_lens = torch.full((batch,), gather_len, device=device, dtype=torch.int32)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(
        batch, blocks_per_seq
    )

    dequantize_and_gather_k_cache(
        actual,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        rope_dim=rope_dim,
        nope_dim=nope_dim,
        scale_slots=scale_slots,
    )
    vllm_dequantize_and_gather_k_cache(
        expected,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        0,
    )

    fg_testing.assert_close(actual, expected, dtype=torch.bfloat16, equal_nan=True)
