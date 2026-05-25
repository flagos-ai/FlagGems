import pytest
import torch

import flag_gems.testing as fg_testing
from flag_gems.fused.deepseek_v4_attention_dequantize_and_gather_k_cache import (
    dequantize_and_gather_k_cache,
)
from flag_gems.utils.device_info import get_device_capability


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


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_support_fp8e4nv(),
    reason="requires cuda with fp8e4nv support (capability >= 89)",
)
def test_dequantize_and_gather_k_cache_accuracy():
    device = "cuda"
    block_size = 4
    nope_dim = 64
    rope_dim = 16
    scale_slots = 2
    output_dim = nope_dim + rope_dim
    token_data_size = nope_dim + rope_dim * 2
    block_stride = block_size * token_data_size + block_size * scale_slots
    k_cache = torch.zeros((2, block_stride), device=device, dtype=torch.uint8)
    out = torch.empty((1, 3, output_dim), device=device, dtype=torch.bfloat16)
    expected = torch.empty_like(out)
    rows = {3: expected[:, 0:1, :], 4: expected[:, 1:2, :], 5: expected[:, 2:3, :]}
    _fill_cache(k_cache, rows, block_size, nope_dim, rope_dim, scale_slots)

    seq_lens = torch.tensor([6], device=device, dtype=torch.int32)
    gather_lens = torch.tensor([3], device=device, dtype=torch.int32)
    block_table = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
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
