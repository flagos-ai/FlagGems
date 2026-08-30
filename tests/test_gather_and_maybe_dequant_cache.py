import pytest
import torch

from flag_gems.runtime.backend._thead.fused.gather_and_maybe_dequant_cache import (
    gather_and_maybe_dequant_cache,
)


def _torch_gather_baseline(
    src_cache,
    dst,
    block_table,
    cu_seq_lens,
    token_to_seq,
    num_tokens,
    kv_cache_dtype,
    scale,
    seq_starts,
):
    """Pure PyTorch vectorized reference for gather_and_maybe_dequant_cache."""
    block_size = src_cache.shape[1]
    entry_size = dst.shape[-1]
    device = dst.device

    batch_ids = token_to_seq[:num_tokens].long()
    batch_starts = cu_seq_lens[batch_ids].long()
    token_indices = torch.arange(num_tokens, device=device)
    batch_offsets = token_indices - batch_starts

    if seq_starts is not None:
        offsets = seq_starts[batch_ids].long()
        batch_offsets = batch_offsets + offsets

    block_table_ids = batch_offsets // block_size
    slot_ids = batch_offsets % block_size

    block_ids = block_table[batch_ids, block_table_ids.long()].long()
    src_data = src_cache[block_ids, slot_ids.long(), :entry_size]

    if kv_cache_dtype == "auto":
        dst[:num_tokens, :entry_size] = src_data.to(dst.dtype)
    else:
        scale_val = scale.item() if scale.numel() == 1 else scale[0].item()
        dst[:num_tokens, :entry_size] = (src_data.float() * scale_val).to(dst.dtype)


def _make_inputs(num_seqs, seq_len, entry_size, block_size=16, device="cuda"):
    """Create test inputs simulating MLA chunked prefill gather."""
    num_tokens = num_seqs * seq_len
    max_blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = num_seqs * max_blocks_per_seq + 1

    src_cache = torch.randn(
        num_blocks, block_size, entry_size, device=device, dtype=torch.bfloat16
    )
    dst_triton = torch.zeros(
        num_tokens, entry_size, device=device, dtype=torch.bfloat16
    )
    dst_ref = torch.zeros(num_tokens, entry_size, device=device, dtype=torch.bfloat16)

    block_table = torch.zeros(
        num_seqs, max_blocks_per_seq, device=device, dtype=torch.int32
    )
    for b in range(num_seqs):
        for i in range(max_blocks_per_seq):
            block_table[b, i] = b * max_blocks_per_seq + i

    seq_lens_t = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)
    cu_seq_lens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    cu_seq_lens[1:] = torch.cumsum(seq_lens_t, dim=0)

    token_to_seq = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    for b in range(num_seqs):
        start = cu_seq_lens[b].item()
        end = cu_seq_lens[b + 1].item()
        token_to_seq[start:end] = b

    scale = torch.tensor([1.0], device=device, dtype=torch.float32)
    return (
        src_cache,
        dst_triton,
        dst_ref,
        block_table,
        cu_seq_lens,
        token_to_seq,
        num_tokens,
        scale,
    )


# DeepSeekV4: entry_size=576, DeepSeekV2: entry_size=320
@pytest.mark.parametrize(
    "num_seqs,seq_len,entry_size,block_size",
    [
        (1, 16, 576, 16),
        (4, 64, 576, 16),
        (8, 256, 576, 16),
        (16, 1024, 576, 16),
        (64, 64, 576, 16),
        (64, 250, 576, 16),
        (1, 16, 320, 16),
        (4, 64, 320, 16),
        (8, 256, 320, 16),
        (64, 64, 320, 16),
        # Different block sizes
        (4, 128, 576, 8),
        (4, 128, 576, 32),
        (4, 128, 576, 64),
    ],
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA",
)
def test_gather_and_maybe_dequant_cache_auto(num_seqs, seq_len, entry_size, block_size):
    """Test gather with kv_cache_dtype='auto' (BF16 direct copy)."""
    torch.manual_seed(42)
    device = "cuda"
    (
        src_cache,
        dst_triton,
        dst_ref,
        block_table,
        cu_seq_lens,
        token_to_seq,
        num_tokens,
        scale,
    ) = _make_inputs(num_seqs, seq_len, entry_size, block_size, device)

    _torch_gather_baseline(
        src_cache,
        dst_ref,
        block_table,
        cu_seq_lens,
        token_to_seq,
        num_tokens,
        "auto",
        scale,
        None,
    )

    gather_and_maybe_dequant_cache(
        src_cache,
        dst_triton,
        block_table,
        cu_seq_lens,
        token_to_seq,
        num_tokens,
        "auto",
        scale,
        None,
    )

    # Direct copy should be exact
    assert torch.equal(
        dst_triton, dst_ref
    ), f"max_diff={(dst_triton - dst_ref).abs().max().item()}"


@pytest.mark.parametrize(
    "num_seqs,seq_len,entry_size",
    [
        (1, 16, 576),
        (4, 64, 576),
        (8, 256, 576),
        (1, 16, 320),
        (4, 64, 320),
    ],
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA",
)
def test_gather_and_maybe_dequant_cache_fp8(num_seqs, seq_len, entry_size):
    """Test gather with FP8 dequantization."""
    torch.manual_seed(42)
    device = "cuda"
    block_size = 16
    num_tokens = num_seqs * seq_len
    max_blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = num_seqs * max_blocks_per_seq + 1

    src_cache = torch.randint(
        0, 255, (num_blocks, block_size, entry_size), device=device, dtype=torch.uint8
    )
    dst_triton = torch.zeros(
        num_tokens, entry_size, device=device, dtype=torch.bfloat16
    )
    dst_ref = torch.zeros(num_tokens, entry_size, device=device, dtype=torch.bfloat16)

    block_table = torch.zeros(
        num_seqs, max_blocks_per_seq, device=device, dtype=torch.int32
    )
    for b in range(num_seqs):
        for i in range(max_blocks_per_seq):
            block_table[b, i] = b * max_blocks_per_seq + i

    seq_lens_t = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)
    cu_seq_lens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    cu_seq_lens[1:] = torch.cumsum(seq_lens_t, dim=0)

    token_to_seq = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    for b in range(num_seqs):
        start = cu_seq_lens[b].item()
        end = cu_seq_lens[b + 1].item()
        token_to_seq[start:end] = b

    scale = torch.tensor([0.01], device=device, dtype=torch.float32)

    _torch_gather_baseline(
        src_cache,
        dst_ref,
        block_table,
        cu_seq_lens,
        token_to_seq,
        num_tokens,
        "fp8_e4m3",
        scale,
        None,
    )

    gather_and_maybe_dequant_cache(
        src_cache,
        dst_triton,
        block_table,
        cu_seq_lens,
        token_to_seq,
        num_tokens,
        "fp8_e4m3",
        scale,
        None,
    )

    torch.testing.assert_close(dst_triton, dst_ref, rtol=1e-2, atol=1e-2)
