import pytest
import torch

from flag_gems.runtime.backend._thead.fused.gather_and_maybe_dequant_cache import (
    gather_and_maybe_dequant_cache,
)

from . import base


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
    """Pure PyTorch vectorized baseline for gather_and_maybe_dequant_cache."""
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


def _torch_baseline_wrapper(
    src_cache,
    dst,
    block_table,
    cu_seq_lens,
    token_to_seq,
    num_tokens,
    kv_cache_dtype="auto",
    scale=None,
    seq_starts=None,
):
    """Wrapper with same signature as gems_op for benchmark framework."""
    if scale is None:
        scale = torch.tensor([1.0], device=dst.device, dtype=torch.float32)
    _torch_gather_baseline(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        token_to_seq,
        num_tokens,
        kv_cache_dtype,
        scale,
        seq_starts,
    )
    return dst


class GatherAndMaybeDequantCacheBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "gather_and_maybe_dequant_cache",
            _torch_baseline_wrapper,
            [torch.bfloat16],
            gems_op=gather_and_maybe_dequant_cache,
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        # (num_seqs, seq_len, entry_size) - typical MLA chunked prefill shapes
        self.shapes = [
            # DeepSeekV4 (entry_size=576)
            (64, 64, 576),  # 4k tokens - decode-like
            (64, 128, 576),  # 8k tokens
            (64, 250, 576),  # 16k tokens - prefill
            (64, 500, 576),  # 32k tokens - large prefill
            (16, 1024, 576),  # 16k tokens (fewer seqs, longer ctx)
            (8, 2048, 576),  # 16k tokens
            # DeepSeekV2 (entry_size=320)
            (64, 64, 320),  # 4k tokens
            (64, 250, 320),  # 16k tokens
        ]

    def get_input_iter(self, cur_dtype):
        device = self.device
        block_size = 16
        for num_seqs, seq_len, entry_size in self.shapes:
            num_tokens = num_seqs * seq_len
            max_blocks_per_seq = (seq_len + block_size - 1) // block_size
            num_blocks = num_seqs * max_blocks_per_seq + 1

            src_cache = torch.randn(
                num_blocks,
                block_size,
                entry_size,
                device=device,
                dtype=torch.bfloat16,
            )
            dst = torch.zeros(
                num_tokens, entry_size, device=device, dtype=torch.bfloat16
            )

            block_table = torch.zeros(
                num_seqs, max_blocks_per_seq, device=device, dtype=torch.int32
            )
            for b in range(num_seqs):
                for i in range(max_blocks_per_seq):
                    block_table[b, i] = b * max_blocks_per_seq + i

            seq_lens_t = torch.full(
                (num_seqs,), seq_len, dtype=torch.int32, device=device
            )
            cu_seq_lens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
            cu_seq_lens[1:] = torch.cumsum(seq_lens_t, dim=0)

            token_to_seq = torch.zeros(num_tokens, dtype=torch.int32, device=device)
            for b in range(num_seqs):
                start = cu_seq_lens[b].item()
                end = cu_seq_lens[b + 1].item()
                token_to_seq[start:end] = b

            scale = torch.tensor([1.0], device=device, dtype=torch.float32)

            yield (
                src_cache,
                dst,
                block_table,
                cu_seq_lens,
                token_to_seq,
                num_tokens,
                "auto",
                scale,
                None,
            )


@pytest.mark.gather_and_maybe_dequant_cache
def test_gather_and_maybe_dequant_cache_benchmark():
    bench = GatherAndMaybeDequantCacheBenchmark()
    bench.run()
