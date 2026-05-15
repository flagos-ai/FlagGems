"""Benchmark for top_k_per_row_prefill (DeepSeek V4 sparse attention).

Shapes match DeepSeek V4 production config:
    - vocab_size=129280: DeepSeek V4 vocabulary size (full BPE vocab)
    - top_k=1024: number of KV cache slots selected per token in sparse attention
    - num_rows=1: single-token decode (latency-critical path)
    - num_rows=32: typical prefill micro-batch
    - num_rows=64: larger prefill batch
    - num_rows=2048: max prefill sequence length

The torch reference uses torch.topk directly (no masking), representing the
theoretical minimum for selection-only. The Triton kernel additionally handles
masking and index adjustment.
"""

import pytest
import torch

import flag_gems
from flag_gems.fused.top_k_per_row_prefill import top_k_per_row_prefill

from . import base

device = flag_gems.device

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required",
)


class TopKPerRowPrefillBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "num_rows, vocab_size, top_k"

    def set_shapes(self, shape_file_path=None):
        # DeepSeek V4 production shapes:
        # - (1, 129280, 1024): decode path, single token, latency-critical
        # - (32, 129280, 1024): typical prefill micro-batch
        # - (64, 129280, 1024): larger prefill batch
        # - (2048, 129280, 1024): max sequence length prefill
        self.shapes = [
            (1, 129280, 1024),
            (32, 129280, 1024),
            (64, 129280, 1024),
            (2048, 129280, 1024),
        ]

    def get_input_iter(self, dtype):
        for num_rows, vocab_size, top_k in self.shapes:
            logits = torch.randn(
                num_rows, vocab_size, device=device, dtype=torch.float32
            )
            # Full vocab range: row_starts=0, row_ends=vocab_size (common case)
            row_starts = torch.zeros(num_rows, dtype=torch.int32, device=device)
            row_ends = torch.full(
                (num_rows,), vocab_size, dtype=torch.int32, device=device
            )
            # Pre-allocated output buffer (matches vLLM calling convention)
            indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=device)
            stride0 = logits.stride(0)  # = vocab_size for contiguous
            stride1 = logits.stride(1)  # = 1 for contiguous

            yield {
                "logits": logits,
                "row_starts": row_starts,
                "row_ends": row_ends,
                "indices": indices,
                "num_rows": num_rows,
                "stride0": stride0,
                "stride1": stride1,
                "top_k": top_k,
            }

    def get_gems_op(self):
        return top_k_per_row_prefill

    def get_torch_op(self):
        def torch_topk_ref(
            logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
        ):
            # Baseline: torch.topk without masking or index adjustment.
            # This under-counts the Triton kernel's work (which also masks + adjusts),
            # but provides a consistent PyTorch-native comparison point.
            _, top_idx = torch.topk(logits, top_k, dim=1, largest=True, sorted=False)
            indices.copy_(top_idx.to(torch.int32))

        return torch_topk_ref


@pytest.mark.top_k_per_row_prefill
def test_perf_top_k_per_row_prefill():
    bench = TopKPerRowPrefillBenchmark(
        op_name="top_k_per_row_prefill",
        torch_op=None,
        dtypes=[torch.float32],
    )
    bench.run()
