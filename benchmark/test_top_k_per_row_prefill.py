"""Benchmark for top_k_per_row_prefill (DeepSeek V4 sparse attention).

Shapes match DeepSeek V4 production config:
    - vocab_size=129280: DeepSeek V4 vocabulary size (full BPE vocab)
    - top_k=1024: number of KV cache slots selected per token in sparse attention
    - num_rows=1: single-token decode (latency-critical path)
    - num_rows=32: typical prefill micro-batch
    - num_rows=64: larger prefill batch
    - num_rows=2048: max prefill sequence length

The baseline uses vLLM's persistent_topk CUDA kernel as the reference implementation.
"""

import pytest
import torch

import flag_gems
from flag_gems.fused import top_k_per_row_prefill

from . import base

device = flag_gems.device

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required",
)


_VLLM_SO = "/data/yzw/vllm/vllm/_C.abi3.so"
_loaded = False


def _ensure_vllm_loaded():
    global _loaded
    if not _loaded:
        torch.ops.load_library(_VLLM_SO)
        _loaded = True


def _vllm_top_k_ref(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """vLLM persistent_topk CUDA kernel reference."""
    _ensure_vllm_loaded()
    torch.ops._C.top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        indices,
        num_rows,
        stride0,
        stride1,
        top_k,
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

            yield logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k


@pytest.mark.top_k_per_row_prefill
def test_top_k_per_row_prefill():
    bench = TopKPerRowPrefillBenchmark(
        op_name="top_k_per_row_prefill",
        torch_op=_vllm_top_k_ref,
        gems_op=top_k_per_row_prefill,
        dtypes=[torch.float32],
    )
    bench.run()
