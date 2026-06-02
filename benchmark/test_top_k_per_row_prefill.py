"""Benchmark for top_k_per_row_prefill (DeepSeek V4 sparse attention).

Shapes match DeepSeek V4 production config:
    - vocab_size=129280: DeepSeek V4 vocabulary size (full BPE vocab)
    - top_k=1024: number of KV cache slots selected per token in sparse attention
    - num_rows=1: single-token decode (latency-critical path)
    - num_rows=32: typical prefill micro-batch
    - num_rows=64: larger prefill batch
    - num_rows=2048: max prefill sequence length

The baseline uses vLLM's persistent_topk CUDA kernel when available,
falling back to torch.topk otherwise.
"""

import os

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


_VLLM_SO = os.environ.get("VLLM_SO_PATH", "/data/yzw/vllm/vllm/_C.abi3.so")
_vllm_available = False


def _try_load_vllm():
    global _vllm_available
    if _vllm_available:
        return True
    try:
        if os.path.exists(_VLLM_SO):
            torch.ops.load_library(_VLLM_SO)
            _vllm_available = True
    except Exception:
        pass
    return _vllm_available


def _torch_topk_ref(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """torch.topk baseline (fallback when vLLM is not available)."""
    _, top_idx = torch.topk(logits, top_k, dim=1, largest=True, sorted=False)
    indices.copy_(top_idx.to(torch.int32))


def _vllm_top_k_ref(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """vLLM persistent_topk CUDA kernel reference."""
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


def _get_ref_op():
    if _try_load_vllm():
        return _vllm_top_k_ref
    return _torch_topk_ref


class TopKPerRowPrefillBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "num_rows, vocab_size, top_k"

    def set_shapes(self, shape_file_path=None):
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
            row_starts = torch.zeros(num_rows, dtype=torch.int32, device=device)
            row_ends = torch.full(
                (num_rows,), vocab_size, dtype=torch.int32, device=device
            )
            indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=device)
            stride0 = logits.stride(0)
            stride1 = logits.stride(1)

            yield logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k


@pytest.mark.top_k_per_row_prefill
def test_top_k_per_row_prefill():
    bench = TopKPerRowPrefillBenchmark(
        op_name="top_k_per_row_prefill",
        torch_op=_get_ref_op(),
        gems_op=top_k_per_row_prefill,
        dtypes=[torch.float32],
    )
    bench.run()
