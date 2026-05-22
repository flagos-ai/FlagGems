"""Benchmark for top_k_per_row_decode (DeepSeek V4 decode-phase top-K).

Shapes match DeepSeek V4 production config (vocab=129280, top_k=1024).
The baseline uses vLLM's CUDA kernel when available,
falling back to a pure-PyTorch reference (torch.topk).
"""

import inspect

import pytest
import torch
import triton.language as tl

from flag_gems.fused import top_k_per_row_decode

from . import base


def _has_histogram_mask():
    if not hasattr(tl, "histogram"):
        return False
    try:
        return "mask" in inspect.signature(tl.histogram).parameters
    except (ValueError, TypeError):
        return False


pytestmark = pytest.mark.skipif(
    not _has_histogram_mask(),
    reason="tl.histogram with mask parameter not available",
)

# --- vLLM CUDA baseline (preferred) with PyTorch fallback ---
try:
    import vllm._custom_ops  # noqa: F401 — loads torch.ops._C

    def _vllm_top_k_per_row_decode(
        logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
    ):
        torch.ops._C.top_k_per_row_decode(
            logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
        )

    HAS_VLLM = True
except (ImportError, AttributeError):
    HAS_VLLM = False
    _vllm_top_k_per_row_decode = None


def _torch_topk_ref(
    logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
):
    """Pure-PyTorch fallback reference using torch.topk."""
    seq_len = seq_lens[0].item()
    valid_logits = logits[:, :seq_len]
    _, top_idx = torch.topk(valid_logits, top_k, dim=1, largest=True, sorted=False)
    indices.copy_(top_idx.to(torch.int32))


_baseline_op = _vllm_top_k_per_row_decode if HAS_VLLM else _torch_topk_ref


class TopKPerRowDecodeBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "vocab_size, top_k"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (129280, 1024),
            (32768, 512),
            (16384, 256),
            (8192, 128),
            (4096, 64),
        ]

    def get_input_iter(self, dtype):
        for vocab_size, top_k in self.shapes:
            torch.manual_seed(42)
            logits = torch.randn(
                (1, vocab_size), dtype=torch.float32, device=self.device
            )
            seq_lens = torch.tensor([vocab_size], dtype=torch.int32, device=self.device)
            indices = torch.zeros((1, top_k), dtype=torch.int32, device=self.device)
            num_rows = 1
            next_n = 1
            stride0 = logits.stride(0)
            stride1 = logits.stride(1)

            yield (
                logits,
                next_n,
                seq_lens,
                indices,
                num_rows,
                stride0,
                stride1,
                top_k,
            )


@pytest.mark.top_k_per_row_decode
def test_top_k_per_row_decode():
    bench = TopKPerRowDecodeBenchmark(
        op_name="top_k_per_row_decode",
        torch_op=_baseline_op,
        gems_op=top_k_per_row_decode,
        dtypes=[torch.float32],
    )
    bench.run()
