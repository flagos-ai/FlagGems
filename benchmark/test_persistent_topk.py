"""Benchmark for persistent_topk (DeepSeek V4 sparse attention top-K).

Shapes simulate DeepSeek V4 sparse attention inference patterns:
    - (1, 1024-4096, k=1024): decode path, single token
    - (4-32, 4096-32768, k=1024): batched decode / prefill
    - k=512/2048 variants: alternative top-k sizes

The baseline uses vLLM's CUDA persistent_topk when available,
falling back to torch.topk.
"""

import pytest
import torch

import flag_gems
from flag_gems.fused import persistent_topk

from . import base

device = flag_gems.device

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required",
)

# --- vLLM CUDA baseline (preferred) with PyTorch fallback ---
try:
    import vllm._custom_ops  # noqa: F401 — loads torch.ops._C

    def _vllm_persistent_topk(logits, seq_lens, topk_indices, workspace, k, stride):
        torch.ops._C.persistent_topk(
            logits, seq_lens, topk_indices, workspace, k, stride
        )

    HAS_VLLM = True
except (ImportError, AttributeError):
    HAS_VLLM = False
    _vllm_persistent_topk = None


def _torch_topk_ref(logits, seq_lens, topk_indices, workspace, k, stride):
    """Pure-PyTorch fallback reference using torch.topk."""
    num_rows = logits.shape[0]
    for i in range(num_rows):
        seq_len = seq_lens[i].item()
        valid_logits = logits[i, :seq_len]
        _, top_idx = torch.topk(valid_logits, k, largest=True, sorted=False)
        topk_indices[i].copy_(top_idx.to(torch.int32))


_baseline_op = _vllm_persistent_topk if HAS_VLLM else _torch_topk_ref


class PersistentTopkBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "num_rows, seq_len, top_k"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 1024, 1024),
            (1, 2048, 1024),
            (1, 4096, 1024),
            (4, 4096, 1024),
            (8, 8192, 1024),
            (16, 16384, 1024),
            (32, 32768, 1024),
            (4, 4096, 512),
            (4, 4096, 2048),
        ]

    def get_input_iter(self, dtype):
        for num_rows, seq_len, k in self.shapes:
            stride = seq_len
            logits = torch.randn(num_rows, stride, device=device, dtype=torch.float32)
            seq_lens = torch.full(
                (num_rows,), seq_len, device=device, dtype=torch.int32
            )
            topk_indices = torch.zeros(num_rows, k, device=device, dtype=torch.int32)
            workspace = torch.zeros(
                RADIX_TOPK_WORKSPACE_SIZE, device=device, dtype=torch.uint8
            )

            yield logits, seq_lens, topk_indices, workspace, k, stride


@pytest.mark.persistent_topk
def test_persistent_topk():
    bench = PersistentTopkBenchmark(
        op_name="persistent_topk",
        torch_op=_baseline_op,
        gems_op=persistent_topk,
        dtypes=[torch.float32],
    )
    bench.run()
