"""Accuracy tests for persistent_topk (DeepSeek V4 sparse attention top-K).

Tests the Triton workspace-cached ternary search kernel against the vLLM CUDA
reference. Uses value-based comparison (sorted selected values must match) to
handle non-deterministic tie-breaking between implementations.
"""

import pytest
import torch

import flag_gems
from flag_gems.fused import persistent_topk

from . import conftest as cfg

device = flag_gems.device

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# --- vLLM CUDA reference (optional) ---
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

# --- Shape configuration with QUICK_MODE support ---
if cfg.QUICK_MODE:
    SHAPES = [(1, 4096, 1024)]
else:
    SHAPES = [
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


@pytest.mark.parametrize("shape", SHAPES, ids=[f"{r}x{s}_k{k}" for r, s, k in SHAPES])
def test_persistent_topk(shape):
    num_rows, seq_len, k = shape
    stride = seq_len

    logits = torch.randn(num_rows, stride, device=device, dtype=torch.float32)
    seq_lens = torch.full((num_rows,), seq_len, device=device, dtype=torch.int32)
    topk_indices_ref = torch.zeros(num_rows, k, device=device, dtype=torch.int32)
    topk_indices_res = torch.zeros(num_rows, k, device=device, dtype=torch.int32)
    workspace = torch.zeros(RADIX_TOPK_WORKSPACE_SIZE, device=device, dtype=torch.uint8)

    # Reference
    _baseline_op(logits, seq_lens, topk_indices_ref, workspace, k, stride)

    # Triton implementation
    persistent_topk(logits, seq_lens, topk_indices_res, workspace, k, stride)

    # Compare: top-k indices are unordered, so compare selected values
    for i in range(num_rows):
        ref_idx = topk_indices_ref[i].long()
        res_idx = topk_indices_res[i].long()

        ref_vals = torch.sort(logits[i][ref_idx], descending=True).values
        res_vals = torch.sort(logits[i][res_idx], descending=True).values

        torch.testing.assert_close(res_vals, ref_vals, atol=1e-6, rtol=1e-6)
