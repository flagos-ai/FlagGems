"""Accuracy tests for top_k_per_row_decode (DeepSeek V4 decode-phase top-K).

Tests the Triton radix-select kernel against the vLLM CUDA reference.
Uses value-based comparison (sorted selected values must match) to handle
non-deterministic tie-breaking between implementations.
"""

import inspect

import pytest
import torch
import triton.language as tl

import flag_gems
from flag_gems.fused import top_k_per_row_decode

from . import conftest as cfg

device = flag_gems.device


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

# --- Shape configuration with QUICK_MODE support ---
if cfg.QUICK_MODE:
    VOCAB_SIZE_LIST = [129280]
    TOP_K_LIST = [1024]
else:
    VOCAB_SIZE_LIST = [4096, 8192, 16384, 32768, 129280]
    TOP_K_LIST = [64, 128, 256, 512, 1024]

# --- vLLM CUDA reference (optional) ---
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


def _selected_values(logits, indices):
    """Gather values at selected indices, sort for order-independent comparison."""
    return logits.gather(1, indices.long()).sort(dim=1).values


def _make_inputs(vocab_size, top_k, seq_len=None):
    """Generate test inputs matching DeepSeek V4 decode config."""
    if seq_len is None:
        seq_len = vocab_size
    logits = torch.randn(1, vocab_size, dtype=torch.float32, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    indices = torch.zeros(1, top_k, dtype=torch.int32, device=device)
    num_rows = 1
    next_n = 1
    stride0 = logits.stride(0)
    stride1 = logits.stride(1)
    return logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k


def _torch_topk_ref(
    logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
):
    """Pure-PyTorch fallback reference using torch.topk."""
    seq_len = seq_lens[0].item()
    valid_logits = logits[:, :seq_len]
    _, top_idx = torch.topk(valid_logits, top_k, dim=1, largest=True, sorted=False)
    indices.copy_(top_idx.to(torch.int32))


@pytest.mark.top_k_per_row_decode
@pytest.mark.parametrize(
    "vocab_size, top_k",
    [(v, k) for v, k in zip(VOCAB_SIZE_LIST, TOP_K_LIST)],
    ids=[f"V{v}_K{k}" for v, k in zip(VOCAB_SIZE_LIST, TOP_K_LIST)],
)
def test_top_k_per_row_decode(vocab_size, top_k):
    """Test top-k correctness: selected values must match reference."""
    torch.manual_seed(42)
    ref_fn = _vllm_top_k_per_row_decode if HAS_VLLM else _torch_topk_ref

    logits, next_n, seq_lens, indices, num_rows, s0, s1, k = _make_inputs(
        vocab_size, top_k
    )
    logits_ref = logits.clone()
    indices_ref = torch.zeros_like(indices)

    top_k_per_row_decode(logits, next_n, seq_lens, indices, num_rows, s0, s1, k)
    ref_fn(logits_ref, next_n, seq_lens, indices_ref, num_rows, s0, s1, k)

    vals_tri = _selected_values(logits, indices)
    vals_ref = _selected_values(logits_ref, indices_ref)
    torch.testing.assert_close(vals_tri, vals_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.top_k_per_row_decode
@pytest.mark.parametrize(
    "vocab_size, top_k, seq_len",
    [
        (129280, 1024, 100000),
        (32768, 256, 16384),
        (8192, 64, 4096),
    ],
    ids=["V129280_K1024_S100000", "V32768_K256_S16384", "V8192_K64_S4096"],
)
def test_top_k_per_row_decode_partial_seqlen(vocab_size, top_k, seq_len):
    """Test with seq_len < vocab_size (partial valid range)."""
    torch.manual_seed(123)
    ref_fn = _vllm_top_k_per_row_decode if HAS_VLLM else _torch_topk_ref

    logits, next_n, seq_lens, indices, num_rows, s0, s1, k = _make_inputs(
        vocab_size, top_k, seq_len=seq_len
    )
    logits_ref = logits.clone()
    indices_ref = torch.zeros_like(indices)

    top_k_per_row_decode(logits, next_n, seq_lens, indices, num_rows, s0, s1, k)
    ref_fn(logits_ref, next_n, seq_lens, indices_ref, num_rows, s0, s1, k)

    vals_tri = _selected_values(logits, indices)
    vals_ref = _selected_values(logits_ref, indices_ref)
    torch.testing.assert_close(vals_tri, vals_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.top_k_per_row_decode
def test_top_k_per_row_decode_indices_in_range():
    """Verify all selected indices are within [0, seq_len)."""
    torch.manual_seed(7)
    vocab_size, top_k, seq_len = 129280, 1024, 100000
    logits, next_n, seq_lens, indices, num_rows, s0, s1, k = _make_inputs(
        vocab_size, top_k, seq_len=seq_len
    )
    top_k_per_row_decode(logits, next_n, seq_lens, indices, num_rows, s0, s1, k)
    assert indices.min().item() >= 0
    assert indices.max().item() < seq_len


@pytest.mark.top_k_per_row_decode
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is not installed")
@pytest.mark.parametrize(
    "vocab_size, top_k",
    [(129280, 1024), (32768, 512), (4096, 64)],
    ids=["V129280_K1024", "V32768_K512", "V4096_K64"],
)
def test_top_k_per_row_decode_vs_vllm(vocab_size, top_k):
    """Test against vLLM CUDA kernel."""
    torch.manual_seed(42)
    logits, next_n, seq_lens, indices, num_rows, s0, s1, k = _make_inputs(
        vocab_size, top_k
    )
    logits_ref = logits.clone()
    indices_ref = torch.zeros_like(indices)

    top_k_per_row_decode(logits, next_n, seq_lens, indices, num_rows, s0, s1, k)
    _vllm_top_k_per_row_decode(
        logits_ref, next_n, seq_lens, indices_ref, num_rows, s0, s1, k
    )

    vals_tri = _selected_values(logits, indices)
    vals_ref = _selected_values(logits_ref, indices_ref)
    torch.testing.assert_close(vals_tri, vals_ref, rtol=1e-5, atol=1e-5)
