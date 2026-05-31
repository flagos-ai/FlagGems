"""Accuracy tests for top_k_per_row_prefill (DeepSeek V4 sparse attention).

Tests the Triton kernel against vLLM CUDA reference (when available) and a
pure-PyTorch fallback. Verifies that the selected top-K values match
(set comparison, order-independent).

Test shapes match DeepSeek V4 production config:
    - vocab_size=129280: DeepSeek V4 vocabulary size
    - top_k=1024: number of KV cache slots selected per token
    - num_rows=1: single-token decode
    - num_rows=32/64/2048: prefill batch sizes
"""

import pytest
import torch

import flag_gems
from flag_gems.fused import top_k_per_row_prefill

from . import conftest as cfg

device = flag_gems.device

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required",
)

# --- Shape configuration with QUICK_MODE support ---
if cfg.QUICK_MODE:
    NUM_ROWS_LIST = [1, 32]
    VOCAB_SIZE_LIST = [129280]
    TOP_K_LIST = [1024]
else:
    NUM_ROWS_LIST = [1, 32, 64, 2048]
    VOCAB_SIZE_LIST = [20000, 129280]
    TOP_K_LIST = [1024, 2048]

# --- vLLM CUDA reference (optional) ---
try:
    import vllm._custom_ops  # noqa: F401 — loads torch.ops._C

    def _vllm_top_k_per_row_prefill(
        logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
    ):
        torch.ops._C.top_k_per_row_prefill(
            logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
        )

    HAS_VLLM = True
except (ImportError, AttributeError):
    HAS_VLLM = False
    _vllm_top_k_per_row_prefill = None


def reference_top_k_per_row(logits, row_starts, row_ends, top_k):
    """Pure-PyTorch reference: per-row torch.topk on valid range [start, end).

    Returns indices that are 0-based relative to row_starts[i], matching
    the Triton kernel's output convention.
    """
    num_rows, vocab_size = logits.shape
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=logits.device)

    for i in range(num_rows):
        start = row_starts[i].item()
        end = row_ends[i].item()
        row_slice = logits[i, start:end]
        k = min(top_k, end - start)
        _, topk_idx = torch.topk(row_slice, k, largest=True, sorted=False)
        indices[i, :k] = topk_idx.to(torch.int32)
        if k < top_k:
            indices[i, k:] = -1

    return indices


def check_topk_values_match(logits, indices_test, indices_ref, row_starts, top_k):
    """Value-based set comparison: verify that the actual logit values selected
    by test indices match those selected by reference indices.

    Order-independent — we only care that the same top-K values are found,
    not that they appear in the same order (argsort and topk break ties differently).
    """
    num_rows = logits.shape[0]
    for i in range(num_rows):
        offset = row_starts[i].item()
        abs_test = indices_test[i].long() + offset
        abs_ref = indices_ref[i].long() + offset

        valid_test = abs_test[abs_test >= offset]
        valid_ref = abs_ref[abs_ref >= offset]

        vals_test = logits[i].gather(0, valid_test)
        vals_ref = logits[i].gather(0, valid_ref)

        vals_test_sorted, _ = vals_test.sort(descending=True)
        vals_ref_sorted, _ = vals_ref.sort(descending=True)

        if not torch.allclose(vals_test_sorted, vals_ref_sorted, atol=1e-6, rtol=1e-6):
            return False
    return True


@pytest.mark.top_k_per_row_prefill
@pytest.mark.parametrize("num_rows", NUM_ROWS_LIST)
@pytest.mark.parametrize("vocab_size", VOCAB_SIZE_LIST)
@pytest.mark.parametrize("top_k", TOP_K_LIST)
def test_top_k_per_row_prefill_full_vocab(num_rows, vocab_size, top_k):
    """Test with full vocab range (row_starts=0, row_ends=vocab_size).

    This is the most common case in inference: every token sees the full vocabulary.
    The masking kernel should early-exit for all rows.
    """
    if top_k > vocab_size:
        return

    torch.manual_seed(42)

    logits = torch.randn(num_rows, vocab_size, device=device, dtype=torch.float32)
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device=device)
    row_ends = torch.full((num_rows,), vocab_size, dtype=torch.int32, device=device)
    stride0 = logits.stride(0)
    stride1 = logits.stride(1)

    indices_ref = reference_top_k_per_row(logits.clone(), row_starts, row_ends, top_k)

    indices_test = torch.empty((num_rows, top_k), dtype=torch.int32, device=device)
    top_k_per_row_prefill(
        logits, row_starts, row_ends, indices_test, num_rows, stride0, stride1, top_k
    )

    assert check_topk_values_match(
        logits, indices_test, indices_ref, row_starts, top_k
    ), f"FAIL: num_rows={num_rows}, vocab_size={vocab_size}, top_k={top_k}"


@pytest.mark.top_k_per_row_prefill
@pytest.mark.parametrize("num_rows", NUM_ROWS_LIST[:2])
@pytest.mark.parametrize("vocab_size", VOCAB_SIZE_LIST)
@pytest.mark.parametrize("top_k", TOP_K_LIST)
def test_top_k_per_row_prefill_variable_lengths(num_rows, vocab_size, top_k):
    """Test with variable row lengths (partial vocab per row).

    Simulates the case where different tokens in a batch have different valid
    KV ranges (e.g., due to causal masking or sequence packing).
    row_ends is randomized in [top_k, vocab_size] to ensure enough valid elements.
    """
    if top_k > vocab_size:
        return

    torch.manual_seed(123)

    logits = torch.randn(num_rows, vocab_size, device=device, dtype=torch.float32)
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device=device)
    row_ends = torch.randint(
        top_k, vocab_size + 1, (num_rows,), dtype=torch.int32, device=device
    )
    stride0 = logits.stride(0)
    stride1 = logits.stride(1)

    indices_ref = reference_top_k_per_row(logits.clone(), row_starts, row_ends, top_k)

    indices_test = torch.empty((num_rows, top_k), dtype=torch.int32, device=device)
    top_k_per_row_prefill(
        logits, row_starts, row_ends, indices_test, num_rows, stride0, stride1, top_k
    )

    assert check_topk_values_match(
        logits, indices_test, indices_ref, row_starts, top_k
    ), f"FAIL: num_rows={num_rows}, vocab_size={vocab_size}, top_k={top_k}"


@pytest.mark.top_k_per_row_prefill
@pytest.mark.parametrize("num_rows", [1, 16])
def test_top_k_per_row_prefill_nonzero_starts(num_rows):
    """Test with non-zero row_starts.

    Verifies the index subtraction logic: output indices must be 0-based relative
    to row_starts[i], not absolute vocab positions.
    """
    torch.manual_seed(456)
    vocab_size = 50000
    top_k = 1024

    logits = torch.randn(num_rows, vocab_size, device=device, dtype=torch.float32)
    row_starts = torch.randint(0, 1000, (num_rows,), dtype=torch.int32, device=device)
    row_ends = torch.randint(
        top_k + 1000, vocab_size + 1, (num_rows,), dtype=torch.int32, device=device
    )
    stride0 = logits.stride(0)
    stride1 = logits.stride(1)

    indices_ref = reference_top_k_per_row(logits.clone(), row_starts, row_ends, top_k)

    indices_test = torch.empty((num_rows, top_k), dtype=torch.int32, device=device)
    top_k_per_row_prefill(
        logits, row_starts, row_ends, indices_test, num_rows, stride0, stride1, top_k
    )

    assert check_topk_values_match(
        logits, indices_test, indices_ref, row_starts, top_k
    ), f"FAIL: num_rows={num_rows}, nonzero starts"


@pytest.mark.top_k_per_row_prefill
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is not installed")
@pytest.mark.parametrize("num_rows", [1, 32, 64])
def test_top_k_per_row_prefill_vs_vllm(num_rows):
    """Test against vLLM CUDA kernel (persistent_topk)."""
    torch.manual_seed(789)
    vocab_size = 129280
    top_k = 1024

    logits = torch.randn(num_rows, vocab_size, device=device, dtype=torch.float32)
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device=device)
    row_ends = torch.full((num_rows,), vocab_size, dtype=torch.int32, device=device)
    stride0 = logits.stride(0)
    stride1 = logits.stride(1)

    # vLLM CUDA reference
    logits_vllm = logits.clone()
    indices_vllm = torch.empty((num_rows, top_k), dtype=torch.int32, device=device)
    _vllm_top_k_per_row_prefill(
        logits_vllm,
        row_starts,
        row_ends,
        indices_vllm,
        num_rows,
        stride0,
        stride1,
        top_k,
    )

    # FlagGems Triton kernel
    indices_test = torch.empty((num_rows, top_k), dtype=torch.int32, device=device)
    top_k_per_row_prefill(
        logits, row_starts, row_ends, indices_test, num_rows, stride0, stride1, top_k
    )

    assert check_topk_values_match(
        logits, indices_test, indices_vllm, row_starts, top_k
    ), f"FAIL vs vLLM: num_rows={num_rows}"
