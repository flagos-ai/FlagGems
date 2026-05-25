"""Correctness tests for sparse_attn_indexer Triton kernel.

Tests verify the kernel against a pure-PyTorch reference implementation that:
1. Quantizes K vectors to FP8 E4M3 with per-token absmax scaling
2. Stores quantized keys + scale in a uint8 KV cache
3. Computes logits = sum_h(ReLU(q_h . k_pos) * weight_h) * k_scale
4. Selects top-K indices per token

Correctness is checked via set comparison (order-independent) since the
kernel does not guarantee sorted output.

Target shapes match DeepSeek V4:
    - num_heads=64, head_dim=128, topk=1024
"""

import pytest
import torch

import flag_gems
from flag_gems.fused.sparse_attn_indexer import sparse_attn_indexer
from tests.accuracy_utils import gems_assert_equal

device = flag_gems.device

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required",
)


# ---------------------------------------------------------------------------
# Pure-PyTorch reference implementation
# ---------------------------------------------------------------------------


def _quantize_k_to_cache_pytorch(k: torch.Tensor, head_dim: int):
    """Quantize K to FP8 E4M3 and store in uint8 cache (PyTorch reference)."""
    num_tokens = k.shape[0]
    cache_stride_slot = head_dim + 4
    kv_cache = torch.zeros(
        num_tokens, cache_stride_slot, dtype=torch.uint8, device=k.device
    )

    k_f32 = k.float()
    amax = k_f32.abs().amax(dim=1).clamp(min=1e-4)
    scale = amax / 448.0

    k_scaled = k_f32 / scale[:, None]
    k_fp8 = k_scaled.to(torch.float8_e4m3fn)
    k_uint8 = k_fp8.view(torch.uint8)
    kv_cache[:, :head_dim] = k_uint8

    scale_bytes = scale.to(torch.float32).view(torch.uint8).reshape(num_tokens, 4)
    kv_cache[:, head_dim : head_dim + 4] = scale_bytes

    return kv_cache


def _compute_logits_pytorch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    num_heads: int,
    head_dim: int,
    num_tokens: int,
):
    """Compute attention logits using PyTorch (reference)."""
    logits = torch.full(
        (num_tokens, num_tokens),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )

    for i in range(num_tokens):
        kv_end = i + 1
        k_uint8 = kv_cache[:kv_end, :head_dim]
        k_fp8 = k_uint8.view(torch.float8_e4m3fn)
        k_f16 = k_fp8.to(torch.float16)

        scale_bytes = kv_cache[:kv_end, head_dim : head_dim + 4]
        k_scale = scale_bytes.contiguous().view(torch.float32)

        qi = q[i].to(torch.float16)
        dots = torch.mm(qi, k_f16.T).float()
        dots = torch.relu(dots)

        w_i = weights[i]
        acc = (dots * w_i[:, None]).sum(dim=0)
        acc = acc * k_scale

        logits[i, :kv_end] = acc

    return logits


def sparse_attn_indexer_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    num_heads: int,
    head_dim: int,
    topk: int,
    num_tokens: int,
    insert_k: bool = True,
):
    """Pure-PyTorch reference for sparse_attn_indexer."""
    kv_cache = _quantize_k_to_cache_pytorch(k, head_dim)

    q_reshaped = q.reshape(num_tokens, num_heads, head_dim)

    logits = _compute_logits_pytorch(
        q_reshaped, kv_cache, weights, num_heads, head_dim, num_tokens
    )

    topk_indices = torch.full(
        (num_tokens, topk), -1, dtype=torch.int32, device=q.device
    )
    for i in range(num_tokens):
        kv_end = i + 1
        if kv_end <= topk:
            topk_indices[i, :kv_end] = torch.arange(
                kv_end, dtype=torch.int32, device=q.device
            )
        else:
            row_logits = logits[i, :kv_end]
            _, top_idx = torch.topk(row_logits, topk, largest=True)
            topk_indices[i] = top_idx.to(torch.int32)

    return topk_indices, kv_cache


# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------


def _make_inputs(num_tokens, num_heads, head_dim, device):
    """Create random input tensors matching the kernel signature."""
    q = torch.randn(
        num_tokens,
        num_heads * head_dim,
        dtype=torch.float16,
        device=device,
    )
    k = torch.randn(num_tokens, head_dim, dtype=torch.float16, device=device)
    weights = torch.rand(num_tokens, num_heads, dtype=torch.float32, device=device)
    return q, k, weights


def _compare_topk_sets(ref_indices, tri_indices, num_tokens, topk):
    """Compare top-K results as sets (order-independent)."""
    matches = 0
    for i in range(num_tokens):
        ref_set = set(ref_indices[i].cpu().tolist()) - {-1}
        tri_set = set(tri_indices[i].cpu().tolist()) - {-1}
        if ref_set == tri_set:
            matches += 1
    return matches, num_tokens


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@pytest.mark.sparse_attn_indexer
@pytest.mark.parametrize(
    "num_tokens,num_heads,head_dim,topk",
    [
        (1, 64, 128, 1024),
        (32, 64, 128, 1024),
        (128, 64, 128, 1024),
    ],
    ids=[
        "single_token",
        "batch_32",
        "batch_128",
    ],
)
def test_shortcut_path(num_tokens, num_heads, head_dim, topk):
    """Test tokens where kv_end <= topk (shortcut path fills trivially)."""
    q, k, weights = _make_inputs(num_tokens, num_heads, head_dim, device)

    assert num_tokens <= topk, "This test targets the shortcut path"

    cache_stride_slot = head_dim + 4
    max_model_len = 2048
    kv_cache = torch.zeros(
        max_model_len, cache_stride_slot, dtype=torch.uint8, device=device
    )

    result = sparse_attn_indexer(
        q,
        k,
        weights,
        kv_cache,
        num_heads=num_heads,
        head_dim=head_dim,
        topk=topk,
        num_tokens=num_tokens,
        total_kv_len=num_tokens,
        insert_k=True,
    )

    # Build expected: token i should have indices [0, 1, ..., i], rest -1
    expected = torch.full(
        (num_tokens, topk), -1, dtype=torch.int32, device=device
    )
    for i in range(num_tokens):
        expected[i, : i + 1] = torch.arange(i + 1, dtype=torch.int32, device=device)

    # Sort valid indices in each row for order-independent comparison
    for i in range(num_tokens):
        valid_len = i + 1
        result_row = result[i, :valid_len].sort().values
        expected_row = expected[i, :valid_len].sort().values
        gems_assert_equal(result_row, expected_row)


@pytest.mark.sparse_attn_indexer
@pytest.mark.parametrize(
    "num_tokens,num_heads,head_dim,topk,max_model_len",
    [
        (2048, 64, 128, 1024, 4096),
        (4096, 64, 128, 1024, 4096),
    ],
    ids=[
        "prefill_2048",
        "prefill_4096",
    ],
)
def test_topk_selection(num_tokens, num_heads, head_dim, topk, max_model_len):
    """Test top-K selection for tokens where kv_end > topk.

    Verifies that the Triton kernel selects the same set of top-K
    indices as the PyTorch reference (order-independent comparison).
    """
    q, k, weights = _make_inputs(num_tokens, num_heads, head_dim, device)

    cache_stride_slot = head_dim + 4
    kv_cache = torch.zeros(
        max_model_len, cache_stride_slot, dtype=torch.uint8, device=device
    )

    tri_result = sparse_attn_indexer(
        q,
        k,
        weights,
        kv_cache,
        num_heads=num_heads,
        head_dim=head_dim,
        topk=topk,
        num_tokens=num_tokens,
        total_kv_len=num_tokens,
        insert_k=True,
    )

    ref_result, _ = sparse_attn_indexer_ref(
        q, k, weights, num_heads, head_dim, topk, num_tokens
    )

    # Compare only tokens in the multi-kernel path (kv_end > topk)
    active_start = topk
    if active_start >= num_tokens:
        pytest.skip("No active tokens for multi-kernel path")

    matches, total = _compare_topk_sets(
        ref_result[active_start:],
        tri_result[active_start:],
        num_tokens - active_start,
        topk,
    )

    match_rate = matches / total
    assert match_rate >= 0.95, (
        f"Match rate {match_rate:.2%} below 95% threshold "
        f"({matches}/{total} rows match)"
    )


@pytest.mark.sparse_attn_indexer
@pytest.mark.parametrize(
    "num_tokens,num_heads,head_dim,topk",
    [
        (32, 64, 128, 1024),
        (128, 64, 128, 1024),
    ],
    ids=[
        "cache_32",
        "cache_128",
    ],
)
def test_k_cache_insert(num_tokens, num_heads, head_dim, topk):
    """Test that K vectors are correctly quantized and stored in cache."""
    q, k, weights = _make_inputs(num_tokens, num_heads, head_dim, device)

    cache_stride_slot = head_dim + 4
    max_model_len = 2048
    kv_cache = torch.zeros(
        max_model_len, cache_stride_slot, dtype=torch.uint8, device=device
    )

    sparse_attn_indexer(
        q,
        k,
        weights,
        kv_cache,
        num_heads=num_heads,
        head_dim=head_dim,
        topk=topk,
        num_tokens=num_tokens,
        total_kv_len=num_tokens,
        insert_k=True,
    )

    ref_cache = _quantize_k_to_cache_pytorch(k, head_dim)

    # Use gems_assert_equal for strict byte-level comparison
    gems_assert_equal(kv_cache[:num_tokens], ref_cache)


@pytest.mark.sparse_attn_indexer
@pytest.mark.parametrize(
    "num_tokens,num_heads,head_dim,topk",
    [
        (32, 64, 128, 1024),
    ],
    ids=["skip_insert"],
)
def test_skip_k_insert(num_tokens, num_heads, head_dim, topk):
    """Test that insert_k=False skips cache insertion."""
    q, k, weights = _make_inputs(num_tokens, num_heads, head_dim, device)

    cache_stride_slot = head_dim + 4
    max_model_len = 2048
    kv_cache = torch.zeros(
        max_model_len, cache_stride_slot, dtype=torch.uint8, device=device
    )

    # Pre-fill cache manually so logits can be computed
    ref_cache = _quantize_k_to_cache_pytorch(k, head_dim)
    kv_cache[:num_tokens] = ref_cache

    sentinel = kv_cache[:num_tokens].clone()

    result = sparse_attn_indexer(
        q,
        k,
        weights,
        kv_cache,
        num_heads=num_heads,
        head_dim=head_dim,
        topk=topk,
        num_tokens=num_tokens,
        total_kv_len=num_tokens,
        insert_k=False,
    )

    # Cache should be unchanged
    gems_assert_equal(kv_cache[:num_tokens], sentinel)

    # Result should still be valid (shortcut path for small num_tokens)
    for i in range(num_tokens):
        row = result[i].cpu().tolist()
        expected = set(range(i + 1))
        actual = set(row) - {-1}
        assert actual == expected


@pytest.mark.sparse_attn_indexer
@pytest.mark.parametrize(
    "num_tokens",
    [1, 16, 64],
    ids=["1_token", "16_tokens", "64_tokens"],
)
def test_output_shape_and_dtype(num_tokens):
    """Verify output tensor shape and dtype."""
    num_heads = 64
    head_dim = 128
    topk = 1024

    q, k, weights = _make_inputs(num_tokens, num_heads, head_dim, device)

    cache_stride_slot = head_dim + 4
    kv_cache = torch.zeros(2048, cache_stride_slot, dtype=torch.uint8, device=device)

    result = sparse_attn_indexer(
        q,
        k,
        weights,
        kv_cache,
        num_heads=num_heads,
        head_dim=head_dim,
        topk=topk,
        num_tokens=num_tokens,
        total_kv_len=num_tokens,
        insert_k=True,
    )

    assert result.shape == (num_tokens, topk)
    assert result.dtype == torch.int32
    assert result.device.type == "cuda"
