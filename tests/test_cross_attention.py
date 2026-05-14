import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Cross-attention test shapes (batch, head, seq, head_dim)
# Using simpler shapes that are known to work well with scaled_dot_product_attention
CROSS_ATTENTION_SHAPES = [
    (1, 4, 8, 32),
    (2, 4, 16, 32),
    (1, 8, 16, 64),
]


@pytest.mark.cross_attention
@pytest.mark.parametrize("shape", CROSS_ATTENTION_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_cross_attention(shape, dtype):
    """Test cross_attention accuracy against PyTorch's scaled_dot_product_attention"""
    batch, num_heads, q_seq_len, head_dim = shape
    kv_seq_len = q_seq_len  # For cross-attention, KV seq can differ but we test same for simplicity

    # Create Q, K, V tensors
    # Q: (batch, num_heads, q_seq_len, head_dim)
    # K, V: (batch, num_heads, kv_seq_len, head_dim)
    q = torch.randn(
        batch, num_heads, q_seq_len, head_dim, dtype=dtype, device=flag_gems.device
    )
    k = torch.randn(
        batch, num_heads, kv_seq_len, head_dim, dtype=dtype, device=flag_gems.device
    )
    v = torch.randn(
        batch, num_heads, kv_seq_len, head_dim, dtype=dtype, device=flag_gems.device
    )

    # Reference: use PyTorch's scaled_dot_product_attention
    ref_q = to_reference(q)
    ref_k = to_reference(k)
    ref_v = to_reference(v)

    ref_out = torch.nn.functional.scaled_dot_product_attention(
        ref_q, ref_k, ref_v, is_causal=False
    )

    # GEMS implementation
    with flag_gems.use_gems():
        res_out = flag_gems.cross_attention(q, k, v)

    # Compare outputs
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cross_attention
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_cross_attention_different_seq_len(dtype):
    """Test cross_attention with different Q and KV sequence lengths"""
    # Use smaller shapes for different seq len tests
    batch, num_heads, q_seq_len, head_dim = (1, 4, 8, 32)
    kv_seq_len = 16  # Different sequence length for KV

    # Create Q, K, V tensors with different sequence lengths
    q = torch.randn(
        batch, num_heads, q_seq_len, head_dim, dtype=dtype, device=flag_gems.device
    )
    k = torch.randn(
        batch, num_heads, kv_seq_len, head_dim, dtype=dtype, device=flag_gems.device
    )
    v = torch.randn(
        batch, num_heads, kv_seq_len, head_dim, dtype=dtype, device=flag_gems.device
    )

    # Reference: use PyTorch's scaled_dot_product_attention
    ref_q = to_reference(q)
    ref_k = to_reference(k)
    ref_v = to_reference(v)

    ref_out = torch.nn.functional.scaled_dot_product_attention(
        ref_q, ref_k, ref_v, is_causal=False
    )

    # GEMS implementation
    with flag_gems.use_gems():
        res_out = flag_gems.cross_attention(q, k, v)

    # Compare outputs
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cross_attention
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_cross_attention_with_scale(dtype):
    """Test cross_attention with custom scale factor"""
    shape = (2, 4, 16, 32)
    batch, num_heads, q_seq_len, head_dim = shape

    q = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    k = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    v = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    custom_scale = 0.5

    # Reference
    ref_q = to_reference(q)
    ref_k = to_reference(k)
    ref_v = to_reference(v)

    ref_out = torch.nn.functional.scaled_dot_product_attention(
        ref_q, ref_k, ref_v, scale=custom_scale, is_causal=False
    )

    # GEMS implementation
    with flag_gems.use_gems():
        res_out = flag_gems.cross_attention(q, k, v, scale=custom_scale)

    # Compare outputs
    gems_assert_close(res_out, ref_out, dtype)
