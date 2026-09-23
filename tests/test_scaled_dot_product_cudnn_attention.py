import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.scaled_dot_product_cudnn_attention
@pytest.mark.skipif(
    flag_gems.device != "cuda",
    reason="_scaled_dot_product_cudnn_attention is a CUDA-only operator (cuDNN)",
)
@pytest.mark.parametrize(
    "batch, num_head, q_seq_len, kv_seq_len, head_size",
    [
        (1, 4, 64, 64, 64),
        (2, 8, 128, 128, 64),
        (4, 4, 256, 256, 128),
        (1, 8, 512, 512, 64),
    ],
)
# cuDNN attention only supports fp16/bf16
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [False, True])
def test_scaled_dot_product_cudnn_attention(
    batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, is_causal
):
    """Test _scaled_dot_product_cudnn_attention accuracy against reference implementation."""
    utils.init_seed(42)

    # Create input tensors in the format expected by scaled_dot_product_cudnn_attention
    # Shape: (batch, num_heads, seq_len, head_dim)
    q = torch.randn(
        batch, num_head, q_seq_len, head_size, dtype=dtype, device=flag_gems.device
    )
    k = torch.randn(
        batch, num_head, kv_seq_len, head_size, dtype=dtype, device=flag_gems.device
    )
    v = torch.randn(
        batch, num_head, kv_seq_len, head_size, dtype=dtype, device=flag_gems.device
    )

    # Compute scale
    scale = 1.0 / (head_size**0.5)

    # Reference using torch.nn.functional.scaled_dot_product_attention
    ref_q = utils.to_reference(q)
    ref_k = utils.to_reference(k)
    ref_v = utils.to_reference(v)

    # Use PyTorch's SDPA as reference
    ref_out = torch.nn.functional.scaled_dot_product_attention(
        ref_q,
        ref_k,
        ref_v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
    )

    # Test FlagGems implementation
    with flag_gems.use_gems():
        (
            gems_out,
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            philox_seed,
            philox_offset,
            debug_attn_mask,
        ) = flag_gems._scaled_dot_product_cudnn_attention(
            q,
            k,
            v,
            attn_bias=None,
            compute_log_sumexp=True,
            dropout_p=0.0,
            is_causal=is_causal,
            return_debug_mask=False,
            scale=scale,
        )

    # Use relaxed tolerance for attention (atol=5e-2 for attention operations)
    utils.gems_assert_close(gems_out, ref_out, dtype, atol=5e-2)

    # Verify returned tensor shapes
    assert gems_out.shape == q.shape
    assert logsumexp.shape == (batch, num_head, q_seq_len)
    assert max_q == q_seq_len
    assert max_k == kv_seq_len


@pytest.mark.scaled_dot_product_cudnn_attention
@pytest.mark.skipif(
    flag_gems.device != "cuda",
    reason="_scaled_dot_product_cudnn_attention is a CUDA-only operator (cuDNN)",
)
@pytest.mark.parametrize(
    "batch, num_head, q_seq_len, kv_seq_len, head_size",
    [
        (2, 4, 64, 64, 64),
    ],
)
# attn_bias path validated with fp16 (cuDNN attention supports fp16/bf16)
@pytest.mark.parametrize("dtype", [torch.float16])
def test_scaled_dot_product_cudnn_attention_with_attn_bias(
    batch, num_head, q_seq_len, kv_seq_len, head_size, dtype
):
    """Test _scaled_dot_product_cudnn_attention with attention bias."""
    utils.init_seed(42)

    q = torch.randn(
        batch, num_head, q_seq_len, head_size, dtype=dtype, device=flag_gems.device
    )
    k = torch.randn(
        batch, num_head, kv_seq_len, head_size, dtype=dtype, device=flag_gems.device
    )
    v = torch.randn(
        batch, num_head, kv_seq_len, head_size, dtype=dtype, device=flag_gems.device
    )

    # Create attention bias
    attn_bias = (
        torch.randn(
            batch, num_head, q_seq_len, kv_seq_len, dtype=dtype, device=flag_gems.device
        )
        * 0.1
    )

    scale = 1.0 / (head_size**0.5)

    ref_q = utils.to_reference(q)
    ref_k = utils.to_reference(k)
    ref_v = utils.to_reference(v)
    ref_bias = utils.to_reference(attn_bias)

    # Reference using SDPA with attn_mask as bias
    ref_out = torch.nn.functional.scaled_dot_product_attention(
        ref_q,
        ref_k,
        ref_v,
        attn_mask=ref_bias,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )

    with flag_gems.use_gems():
        (
            gems_out,
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            philox_seed,
            philox_offset,
            debug_attn_mask,
        ) = flag_gems._scaled_dot_product_cudnn_attention(
            q,
            k,
            v,
            attn_bias=attn_bias,
            compute_log_sumexp=True,
            dropout_p=0.0,
            is_causal=False,
            return_debug_mask=False,
            scale=scale,
        )

    # Use relaxed tolerance for attention (atol=5e-2 for attention operations)
    utils.gems_assert_close(gems_out, ref_out, dtype, atol=5e-2)
