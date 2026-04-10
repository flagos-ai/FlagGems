"""
Transformer Layer End-to-End Combination Tests.

This module tests complete Transformer layers with FlagGems operators,
verifying numerical stability, gradient correctness, and performance.
"""

import math

import pytest
import torch
import torch.nn as nn

import flag_gems

from .models.transformer_block import TransformerBlock, LLaMABlock
from .accuracy_utils import compute_reference, combo_assert_close
from .utils.numerical_stability import (
    check_finite,
    check_gradient_health,
    check_no_nan,
)

device = flag_gems.device


class TestTransformerLayerE2E:
    """
    End-to-end tests for complete Transformer layers.

    These tests verify:
    - Basic forward pass correctness
    - Gradient computation
    - Numerical stability with various inputs
    - Different model architectures (standard, LLaMA)
    """

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "batch_size,seq_len,d_model",
        [
            (1, 128, 768),  # Small batch
            (4, 256, 768),  # Medium
            (16, 512, 768),  # Large batch
            (2, 2048, 4096),  # Long sequence (LLaMA-style)
        ],
    )
    def test_transformer_block_forward(
        self, batch_size, seq_len, d_model, use_gems
    ):
        """Test standard Transformer block forward pass."""
        nhead = 12
        dim_feedforward = d_model * 4

        # Create model
        model = TransformerBlock(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,  # Disable dropout for deterministic testing
            activation="gelu",
        )
        model = model.to(device).to(torch.float32)

        # Create input
        x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x)

        # Verify output
        assert output.shape == (batch_size, seq_len, d_model), "Output shape mismatch"
        check_no_nan(output, "Transformer forward")
        check_finite(output, "Transformer forward")

        # Reference comparison (TransformerBlock ≈ 10 ops: attn + norm + ffn + norm)
        ref_output = compute_reference(model, x)
        combo_assert_close(output, ref_output, torch.float32, num_ops=10, name="Transformer forward")

    @pytest.mark.integration
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_transformer_block_mixed_precision(self, dtype, use_gems):
        """Test Transformer block with mixed precision."""
        batch_size, seq_len, d_model = 4, 256, 768
        nhead = 12
        dim_feedforward = d_model * 4

        # Create model in fp32, cast to target dtype
        model = TransformerBlock(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="gelu",
        )
        model = model.to(device).to(dtype)

        # Create input
        x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x)

        # Verify
        assert output.shape == (batch_size, seq_len, d_model)
        check_no_nan(output, f"Transformer {dtype} forward")

        # Reference comparison
        ref_output = compute_reference(model, x)
        combo_assert_close(output, ref_output, dtype, num_ops=10, name=f"Transformer {dtype}")

    @pytest.mark.integration
    @pytest.mark.parametrize("d_model,nhead", [(768, 12), (1024, 16), (4096, 32)])
    def test_llama_block_forward(self, d_model, nhead, use_gems):
        """Test LLaMA-style block."""
        batch_size, seq_len = 2, 512
        dim_feedforward = int(d_model * 8 / 3)  # LLaMA uses 2.67x ratio

        # Create model
        model = LLaMABlock(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
        )
        model = model.to(device).to(torch.float32)

        # Create input
        x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)

        # Forward pass with causal masking
        model.eval()
        with torch.no_grad():
            output = model(x, is_causal=True)

        # Verify
        assert output.shape == (batch_size, seq_len, d_model)
        check_no_nan(output, "LLaMA block forward")
        check_finite(output, "LLaMA block forward")

        # Reference comparison (LLaMA block ≈ 12 ops: RMSNorm + attn + RMSNorm + SwiGLU FFN)
        ref_output = compute_reference(model, x, is_causal=True)
        combo_assert_close(output, ref_output, torch.float32, num_ops=12, name="LLaMA block")

    @pytest.mark.integration
    def test_transformer_backward(self, use_gems):
        """Test gradient computation through Transformer block."""
        batch_size, seq_len, d_model = 2, 128, 768
        nhead = 12

        # Create model
        model = TransformerBlock(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation="gelu",
        )
        model = model.to(device).to(torch.float32)

        # Create input with gradient
        x = torch.randn(
            batch_size, seq_len, d_model, device=device, dtype=torch.float32, requires_grad=True
        )

        # Forward pass
        output = model(x)

        # Create target
        target = torch.randn_like(output)

        # Compute loss
        loss = nn.functional.mse_loss(output, target)

        # Backward pass
        loss.backward()

        # Check gradients
        assert x.grad is not None, "Input gradient is None"
        check_gradient_health(model, "Transformer backward")

        # Verify gradient shapes
        assert x.grad.shape == x.shape, "Gradient shape mismatch"

    @pytest.mark.integration
    @pytest.mark.parametrize("is_causal", [True, False])
    def test_transformer_with_mask(self, is_causal, use_gems):
        """Test Transformer with different masking strategies."""
        batch_size, seq_len, d_model = 2, 256, 512
        nhead = 8

        model = TransformerBlock(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0,
        )
        model = model.to(device).to(torch.float32)

        x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x, is_causal=is_causal)

        # Verify
        assert output.shape == (batch_size, seq_len, d_model)
        check_no_nan(output, f"Transformer with is_causal={is_causal}")

        # Reference comparison
        ref_output = compute_reference(model, x, is_causal=is_causal)
        combo_assert_close(
            output, ref_output, torch.float32, num_ops=10,
            name=f"Transformer is_causal={is_causal}",
        )

    @pytest.mark.stress
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for stress test")
    def test_transformer_long_sequence(self, use_gems):
        """Test Transformer with very long sequences."""
        batch_size, seq_len, d_model = 1, 8192, 4096
        nhead = 32

        model = LLaMABlock(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(d_model * 8 / 3),
            dropout=0.0,
        )
        model = model.to(device).to(torch.float16)

        x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)

        # Forward pass
        output = model(x, is_causal=True)

        # Verify
        assert output.shape == (batch_size, seq_len, d_model)
        check_no_nan(output, "Long sequence Transformer")


class TestAttentionBlock:
    """
    Tests specifically for attention blocks within Transformer.
    """

    @pytest.mark.integration
    def test_multihead_attention_basic(self, use_gems):
        """Test basic multi-head attention."""
        from .models.attention import MultiHeadAttention

        batch_size, seq_len, d_model = 4, 256, 512
        nhead = 8

        attn = MultiHeadAttention(d_model, nhead, dropout=0.0)
        attn = attn.to(device).to(torch.float32)

        x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)

        attn.eval()
        with torch.no_grad():
            output = attn(x)

        assert output.shape == (batch_size, seq_len, d_model)
        check_no_nan(output, "MultiHeadAttention")

        # Reference comparison (attention ≈ 6 ops: QKV proj + matmul + softmax + matmul + out proj)
        ref_output = compute_reference(attn, x)
        combo_assert_close(output, ref_output, torch.float32, num_ops=6, name="MultiHeadAttention")

    @pytest.mark.integration
    @pytest.mark.parametrize("nhead", [4, 8, 12, 16, 32])
    def test_attention_different_heads(self, nhead, use_gems):
        """Test attention with different number of heads."""
        batch_size, seq_len, d_model = 2, 128, 768

        # Ensure d_model is divisible by nhead
        if d_model % nhead != 0:
            d_model = nhead * 64

        from .models.attention import MultiHeadAttention

        attn = MultiHeadAttention(d_model, nhead, dropout=0.0)
        attn = attn.to(device).to(torch.float32)

        x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)

        attn.eval()
        with torch.no_grad():
            output = attn(x)

        assert output.shape == (batch_size, seq_len, d_model)
        check_finite(output, f"Attention with {nhead} heads")

        # Reference comparison
        ref_output = compute_reference(attn, x)
        combo_assert_close(
            output, ref_output, torch.float32, num_ops=6,
            name=f"Attention {nhead} heads",
        )


class TestFFNBlock:
    """
    Tests specifically for FFN blocks within Transformer.
    """

    @pytest.mark.integration
    @pytest.mark.parametrize("activation", ["gelu", "silu"])
    def test_ffn_activations(self, activation, use_gems):
        """Test FFN with different activation functions."""
        batch_size, seq_len, d_model = 4, 256, 768
        dim_feedforward = d_model * 4

        from .models.ffn import StandardFFN

        ffn = StandardFFN(d_model, dim_feedforward, activation=activation, dropout=0.0)
        ffn = ffn.to(device).to(torch.float32)

        x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)

        ffn.eval()
        with torch.no_grad():
            output = ffn(x)

        assert output.shape == (batch_size, seq_len, d_model)
        check_no_nan(output, f"FFN with {activation}")

        # Reference comparison (FFN ≈ 4 ops: linear + activation + dropout + linear)
        ref_output = compute_reference(ffn, x)
        combo_assert_close(output, ref_output, torch.float32, num_ops=4, name=f"FFN {activation}")

    @pytest.mark.integration
    def test_swiglu_ffn(self, use_gems):
        """Test SwiGLU-style FFN."""
        batch_size, seq_len, d_model = 4, 256, 768
        dim_feedforward = d_model * 4

        from .models.ffn import SwiGLUFFN

        ffn = SwiGLUFFN(d_model, dim_feedforward, dropout=0.0)
        ffn = ffn.to(device).to(torch.float32)

        x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)

        ffn.eval()
        with torch.no_grad():
            output = ffn(x)

        assert output.shape == (batch_size, seq_len, d_model)
        check_no_nan(output, "SwiGLU FFN")

        # Reference comparison (SwiGLU ≈ 5 ops: gate_proj + up_proj + silu + mul + down_proj)
        ref_output = compute_reference(ffn, x)
        combo_assert_close(output, ref_output, torch.float32, num_ops=5, name="SwiGLU FFN")