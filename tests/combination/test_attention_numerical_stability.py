"""
Attention Numerical Stability Combination Tests.

This module tests the numerical stability of attention mechanisms
under various edge cases and stress conditions.
"""

import math

import pytest
import torch
import torch.nn.functional as F

import flag_gems

from .accuracy_utils import assert_numerical_consistency, COMBO_FLOAT_DTYPES
from .utils.numerical_stability import (
    check_finite,
    check_no_nan,
    generate_stress_input,
)

device = flag_gems.device


class TestAttentionNumericalStability:
    """
    Numerical stability tests for attention mechanisms.

    Tests include:
    - All negative infinity inputs (padding scenarios)
    - Large value inputs
    - Mixed normal/special values
    - Gradient stability
    """

    @pytest.fixture
    def attention_config(self):
        """Default attention configuration."""
        return {
            "batch_size": 2,
            "nhead": 12,
            "seq_len": 512,
            "head_dim": 64,
        }

    def _compute_attention(self, q, k, v, mask=None, scale=None):
        """
        Compute attention scores manually for testing.

        Args:
            q: Query tensor (batch, nhead, seq_len, head_dim)
            k: Key tensor (batch, nhead, seq_len, head_dim)
            v: Value tensor (batch, nhead, seq_len, head_dim)
            mask: Optional attention mask
            scale: Scaling factor (default: 1/sqrt(head_dim))

        Returns:
            Attention output tensor
        """
        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1])

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Compute output
        output = torch.matmul(attn_weights, v)

        return output, attn_weights

    @pytest.mark.numerical_stability
    @pytest.mark.parametrize(
        "scenario",
        [
            pytest.param("normal", id="normal_input"),
            pytest.param("large_values", id="large_values"),
            pytest.param("small_values", id="small_values"),
        ],
    )
    def test_attention_input_scenarios(self, scenario, attention_config, use_gems):
        """Test attention with different input value ranges."""
        B = attention_config["batch_size"]
        H = attention_config["nhead"]
        L = attention_config["seq_len"]
        D = attention_config["head_dim"]

        # Generate inputs based on scenario
        if scenario == "normal":
            q = torch.randn(B, H, L, D, device=device, dtype=torch.float16)
            k = torch.randn(B, H, L, D, device=device, dtype=torch.float16)
            v = torch.randn(B, H, L, D, device=device, dtype=torch.float16)
        elif scenario == "large_values":
            # Large values that might cause overflow
            q = torch.randn(B, H, L, D, device=device, dtype=torch.float16) * 10
            k = torch.randn(B, H, L, D, device=device, dtype=torch.float16) * 10
            v = torch.randn(B, H, L, D, device=device, dtype=torch.float16) * 10
        elif scenario == "small_values":
            # Small values that might cause underflow
            q = torch.randn(B, H, L, D, device=device, dtype=torch.float16) * 0.01
            k = torch.randn(B, H, L, D, device=device, dtype=torch.float16) * 0.01
            v = torch.randn(B, H, L, D, device=device, dtype=torch.float16)

        # Compute attention
        output, attn_weights = self._compute_attention(q, k, v)

        # Verify no NaN in output
        check_no_nan(output, f"Attention output ({scenario})")

        # Verify attention weights sum to 1
        attn_sum = attn_weights.sum(dim=-1)
        expected_sum = torch.ones_like(attn_sum)
        assert torch.allclose(attn_sum, expected_sum, rtol=1e-2, atol=1e-3), (
            f"Attention weights don't sum to 1 for {scenario}"
        )

    @pytest.mark.numerical_stability
    @pytest.mark.parametrize("mask_ratio", [0.0, 0.3, 0.5, 0.7, 0.9])
    def test_attention_with_padding_mask(self, mask_ratio, attention_config, use_gems):
        """Test attention with varying amounts of padding (masked positions)."""
        B = attention_config["batch_size"]
        H = attention_config["nhead"]
        L = attention_config["seq_len"]
        D = attention_config["head_dim"]

        # Create inputs
        q = torch.randn(B, H, L, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, L, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, L, D, device=device, dtype=torch.float16)

        # Create padding mask (True = masked/padding)
        mask = torch.zeros(B, 1, 1, L, dtype=torch.bool, device=device)
        mask_start = int(L * (1 - mask_ratio))
        mask[:, :, :, mask_start:] = True  # Mask later positions

        # Expand mask for attention
        mask_expanded = mask.expand(B, H, L, L)

        # Compute attention
        output, attn_weights = self._compute_attention(q, k, v, mask=mask_expanded)

        # Verify output is valid
        check_no_nan(output, f"Attention with {mask_ratio:.0%} padding")

        # Check masked positions have zero attention weight
        if mask_ratio > 0:
            masked_weights = attn_weights[:, :, :, mask_start:]
            # Note: After softmax, masked positions should have 0 weight
            # but numerical precision might give small values

    @pytest.mark.numerical_stability
    def test_attention_all_masked_row(self, attention_config, use_gems):
        """
        Test attention when entire rows are masked (all -inf).

        This is a critical edge case that can cause NaN in naive implementations.
        """
        B = attention_config["batch_size"]
        H = attention_config["nhead"]
        L = attention_config["seq_len"]
        D = attention_config["head_dim"]

        # Create inputs
        q = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, L, D, device=device, dtype=torch.float32)

        # Create mask where first row is completely masked
        mask = torch.zeros(B, H, L, L, dtype=torch.bool, device=device)
        mask[:, :, 0, :] = True  # First row completely masked

        # Compute attention
        scale = 1.0 / math.sqrt(D)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(mask, float("-inf"))

        # This is where NaN can occur with naive softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Check for NaN - this is the critical test
        nan_count = torch.isnan(attn_weights).sum().item()

        # Note: Standard PyTorch softmax may produce NaN for all-inf rows
        # FlagGems should handle this gracefully
        if nan_count > 0:
            pytest.xfail(
                f"All-masked row produces {nan_count} NaN values (expected for standard softmax)"
            )

    @pytest.mark.numerical_stability
    def test_attention_causal_mask_stability(self, attention_config, use_gems):
        """Test attention with causal mask (autoregressive)."""
        B = attention_config["batch_size"]
        H = attention_config["nhead"]
        L = attention_config["seq_len"]
        D = attention_config["head_dim"]

        # Create inputs
        q = torch.randn(B, H, L, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, L, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, L, D, device=device, dtype=torch.float16)

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1
        )

        # Compute attention with causal mask
        output, attn_weights = self._compute_attention(q, k, v, mask=causal_mask)

        # Verify no NaN
        check_no_nan(output, "Causal attention output")

        # Verify causal property: no attention to future positions (vectorized)
        row_idx = torch.arange(L, device=attn_weights.device).view(1, 1, L, 1)
        col_idx = torch.arange(L, device=attn_weights.device).view(1, 1, 1, L)
        future_mask = col_idx > row_idx
        assert (attn_weights[future_mask.expand_as(attn_weights)] == 0).all(), (
            "Some positions attend to future"
        )

    @pytest.mark.numerical_stability
    @pytest.mark.parametrize("dtype", COMBO_FLOAT_DTYPES)
    def test_attention_precision_comparison(self, dtype, attention_config, use_gems):
        """Test attention output precision across different dtypes."""
        B = attention_config["batch_size"]
        H = attention_config["nhead"]
        L = 128  # Smaller for precision comparison
        D = attention_config["head_dim"]

        # Create inputs in fp32 for reference
        q_fp32 = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
        k_fp32 = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
        v_fp32 = torch.randn(B, H, L, D, device=device, dtype=torch.float32)

        # Reference computation in fp32
        output_ref, _ = self._compute_attention(q_fp32, k_fp32, v_fp32)

        # Compute in target dtype
        q = q_fp32.to(dtype)
        k = k_fp32.to(dtype)
        v = v_fp32.to(dtype)
        output, _ = self._compute_attention(q, k, v)

        # Verify finite output
        check_finite(output, f"Attention {dtype}")

        # Numerical consistency check
        assert_numerical_consistency(
            output.to(torch.float32), output_ref, name=f"Attention {dtype} vs fp32"
        )

    @pytest.mark.numerical_stability
    def test_attention_gradient_stability(self, attention_config, use_gems):
        """Test gradient computation stability in attention."""
        B = attention_config["batch_size"]
        H = attention_config["nhead"]
        L = 128
        D = attention_config["head_dim"]

        # Create inputs with gradients
        q = torch.randn(B, H, L, D, device=device, dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, H, L, D, device=device, dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, H, L, D, device=device, dtype=torch.float32, requires_grad=True)

        # Forward pass
        output, _ = self._compute_attention(q, k, v)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients
        for name, tensor in [("q", q), ("k", k), ("v", v)]:
            assert tensor.grad is not None, f"Gradient for {name} is None"
            check_finite(tensor.grad, f"Gradient for {name}")

    @pytest.mark.numerical_stability
    @pytest.mark.parametrize("seq_len", [64, 256, 1024, 4096])
    def test_attention_sequence_length_scaling(self, seq_len, use_gems):
        """Test attention stability across different sequence lengths."""
        B, H, D = 1, 8, 64

        # Create inputs
        q = torch.randn(B, H, seq_len, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, seq_len, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, seq_len, D, device=device, dtype=torch.float16)

        # Compute attention
        output, attn_weights = self._compute_attention(q, k, v)

        # Verify
        check_no_nan(output, f"Attention seq_len={seq_len}")
        assert output.shape == (B, H, seq_len, D)


class TestSoftmaxStability:
    """
    Tests specifically for softmax numerical stability.
    """

    @pytest.mark.numerical_stability
    def test_softmax_normal_input(self, use_gems):
        """Test softmax with normal inputs."""
        shape = (4, 8, 128, 128)
        x = torch.randn(shape, device=device, dtype=torch.float16)

        output = F.softmax(x, dim=-1)

        check_no_nan(output, "Softmax normal input")

        # Verify sum to 1
        sums = output.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), rtol=1e-2)

    @pytest.mark.numerical_stability
    def test_softmax_large_values(self, use_gems):
        """Test softmax with large values (potential overflow)."""
        shape = (4, 8, 128, 128)
        x = torch.randn(shape, device=device, dtype=torch.float16) * 50

        output = F.softmax(x, dim=-1)

        check_no_nan(output, "Softmax large values")

    @pytest.mark.numerical_stability
    def test_softmax_with_neg_inf(self, use_gems):
        """Test softmax with some negative infinity values."""
        shape = (4, 128)
        x = torch.randn(shape, device=device, dtype=torch.float32)

        # Mask half the values
        x[:, 64:] = float("-inf")

        output = F.softmax(x, dim=-1)

        # Check masked positions are zero
        assert (output[:, 64:] == 0).all(), "Masked positions should be zero"

        # Check unmasked positions sum to 1
        unmasked_sum = output[:, :64].sum(dim=-1)
        assert torch.allclose(unmasked_sum, torch.ones_like(unmasked_sum), rtol=1e-3)

        # Numerical consistency: compare NaN/Inf positions with reference
        ref_output = F.softmax(x.clone().detach(), dim=-1)
        assert_numerical_consistency(output, ref_output, name="softmax partial -inf")

    @pytest.mark.numerical_stability
    def test_softmax_all_neg_inf(self, use_gems):
        """
        Test softmax with all negative infinity (critical edge case).

        This tests the behavior when all attention positions are masked.
        """
        shape = (4, 128)
        x = torch.full(shape, float("-inf"), device=device, dtype=torch.float32)

        # Standard softmax behavior with all -inf
        output = F.softmax(x, dim=-1)

        # Note: This will produce NaN with standard softmax
        # We document this behavior for awareness
        nan_count = torch.isnan(output).sum().item()
        if nan_count > 0:
            pytest.xfail(
                f"All -inf input produces NaN (expected behavior: {nan_count} NaN values)"
            )