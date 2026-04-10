"""
Backward Gradient Flow Combination Tests.

This module tests gradient flow through multiple operator combinations,
verifying gradient correctness and numerical stability in backward propagation.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import flag_gems

from .models.transformer_block import TransformerBlock
from .accuracy_utils import compute_reference_with_grad, combo_assert_close, assert_gradient_close
from .utils.numerical_stability import check_finite, check_gradient_health

device = flag_gems.device


class SimpleMLP(nn.Module):
    """Simple MLP for gradient flow testing."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = F.gelu

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class MultiLayerTransformer(nn.Module):
    """
    Multi-layer Transformer for gradient flow testing.

    Tests gradient propagation through multiple transformer blocks.
    """

    def __init__(self, num_layers, d_model, nhead, dim_feedforward):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, nhead, dim_feedforward, dropout=0.0)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TestBackwardGradientFlow:
    """
    Tests for gradient flow through multi-operator combinations.

    Verify:
    - Gradient propagation through chains
    - Gradient numerical stability
    - Gradient correctness vs finite differences
    """

    @pytest.mark.integration
    def test_single_layer_gradient(self, use_gems):
        """Test gradient through single Transformer layer."""
        batch_size, seq_len, d_model = 2, 64, 256
        nhead = 8
        dim_feedforward = d_model * 4

        model = TransformerBlock(d_model, nhead, dim_feedforward, dropout=0.0)
        model = model.to(device).to(torch.float32)

        x = torch.randn(
            batch_size, seq_len, d_model, device=device, dtype=torch.float32, requires_grad=True
        )

        # Forward
        output = model(x)

        # Backward
        loss = output.sum()
        loss.backward()

        # Check input gradient
        assert x.grad is not None, "Input gradient is None"
        check_finite(x.grad, "Single layer input gradient")

        # Check parameter gradients
        check_gradient_health(model, "Single layer")

        # Reference gradient comparison (≈ 10 ops in a TransformerBlock)
        ref_output, ref_input_grads, ref_param_grads = compute_reference_with_grad(model, x)
        combo_assert_close(output.detach(), ref_output, torch.float32, num_ops=10, name="Single layer forward")
        if 0 in ref_input_grads:
            assert_gradient_close(x.grad, ref_input_grads[0], torch.float32, num_ops=10, name="Single layer input grad")

    @pytest.mark.integration
    @pytest.mark.parametrize("num_layers", [2, 4, 8])
    def test_multi_layer_gradient_flow(self, num_layers, use_gems):
        """Test gradient flow through multiple layers."""
        batch_size, seq_len, d_model = 2, 32, 128
        nhead = 4
        dim_feedforward = d_model * 4

        model = MultiLayerTransformer(num_layers, d_model, nhead, dim_feedforward)
        model = model.to(device).to(torch.float32)

        x = torch.randn(
            batch_size, seq_len, d_model, device=device, dtype=torch.float32, requires_grad=True
        )

        # Forward
        output = model(x)

        # Backward
        loss = output.sum()
        loss.backward()

        # Check gradients
        assert x.grad is not None, f"Input gradient is None ({num_layers} layers)"
        check_finite(x.grad, f"{num_layers}-layer gradient")

        # Gradient should not vanish (check norm)
        grad_norm = x.grad.norm().item()
        assert grad_norm > 1e-6, f"Gradient may have vanished: {grad_norm}"
        assert grad_norm < 1e3, f"Gradient may have exploded: {grad_norm}"

        # Reference gradient comparison (10 ops per layer)
        num_ops = num_layers * 10
        ref_output, ref_input_grads, _ = compute_reference_with_grad(model, x)
        combo_assert_close(output.detach(), ref_output, torch.float32, num_ops=num_ops, name=f"{num_layers}-layer forward")
        if 0 in ref_input_grads:
            assert_gradient_close(x.grad, ref_input_grads[0], torch.float32, num_ops=num_ops, name=f"{num_layers}-layer input grad")

    @pytest.mark.integration
    def test_gradient_with_masking(self, use_gems):
        """Test gradient flow with attention mask."""
        batch_size, seq_len, d_model = 2, 64, 256
        nhead = 8
        dim_feedforward = d_model * 4

        model = TransformerBlock(d_model, nhead, dim_feedforward, dropout=0.0)
        model = model.to(device).to(torch.float32)

        x = torch.randn(
            batch_size, seq_len, d_model, device=device, dtype=torch.float32, requires_grad=True
        )

        # Create padding mask (50% of sequence)
        mask = torch.zeros(batch_size, 1, 1, seq_len, dtype=torch.bool, device=device)
        mask[:, :, :, seq_len // 2 :] = True

        # Forward with mask
        output = model(x, mask=mask)

        # Backward
        loss = output.sum()
        loss.backward()

        # Check gradients
        assert x.grad is not None
        check_finite(x.grad, "Gradient with mask")

    @pytest.mark.integration
    def test_gradient_causal_attention(self, use_gems):
        """Test gradient with causal mask."""
        batch_size, seq_len, d_model = 2, 64, 256
        nhead = 8
        dim_feedforward = d_model * 4

        model = TransformerBlock(d_model, nhead, dim_feedforward, dropout=0.0)
        model = model.to(device).to(torch.float32)

        x = torch.randn(
            batch_size, seq_len, d_model, device=device, dtype=torch.float32, requires_grad=True
        )

        # Forward with causal mask
        output = model(x, is_causal=True)

        # Backward
        loss = output.sum()
        loss.backward()

        # Check gradients
        check_finite(x.grad, "Gradient with causal mask")

    @pytest.mark.numerical_stability
    def test_gradient_large_values(self, use_gems):
        """Test gradient stability with large input values."""
        batch_size, seq_len, d_model = 2, 32, 128

        model = SimpleMLP(d_model * seq_len, d_model * 4, d_model)
        model = model.to(device).to(torch.float32)

        # Large input values
        x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32) * 50
        x = x.reshape(batch_size, -1)
        x.requires_grad = True

        # Forward
        output = model(x)

        # Backward
        loss = output.sum()
        loss.backward()

        # Check gradient finite
        check_finite(x.grad, "Gradient with large values")

    @pytest.mark.numerical_stability
    def test_gradient_small_values(self, use_gems):
        """Test gradient stability with small input values."""
        batch_size, seq_len, d_model = 2, 32, 128

        model = SimpleMLP(d_model * seq_len, d_model * 4, d_model)
        model = model.to(device).to(torch.float32)

        # Small input values
        x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32) * 1e-4
        x = x.reshape(batch_size, -1)
        x.requires_grad = True

        # Forward
        output = model(x)

        # Backward
        loss = output.sum()
        loss.backward()

        # Check gradient finite
        check_finite(x.grad, "Gradient with small values")


class TestGradientCorrectness:
    """
    Tests for gradient correctness using finite differences.
    """

    @pytest.mark.integration
    def test_gradient_vs_finite_difference(self, use_gems):
        """Compare analytical gradient with finite difference."""
        input_dim = 32
        hidden_dim = 64
        output_dim = 16

        model = SimpleMLP(input_dim, hidden_dim, output_dim)
        model = model.to(device).to(torch.float64)  # Use fp64 for numerical stability

        x = torch.randn(1, input_dim, device=device, dtype=torch.float64, requires_grad=True)

        # Compute analytical gradient
        output = model(x)
        loss = output.sum()
        loss.backward()
        analytical_grad = x.grad.clone()

        # Compute finite difference gradient
        eps = 1e-5
        x_nograd = x.detach().clone()

        finite_diff_grad = torch.zeros_like(x_nograd)
        for i in range(input_dim):
            x_plus = x_nograd.clone()
            x_plus[0, i] += eps
            x_minus = x_nograd.clone()
            x_minus[0, i] -= eps

            loss_plus = model(x_plus).sum()
            loss_minus = model(x_minus).sum()

            finite_diff_grad[0, i] = (loss_plus - loss_minus) / (2 * eps)

        # Compare
        max_diff = (analytical_grad - finite_diff_grad).abs().max().item()
        print(f"\nMax gradient difference: {max_diff:.6e}")

        # Should be close (< 1e-3 for double precision)
        assert max_diff < 1e-3, f"Gradient differs from finite difference: {max_diff}"


@pytest.mark.integration
class TestGradientFlowEdgeCases:
    """Edge cases in gradient flow."""

    def test_gradient_through_softmax(self, use_gems):
        """Test gradient through softmax."""
        batch_size, dim = 4, 128
        x = torch.randn(batch_size, dim, device=device, dtype=torch.float32, requires_grad=True)

        output = F.softmax(x, dim=-1)
        loss = output.sum()
        loss.backward()

        check_finite(x.grad, "Gradient through softmax")

    def test_gradient_through_layer_norm(self, use_gems):
        """Test gradient through layer norm."""
        batch_size, seq_len, d_model = 4, 32, 64

        x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32, requires_grad=True)

        weight = torch.ones(d_model, device=device, dtype=torch.float32, requires_grad=True)
        bias = torch.zeros(d_model, device=device, dtype=torch.float32, requires_grad=True)

        output = F.layer_norm(x, [d_model], weight, bias)
        loss = output.sum()
        loss.backward()

        check_finite(x.grad, "Gradient through layer_norm (input)")
        check_finite(weight.grad, "Gradient through layer_norm (weight)")
        check_finite(bias.grad, "Gradient through layer_norm (bias)")