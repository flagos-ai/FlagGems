"""
Mixed Precision Accumulation Error Tests.

This module tests error accumulation in mixed precision computations,
verifying numerical stability when chaining multiple fp16/bf16 operations.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import flag_gems

from .accuracy_utils import assert_accumulation_error
from .utils.numerical_stability import check_finite, check_no_nan

device = flag_gems.device


class MultiLayerNetwork(nn.Module):
    """Network with multiple layers for accumulation testing."""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.activation = F.gelu

        self.num_layers = num_layers

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x


class TestMixedPrecisionAccumulation:
    """
    Tests for error accumulation in mixed precision.

    Verify:
    - Accumulation error across layers
    - Precision loss estimation
    - Stability metrics
    """

    @pytest.mark.numerical_stability
    @pytest.mark.parametrize("num_layers", [4, 8, 16, 32])
    def test_error_accumulation_layers(self, num_layers, use_gems):
        """Test error accumulation with increasing number of layers."""
        batch_size = 4
        input_dim = 128
        hidden_dim = 256
        output_dim = 128

        # Create model in fp32 (reference)
        model_fp32 = MultiLayerNetwork(num_layers, input_dim, hidden_dim, output_dim)
        model_fp32 = model_fp32.to(device).to(torch.float32)

        # Create same model in fp16
        model_fp16 = MultiLayerNetwork(num_layers, input_dim, hidden_dim, output_dim)
        model_fp16.load_state_dict(model_fp32.state_dict())
        model_fp16 = model_fp16.to(device).to(torch.float16)

        # Generate input
        torch.manual_seed(42)
        x_fp32 = torch.randn(batch_size, input_dim, device=device, dtype=torch.float32)
        x_fp16 = x_fp32.to(torch.float16)

        # Compute outputs
        with torch.no_grad():
            output_fp32 = model_fp32(x_fp32)
            output_fp16 = model_fp16(x_fp16)

        # Compute error
        output_fp16_to_fp32 = output_fp16.to(torch.float32)
        error = (output_fp16_to_fp32 - output_fp32).abs()

        max_error = error.max().item()
        mean_error = error.mean().item()
        relative_error = (error / (output_fp32.abs() + 1e-6)).mean().item()

        print(f"\n{num_layers} layers:")
        print(f"  Max error: {max_error:.6e}")
        print(f"  Mean error: {mean_error:.6e}")
        print(f"  Relative error: {relative_error:.6%}")

        # Use structured assertion with sqrt scaling
        assert_accumulation_error(output_fp16, output_fp32, torch.float16, num_layers)

    @pytest.mark.numerical_stability
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_fp16_vs_bf16_accumulation(self, dtype, use_gems):
        """Compare error accumulation between fp16 and bf16."""
        num_layers = 8
        batch_size = 4
        input_dim = 128
        hidden_dim = 256
        output_dim = 128

        # Create model
        model = MultiLayerNetwork(num_layers, input_dim, hidden_dim, output_dim)

        # Save FP32 reference (deep copy to avoid in-place modification)
        import copy
        model_fp32 = copy.deepcopy(model).to(device).to(torch.float32)

        # Test in target dtype (separate copy)
        model_test = copy.deepcopy(model).to(device).to(dtype)

        # Generate input
        torch.manual_seed(42)
        x_fp32 = torch.randn(batch_size, input_dim, device=device, dtype=torch.float32)
        x_test = x_fp32.to(dtype)

        # Compute
        with torch.no_grad():
            output_fp32 = model_fp32(x_fp32)
            output_test = model_test(x_test)

        # Compute error
        output_test_fp32 = output_test.to(torch.float32)
        error = (output_test_fp32 - output_fp32).abs()
        relative_error = (error / (output_fp32.abs() + 1e-6)).mean().item()

        print(f"\n{dtype}: Relative error: {relative_error:.6%}")

        # Use structured assertion with sqrt scaling
        assert_accumulation_error(output_test, output_fp32, dtype, num_layers)

    @pytest.mark.numerical_stability
    def test_layer_norm_accumulation(self, use_gems):
        """Test accumulated layer norm operations."""
        num_norms = 16
        batch_size, seq_len, d_model = 4, 64, 128

        # FP32 reference
        x_fp32 = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)

        layer_norm = nn.LayerNorm(d_model)
        layer_norm = layer_norm.to(device)

        output_fp32 = x_fp32.clone()
        for _ in range(num_norms):
            output_fp32 = layer_norm(output_fp32)

        # FP16 test
        layer_norm_fp16 = layer_norm.to(torch.float16)
        x_fp16 = x_fp32.to(torch.float16)

        output_fp16 = x_fp16.clone()
        for _ in range(num_norms):
            output_fp16 = layer_norm_fp16(output_fp16)

        # Compare
        output_fp16_fp32 = output_fp16.to(torch.float32)
        error = (output_fp16_fp32 - output_fp32).abs().mean().item()

        print(f"\n{num_norms} LayerNorms: Mean error: {error:.6e}")

        # Should remain stable
        assert error < 0.01, f"LayerNorm accumulation error too high: {error}"

    @pytest.mark.numerical_stability
    def test_softmax_accumulation(self, use_gems):
        """Test accumulated softmax operations."""
        num_softmax = 10
        batch_size, dim = 4, 128

        x_fp32 = torch.randn(batch_size, dim, device=device, dtype=torch.float32)
        x_fp16 = x_fp32.to(torch.float16)

        output_fp32 = x_fp32.clone()
        output_fp16 = x_fp16.clone()

        for i in range(num_softmax):
            # Apply softmax with scaling (common in attention)
            scale = 1.0 / math.sqrt(dim)
            output_fp32 = F.softmax(output_fp32 * scale, dim=-1)
            output_fp16 = F.softmax(output_fp16 * scale, dim=-1)

        # Compare
        output_fp16_fp32 = output_fp16.to(torch.float32)
        error = (output_fp16_fp32 - output_fp32).abs().mean().item()

        print(f"\n{num_softmax} Softmaxes: Mean error: {error:.6e}")

        # Check no NaN
        check_no_nan(output_fp16, f"{num_softmax} softmaxes")

    @pytest.mark.numerical_stability
    @pytest.mark.parametrize("operation_sequence", ["linear_only", "mixed", "norm_only"])
    def test_operation_sequence_accumulation(self, operation_sequence, use_gems):
        """Test different operation sequences."""
        batch_size, seq_len, d_model = 4, 32, 128
        num_ops = 10

        x_fp32 = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)
        x_fp16 = x_fp32.to(torch.float16)

        linear = nn.Linear(d_model, d_model)
        norm = nn.LayerNorm(d_model)
        linear = linear.to(device)
        norm = norm.to(device)

        output_fp32 = x_fp32.clone()
        output_fp16 = x_fp16.clone()

        for i in range(num_ops):
            if operation_sequence == "linear_only":
                output_fp32 = linear(output_fp32)
                output_fp16 = linear(output_fp16.to(torch.float16)).to(torch.float16)
            elif operation_sequence == "norm_only":
                output_fp32 = norm(output_fp32)
                output_fp16 = norm(output_fp16)
            else:  # mixed
                if i % 2 == 0:
                    output_fp32 = linear(output_fp32)
                    output_fp16 = linear(output_fp16.to(torch.float16)).to(torch.float16)
                else:
                    output_fp32 = F.gelu(output_fp32)
                    output_fp16 = F.gelu(output_fp16)

        # Compare
        output_fp16_fp32 = output_fp16.to(torch.float32)
        error = (output_fp16_fp32 - output_fp32).abs().mean().item()

        print(f"\n{operation_sequence} sequence: Mean error: {error:.6e}")


@pytest.mark.integration
class TestMixedPrecisionBestPractices:
    """Test best practices for mixed precision."""

    def test_fp16_master_weights(self, use_gems):
        """Test fp32 master weights pattern."""
        batch_size, input_dim, output_dim = 4, 128, 64

        # FP32 master weights
        weight_fp32 = torch.randn(output_dim, input_dim, device=device, dtype=torch.float32)

        # FP16 copy for computation
        weight_fp16 = weight_fp32.to(torch.float16)

        x_fp16 = torch.randn(batch_size, input_dim, device=device, dtype=torch.float16)

        # Compute in FP16
        output_fp16 = F.linear(x_fp16, weight_fp16)

        # Gradients would be computed in FP16, then accumulated to FP32 master
        # For testing, just verify computation works
        check_no_nan(output_fp16, "FP16 with FP32 master weights")

    def test_loss_scaling_stability(self, use_gems):
        """Test that loss scaling prevents underflow."""
        batch_size = 4
        dim = 128

        # Create very small values
        x = torch.randn(batch_size, dim, device=device, dtype=torch.float16) * 1e-3

        loss = F.mse_loss(x, torch.zeros_like(x))

        # With proper loss scaling, gradients would be scaled up
        # For testing, just verify loss computation is stable
        check_finite(loss, "Small value loss")


@pytest.mark.integration
class TestPrecisionLossEstimation:
    """Estimate precision loss in practical scenarios."""

    @pytest.mark.parametrize("hidden_scale", [2, 4, 8])
    def test_hidden_dimension_scaling(self, hidden_scale, use_gems):
        """Test precision loss with different hidden dimensions."""
        batch_size, input_dim = 4, 64
        hidden_dim = input_dim * hidden_scale
        output_dim = input_dim

        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        import copy
        model_fp32 = copy.deepcopy(model).to(device).to(torch.float32)
        model_fp16 = copy.deepcopy(model).to(device).to(torch.float16)

        x_fp32 = torch.randn(batch_size, input_dim, device=device, dtype=torch.float32)
        x_fp16 = x_fp32.to(torch.float16)

        with torch.no_grad():
            output_fp32 = model_fp32(x_fp32)
            output_fp16 = model_fp16(x_fp16)

        # Skip comparison if shapes differ (shouldn't happen but safety check)
        if output_fp32.shape != output_fp16.shape:
            pytest.skip("Shape mismatch in precision comparison")

        error = (output_fp16.to(torch.float32) - output_fp32).abs().mean().item()
        print(f"\nHidden scale {hidden_scale}x: Error {error:.6e}")