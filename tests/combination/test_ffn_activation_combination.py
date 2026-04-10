"""
FFN Activation Function Combination Tests.

This module tests different FFN activation function combinations
including GELU, SiLU, SwiGLU, and GeGLU patterns.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import flag_gems

from .models.ffn import StandardFFN, SwiGLUFFN, GeGLUFFN
from .accuracy_utils import compute_reference, combo_assert_close
from .utils.numerical_stability import (
    check_finite,
    check_no_nan,
    generate_stress_input,
)

device = flag_gems.device


class TestFFNActivationCombinations:
    """
    Tests for FFN activation function combinations.

    Tests include:
    - Standard FFN with different activations (GELU, ReLU, SiLU)
    - Gated variants (SwiGLU, GeGLU)
    - Numerical stability with extreme values
    - Backward pass correctness
    """

    @pytest.fixture
    def ffn_config(self):
        """Default FFN configuration."""
        return {
            "batch_size": 4,
            "seq_len": 256,
            "d_model": 768,
            "dim_feedforward": 3072,
        }

    @pytest.mark.integration
    @pytest.mark.parametrize("activation", ["gelu", "relu", "silu"])
    def test_standard_ffn_activations(self, activation, ffn_config, use_gems):
        """Test standard FFN with different activation functions."""
        B = ffn_config["batch_size"]
        L = ffn_config["seq_len"]
        D = ffn_config["d_model"]
        H = ffn_config["dim_feedforward"]

        # Create FFN
        ffn = StandardFFN(D, H, activation=activation, dropout=0.0)
        ffn = ffn.to(device).to(torch.float32)

        # Create input
        x = torch.randn(B, L, D, device=device, dtype=torch.float32)

        # Forward pass
        output = ffn(x)

        # Verify
        assert output.shape == (B, L, D), f"Output shape mismatch for {activation}"
        check_no_nan(output, f"FFN with {activation}")
        check_finite(output, f"FFN with {activation}")

    @pytest.mark.integration
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_ffn_mixed_precision(self, dtype, ffn_config, use_gems):
        """Test FFN with different precision levels."""
        B = ffn_config["batch_size"]
        L = ffn_config["seq_len"]
        D = ffn_config["d_model"]
        H = ffn_config["dim_feedforward"]

        ffn = StandardFFN(D, H, activation="gelu", dropout=0.0)
        ffn = ffn.to(device).to(dtype)

        x = torch.randn(B, L, D, device=device, dtype=dtype)

        output = ffn(x)

        assert output.shape == (B, L, D)
        check_no_nan(output, f"FFN {dtype}")

    @pytest.mark.integration
    def test_swiglu_ffn(self, ffn_config, use_gems):
        """Test SwiGLU-style FFN (LLaMA architecture)."""
        B = ffn_config["batch_size"]
        L = ffn_config["seq_len"]
        D = ffn_config["d_model"]
        H = ffn_config["dim_feedforward"]

        ffn = SwiGLUFFN(D, H, dropout=0.0)
        ffn = ffn.to(device).to(torch.float32)

        x = torch.randn(B, L, D, device=device, dtype=torch.float32)

        output = ffn(x)

        assert output.shape == (B, L, D)
        check_no_nan(output, "SwiGLU FFN")
        check_finite(output, "SwiGLU FFN")

    @pytest.mark.integration
    def test_geglu_ffn(self, ffn_config, use_gems):
        """Test GeGLU-style FFN."""
        B = ffn_config["batch_size"]
        L = ffn_config["seq_len"]
        D = ffn_config["d_model"]
        H = ffn_config["dim_feedforward"]

        ffn = GeGLUFFN(D, H, dropout=0.0)
        ffn = ffn.to(device).to(torch.float32)

        x = torch.randn(B, L, D, device=device, dtype=torch.float32)

        output = ffn(x)

        assert output.shape == (B, L, D)
        check_no_nan(output, "GeGLU FFN")
        check_finite(output, "GeGLU FFN")

    @pytest.mark.numerical_stability
    @pytest.mark.parametrize("activation", ["gelu", "silu"])
    def test_ffn_large_values(self, activation, ffn_config, use_gems):
        """Test FFN with large input values (potential overflow)."""
        B = ffn_config["batch_size"]
        L = ffn_config["seq_len"]
        D = ffn_config["d_model"]
        H = ffn_config["dim_feedforward"]

        ffn = StandardFFN(D, H, activation=activation, dropout=0.0)
        ffn = ffn.to(device).to(torch.float32)

        # Large values
        x = torch.randn(B, L, D, device=device, dtype=torch.float32) * 20

        output = ffn(x)

        check_no_nan(output, f"FFN {activation} large values")

    @pytest.mark.numerical_stability
    @pytest.mark.parametrize("activation", ["gelu", "silu"])
    def test_ffn_small_values(self, activation, ffn_config, use_gems):
        """Test FFN with small input values (potential underflow)."""
        B = ffn_config["batch_size"]
        L = ffn_config["seq_len"]
        D = ffn_config["d_model"]
        H = ffn_config["dim_feedforward"]

        ffn = StandardFFN(D, H, activation=activation, dropout=0.0)
        ffn = ffn.to(device).to(torch.float32)

        # Small values
        x = torch.randn(B, L, D, device=device, dtype=torch.float32) * 1e-6

        output = ffn(x)

        check_finite(output, f"FFN {activation} small values")

    @pytest.mark.integration
    def test_ffn_backward(self, ffn_config, use_gems):
        """Test FFN gradient computation."""
        B = ffn_config["batch_size"]
        L = ffn_config["seq_len"]
        D = ffn_config["d_model"]
        H = ffn_config["dim_feedforward"]

        ffn = StandardFFN(D, H, activation="gelu", dropout=0.0)
        ffn = ffn.to(device).to(torch.float32)

        x = torch.randn(B, L, D, device=device, dtype=torch.float32, requires_grad=True)

        # Forward
        output = ffn(x)

        # Backward
        loss = output.sum()
        loss.backward()

        # Check input gradient
        assert x.grad is not None, "Input gradient is None"
        check_finite(x.grad, "FFN input gradient")

        # Check parameter gradients
        for name, param in ffn.named_parameters():
            assert param.grad is not None, f"Gradient for {name} is None"
            check_finite(param.grad, f"FFN gradient {name}")

    @pytest.mark.integration
    def test_swiglu_backward(self, ffn_config, use_gems):
        """Test SwiGLU gradient computation."""
        B = ffn_config["batch_size"]
        L = ffn_config["seq_len"]
        D = ffn_config["d_model"]
        H = ffn_config["dim_feedforward"]

        ffn = SwiGLUFFN(D, H, dropout=0.0)
        ffn = ffn.to(device).to(torch.float32)

        x = torch.randn(B, L, D, device=device, dtype=torch.float32, requires_grad=True)

        # Forward
        output = ffn(x)

        # Backward
        loss = output.sum()
        loss.backward()

        # Check gradients
        assert x.grad is not None
        check_finite(x.grad, "SwiGLU input gradient")


class TestActivationFunctions:
    """
    Direct tests for individual activation functions.
    """

    @pytest.mark.parametrize(
        "activation_fn,name",
        [
            (F.gelu, "gelu"),
            (F.silu, "silu"),
            (F.relu, "relu"),
            (torch.sigmoid, "sigmoid"),
            (torch.tanh, "tanh"),
        ],
    )
    def test_activation_basic(self, activation_fn, name, use_gems):
        """Test basic activation function behavior."""
        shape = (4, 256, 768)
        x = torch.randn(shape, device=device, dtype=torch.float32)

        output = activation_fn(x)

        assert output.shape == shape
        check_no_nan(output, f"{name} basic")

    @pytest.mark.parametrize(
        "activation_fn,name",
        [
            (F.gelu, "gelu"),
            (F.silu, "silu"),
        ],
    )
    @pytest.mark.parametrize("scale", [1.0, 10.0, 50.0, 100.0])
    def test_activation_large_values(self, activation_fn, name, scale, use_gems):
        """Test activation functions with large values."""
        shape = (4, 256, 768)
        x = torch.randn(shape, device=device, dtype=torch.float16) * scale

        output = activation_fn(x)

        check_no_nan(output, f"{name} scale={scale}")

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_gelu_precision(self, dtype, use_gems):
        """Test GELU precision across dtypes."""
        shape = (4, 256, 768)

        # Create input in fp32 for reference
        x_fp32 = torch.randn(shape, device=device, dtype=torch.float32)
        output_ref = F.gelu(x_fp32)

        # Compute in target dtype
        x = x_fp32.to(dtype)
        output = F.gelu(x)

        # Compare
        output_fp32 = output.to(torch.float32)
        error = (output_fp32 - output_ref).abs()

        max_error = error.max().item()
        mean_error = error.mean().item()

        print(f"\nGELU {dtype}: max_error={max_error:.6e}, mean_error={mean_error:.6e}")

        check_finite(output, f"GELU {dtype}")

    @pytest.mark.parametrize("approximate", ["none", "tanh"])
    def test_gelu_approximations(self, approximate, use_gems):
        """Test GELU with different approximations."""
        shape = (4, 256, 768)
        x = torch.randn(shape, device=device, dtype=torch.float32)

        output = F.gelu(x, approximate=approximate)

        check_no_nan(output, f"GELU approximate={approximate}")

    def test_gelu_backward(self, use_gems):
        """Test GELU gradient computation."""
        shape = (4, 256, 768)
        x = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)

        output = F.gelu(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        check_finite(x.grad, "GELU gradient")

    def test_silu_backward(self, use_gems):
        """Test SiLU gradient computation."""
        shape = (4, 256, 768)
        x = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)

        output = F.silu(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        check_finite(x.grad, "SiLU gradient")


class TestFFNVsPyTorch:
    """
    Comparison tests between FlagGems and PyTorch FFN.
    """

    @pytest.mark.comparison
    @pytest.mark.parametrize("activation", ["gelu", "relu"])
    def test_ffn_output_comparison(self, activation, use_gems):
        """Compare FFN output between FlagGems (GPU) and PyTorch reference."""
        B, L, D, H = 4, 256, 768, 3072

        # Set seed for reproducibility
        torch.manual_seed(42)

        # Create FFN
        ffn = StandardFFN(D, H, activation=activation, dropout=0.0)
        ffn = ffn.to(device).to(torch.float32)
        ffn.eval()

        # Create input
        x = torch.randn(B, L, D, device=device, dtype=torch.float32)

        # Compute with FlagGems (use_gems fixture is active)
        with torch.no_grad():
            output_gems = ffn(x.clone())

        # Compute reference (three-level strategy: GPU fp64 → CPU fp32)
        output_ref = compute_reference(ffn, x.clone())

        # Compare (FFN ≈ 4 ops: linear + activation + dropout + linear)
        combo_assert_close(
            output_gems, output_ref, torch.float32, num_ops=4,
            name=f"FFN {activation}",
        )