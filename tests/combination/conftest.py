"""
Combination tests configuration and fixtures.

This module provides fixtures and utilities for combination testing
of FlagGems operators in realistic workflows.

NOTE: This conftest.py is designed to be compatible with the parent
tests/conftest.py. It inherits TO_CPU, QUICK_MODE, and other global
configurations.
"""

import sys

import pytest
import torch

# Add parent test directory to path to import parent conftest
sys.path.insert(0, str(__file__).parent.parent)

import flag_gems

device = flag_gems.device

# Import global variables from parent conftest if available
try:
    from tests.conftest import TO_CPU, QUICK_MODE, RECORD_LOG
except ImportError:
    # Default values if parent conftest not available
    TO_CPU = False
    QUICK_MODE = False
    RECORD_LOG = False


# ============================================================================
# Register custom markers for combination tests
# ============================================================================


def pytest_configure(config):
    """Register custom markers for combination tests."""
    # Register markers specific to combination tests
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (end-to-end scenarios)"
    )
    config.addinivalue_line(
        "markers", "numerical_stability: mark test as numerical stability test"
    )
    config.addinivalue_line(
        "markers", "stress: mark test as stress test (may be slow)"
    )
    config.addinivalue_line(
        "markers", "comparison: mark test as comparison test (against PyTorch baseline)"
    )
    config.addinivalue_line(
        "markers", "transformer: mark test related to Transformer models"
    )
    config.addinivalue_line(
        "markers", "attention: mark test related to attention mechanisms"
    )
    config.addinivalue_line(
        markers", "ffn: mark test related to FFN modules"
    )


# ============================================================================
# Fixtures for common test configurations
# ============================================================================


@pytest.fixture(scope="module")
def use_gems():
    """Context manager to enable FlagGems for a test module."""
    with flag_gems.use_gems():
        yield


@pytest.fixture
def Float16():
    """Fixture for torch.float16 dtype."""
    return torch.float16


@pytest.fixture
def BFloat16():
    """Fixture for torch.bfloat16 dtype."""
    return torch.bfloat16


@pytest.fixture
def Float32():
    """Fixture for torch.float32 dtype."""
    return torch.float32


@pytest.fixture(
    params=[
        pytest.param(torch.float16, id="fp16"),
        pytest.param(torch.bfloat16, id="bf16"),
        pytest.param(torch.float32, id="fp32"),
    ],
    scope="module",
)
def dtype(request):
    """Parametrized fixture for common dtypes."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param(1, id="batch1"),
        pytest.param(4, id="batch4", marks=pytest.mark.skipif(QUICK_MODE, reason="Quick mode")),
        pytest.param(16, id="batch16", marks=pytest.mark.skipif(QUICK_MODE, reason="Quick mode")),
    ],
    scope="module",
)
def batch_size(request):
    """Parametrized fixture for batch sizes (respects QUICK_MODE)."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param(128, id="seq128"),
        pytest.param(512, id="seq512", marks=pytest.mark.skipif(QUICK_MODE, reason="Quick mode")),
        pytest.param(2048, id="seq2048", marks=pytest.mark.skipif(QUICK_MODE, reason="Quick mode")),
    ],
    scope="module",
)
def seq_len(request):
    """Parametrized fixture for sequence lengths (respects QUICK_MODE)."""
    return request.param


# ============================================================================
# Transformer configuration fixtures
# ============================================================================


@pytest.fixture
def transformer_config():
    """Standard Transformer configuration."""
    return {
        "d_model": 768,
        "nhead": 12,
        "dim_feedforward": 3072,
        "dropout": 0.1,
        "activation": "gelu",
    }


@pytest.fixture
def llama_config():
    """LLaMA-style Transformer configuration."""
    return {
        "d_model": 4096,
        "nhead": 32,
        "dim_feedforward": 11008,
        "dropout": 0.0,
        "activation": "silu",
        "norm": "rms_norm",
    }


# ============================================================================
# Numerical stability test utilities
# ============================================================================


@pytest.fixture
def numerical_scenarios():
    """Scenarios for numerical stability testing."""
    return {
        "normal": {
            "description": "Normal distribution inputs",
            "generator": lambda shape, dtype, device: torch.randn(shape, dtype=dtype, device=device),
        },
        "all_neg_inf": {
            "description": "All negative infinity (padding scenario)",
            "generator": lambda shape, dtype, device: torch.full(
                shape, float("-inf"), dtype=dtype, device=device
            ),
        },
        "large_values": {
            "description": "Large values (magnitude > 10)",
            "generator": lambda shape, dtype, device: torch.randn(shape, dtype=dtype, device=device) * 20,
        },
        "mixed_specials": {
            "description": "Mixed normal, NaN, Inf values",
            "generator": lambda shape, dtype, device: _generate_mixed_specials(shape, dtype, device),
        },
    }


def _generate_mixed_specials(shape, dtype, device):
    """Generate tensor with mixed normal and special values."""
    x = torch.randn(shape, dtype=dtype, device=device)
    # Inject NaN at 1% locations
    nan_mask = torch.rand(shape, device=device) < 0.01
    x[nan_mask] = float("nan")
    # Inject Inf at 0.5% locations
    inf_mask = torch.rand(shape, device=device) < 0.005
    x[inf_mask] = float("inf")
    return x


# ============================================================================
# Assertion utilities (compatible with parent conftest)
# ============================================================================


def assert_numerical_stability(output, name="", rtol=1e-3, atol=1e-4):
    """
    Assert that output tensor has no NaN or Inf values.

    Args:
        output: Output tensor to check
        name: Name of the test for error messages
        rtol: Relative tolerance for finite check
        atol: Absolute tolerance for finite check
    """
    nan_count = torch.isnan(output).sum().item()
    inf_count = torch.isinf(output).sum().item()

    if nan_count > 0:
        pytest.fail(f"{name}: Output contains {nan_count} NaN values")

    if inf_count > atol * output.numel():
        pytest.fail(f"{name}: Output contains {inf_count} Inf values (exceeds tolerance)")


def assert_gradient_correctness(model, name="", rtol=1e-3, atol=1e-4):
    """
    Assert that all gradients are valid.

    Args:
        model: Model with gradients
        name: Name of the test for error messages
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    for param_name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                pytest.fail(f"{name}: NaN in gradient of {param_name}")
            if not torch.isfinite(param.grad).all():
                pytest.fail(f"{name}: Inf in gradient of {param_name}")


def assert_shape_match(output, expected_shape, name=""):
    """Assert that output shape matches expected shape."""
    if output.shape != expected_shape:
        pytest.fail(f"{name}: Shape mismatch: got {output.shape}, expected {expected_shape}")


# ============================================================================
# Helper to respect QUICK_MODE in tests
# ============================================================================


def skip_if_quick_mode(reason="Skipping in QUICK_MODE"):
    """Decorator/skip marker for tests that should be skipped in quick mode."""
    return pytest.mark.skipif(QUICK_MODE, reason=reason)