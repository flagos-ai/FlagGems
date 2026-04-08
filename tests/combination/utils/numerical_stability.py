"""
Numerical stability utilities for combination testing.

This module provides utilities for testing numerical stability
in operator combinations.
"""

import torch


def check_no_nan(tensor, name=""):
    """
    Check that tensor contains no NaN values.

    Args:
        tensor: Tensor to check
        name: Name for error message

    Raises:
        AssertionError: If NaN values found
    """
    nan_count = torch.isnan(tensor).sum().item()
    if nan_count > 0:
        raise AssertionError(f"{name}: Found {nan_count} NaN values in tensor")


def check_no_inf(tensor, name="", tolerance=0.01):
    """
    Check that tensor contains reasonable number of Inf values.

    Args:
        tensor: Tensor to check
        name: Name for error message
        tolerance: Fraction of Inf values allowed (default 1%)

    Raises:
        AssertionError: If too many Inf values found
    """
    inf_count = torch.isinf(tensor).sum().item()
    total = tensor.numel()
    inf_fraction = inf_count / total if total > 0 else 0

    if inf_fraction > tolerance:
        raise AssertionError(
            f"{name}: Found {inf_count} Inf values ({inf_fraction:.2%} of tensor), exceeds tolerance {tolerance:.2%}"
        )


def check_finite(tensor, name=""):
    """
    Check that all values are finite (no NaN or Inf).

    Args:
        tensor: Tensor to check
        name: Name for error message

    Raises:
        AssertionError: If non-finite values found
    """
    if not torch.isfinite(tensor).all():
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        raise AssertionError(
            f"{name}: Tensor contains {nan_count} NaN and {inf_count} Inf values"
        )


def check_gradient_health(model, name=""):
    """
    Check that all model gradients are finite.

    Args:
        model: Model to check
        name: Name for error message

    Raises:
        AssertionError: If gradient issues found
    """
    for param_name, param in model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                nan_count = torch.isnan(param.grad).sum().item()
                inf_count = torch.isinf(param.grad).sum().item()
                raise AssertionError(
                    f"{name}: Gradient of {param_name} has {nan_count} NaN and {inf_count} Inf values"
                )


def assert_relative_error(actual, expected, rtol=1e-3, atol=1e-5, name=""):
    """
    Assert that actual and expected tensors are close within tolerance.

    Args:
        actual: Actual tensor
        expected: Expected tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        name: Name for error message

    Raises:
        AssertionError: If tensors are not close enough
    """
    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        max_error = (actual - expected).abs().max().item()
        mean_error = (actual - expected).abs().mean().item()
        raise AssertionError(
            f"{name}: Tensors differ - max error: {max_error:.6e}, mean error: {mean_error:.6e}"
        )


def generate_stress_input(shape, dtype, device, stress_type="extreme"):
    """
    Generate input data for stress testing.

    Args:
        shape: Tensor shape
        dtype: Data type
        device: Device
        stress_type: Type of stress ('extreme', 'small', 'mixed')

    Returns:
        Generated tensor
    """
    if stress_type == "extreme":
        # Large magnitude values
        return torch.randn(shape, dtype=dtype, device=device) * 100

    elif stress_type == "small":
        # Small magnitude values
        return torch.randn(shape, dtype=dtype, device=device) * 1e-6

    elif stress_type == "mixed":
        # Mix of normal, large, and small values
        x = torch.randn(shape, dtype=dtype, device=device)
        # Make some values very large
        large_mask = torch.rand(shape, device=device) < 0.1
        x[large_mask] *= 100
        # Make some values very small
        small_mask = torch.rand(shape, device=device) < 0.1
        x[small_mask] *= 1e-4
        return x

    else:
        raise ValueError(f"Unknown stress type: {stress_type}")


class NumericalMonitor:
    """
    Context manager for monitoring numerical stability during operations.
    """

    def __init__(self, name="", check_interval=1):
        """
        Initialize monitor.

        Args:
            name: Name for logging
            check_interval: How often to check (every N operations)
        """
        self.name = name
        self.check_interval = check_interval
        self.operation_count = 0
        self.issues = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.issues:
            print(f"\n[NumericalMonitor] {self.name}: Found {len(self.issues)} issues:")
            for issue in self.issues[:10]:  # Show first 10 issues
                print(f"  - {issue}")
        return False

    def check(self, tensor, operation_name=""):
        """
        Check a tensor for numerical issues.

        Args:
            tensor: Tensor to check
            operation_name: Name of operation that produced tensor
        """
        self.operation_count += 1

        if self.operation_count % self.check_interval != 0:
            return

        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()

        if nan_count > 0:
            self.issues.append(f"{operation_name}: {nan_count} NaN values")

        if inf_count > 0:
            self.issues.append(f"{operation_name}: {inf_count} Inf values")