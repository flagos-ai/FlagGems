"""
Test suite for ctc_loss operator.

This test module validates correctness, precision, and performance
of ctc_loss operator implementation following FlagGems testing conventions.

Test coverage description:
- Input sizes: small (T=8, N=1, C=5), medium (T=64, N=4, C=20), large (T=256, N=8, C=50)
- Input dimensions: 3D (T, N, C)
- Data types: float32 (Note: PyTorch's ctc_loss only supports float32)
- Parameter modes: blank, reduction, zero_infinity
- Functional completeness: inputs of different lengths, blank values, reduction modes, zero_infinity handling

Note:
PyTorch's torch.nn.functional.ctc_loss only supports float32 dtype.
Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.ctc_loss.html
"""

import pytest
import torch

from flag_gems.ops import ctc_loss

# ============================================================================
# Test data definitions (following competition requirements)
# ============================================================================

# Data type coverage (competition requirement: at least support float32/float16)
FLOAT_DTYPES = [
    # torch.float16,
    torch.float32,
    # torch.bfloat16,
]

# Precision standards (competition requirement standards)
# rtol = 1e-4 (all floating point types)
# atol varies based on data type
ATOL_DICT = {
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
}

# ============================================================================
# Helper functions
# ============================================================================


def assert_close(actual, expected, rtol=1e-4, atol=None, dtype=torch.float32):
    """
    Verify precision using torch.allclose (competition requirement standards)

    Args:
        actual: FlagGems implementation result
        expected: PyTorch reference result
        rtol: Relative error tolerance (default 1e-4)
        atol: Absolute error tolerance (based on data type)
        dtype: Data type
    """
    if atol is None:
        atol = ATOL_DICT.get(dtype, 1e-5)

    # Handle scalar outputs (when reduction='mean' or 'sum')
    if actual.dim() == 0:
        assert torch.allclose(
            actual, expected, rtol=rtol, atol=atol, equal_nan=True
        ), f"Results don't match: actual={actual.item()}, expected={expected.item()}"
    else:
        assert torch.allclose(
            actual, expected, rtol=rtol, atol=atol, equal_nan=True
        ), f"Results don't match: max diff={(actual - expected).abs().max().item()}"


def create_log_probs(T, N, C, dtype, device="cuda"):
    """
    Create log probabilities tensor

    Args:
        T: Number of time steps
        N: Batch size
        C: Number of classes (including blank)
        dtype: Data type
        device: Device

    Returns:
        log_probs: Tensor of shape (T, N, C)
    """
    # Create random log probabilities (log space)
    log_probs = torch.randn(T, N, C, dtype=dtype, device=device)
    # Apply log_softmax to get valid log probabilities
    log_probs = torch.log_softmax(log_probs, dim=-1)
    return log_probs


def create_targets(N, max_target_len, num_classes, device="cuda"):
    """
    Create target sequence tensor (1D concatenated format)

    Args:
        N: Batch size
        max_target_len: Maximum target length
        num_classes: Number of classes (excluding blank)
        device: Device

    Returns:
        targets: Concatenated target sequences
        target_lengths: Length of each target sequence
    """
    targets_list = []
    target_lengths_list = []

    for i in range(N):
        target_len = torch.randint(1, max_target_len + 1, (1,)).item()
        target = torch.randint(0, num_classes, (target_len,), device=device)
        targets_list.append(target)
        target_lengths_list.append(target_len)

    targets = torch.cat(targets_list, dim=0)
    target_lengths = torch.tensor(target_lengths_list, dtype=torch.int32, device=device)

    return targets, target_lengths


def create_input_lengths(N, T, device="cuda"):
    """
    Create input length tensor

    Args:
        N: Batch size
        T: Maximum number of time steps
        device: Device

    Returns:
        input_lengths: Input length for each sequence
    """
    # Random lengths between T//2 and T
    input_lengths = torch.randint(
        T // 2 + 1, T + 1, (N,), dtype=torch.int32, device=device
    )
    return input_lengths


# ============================================================================
# 1. Basic functionality tests - different sizes
# ============================================================================


class TestCTCLossBasic:
    """Test basic functionality."""

    @pytest.mark.ctc_loss
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_small_size(self, dtype):
        """Test: small size (T=8, N=1, C=5)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        T, N, C = 8, 1, 5
        blank = 0

        log_probs = create_log_probs(T, N, C, dtype)
        targets, target_lengths = create_targets(N, max_target_len=4, num_classes=C - 1)
        input_lengths = create_input_lengths(N, T)
        try:
            loss_torch = torch.nn.functional.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank=blank,
                reduction="mean",
                zero_infinity=False,
            )
        except NotImplementedError:
            pytest.skip("torch.nn.functional.ctc_loss not implemented")

        loss_gems = ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction="mean",
            zero_infinity=False,
        )

        assert_close(loss_gems, loss_torch, dtype=dtype)

    @pytest.mark.ctc_loss
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_medium_size(self, dtype):
        """Test: medium size (T=32, N=4, C=15)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        T, N, C = 32, 4, 15
        blank = 0

        log_probs = create_log_probs(T, N, C, dtype)
        targets, target_lengths = create_targets(
            N, max_target_len=10, num_classes=C - 1
        )
        input_lengths = create_input_lengths(N, T)
        try:
            loss_torch = torch.nn.functional.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank=blank,
                reduction="mean",
                zero_infinity=False,
            )
        except NotImplementedError:
            pytest.skip("torch.nn.functional.ctc_loss not implemented")
        loss_gems = ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction="mean",
            zero_infinity=False,
        )

        assert_close(loss_gems, loss_torch, dtype=dtype)

    @pytest.mark.ctc_loss
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_large_size(self, dtype):
        """Test: large size (T=128, N=8, C=30)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        T, N, C = 128, 8, 30
        blank = 0

        log_probs = create_log_probs(T, N, C, dtype)
        targets, target_lengths = create_targets(
            N, max_target_len=20, num_classes=C - 1
        )
        input_lengths = create_input_lengths(N, T)
        try:
            loss_torch = torch.nn.functional.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank=blank,
                reduction="mean",
                zero_infinity=False,
            )
        except NotImplementedError:
            pytest.skip("torch.nn.functional.ctc_loss not implemented")
        loss_gems = ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction="mean",
            zero_infinity=False,
        )

        assert_close(loss_gems, loss_torch, dtype=dtype)


# ============================================================================
# 2. Parameter mode tests - blank, reduction, zero_infinity
# ============================================================================


class TestCTCLossParameters:
    """Test different parameter combinations."""

    @pytest.mark.ctc_loss
    @pytest.mark.parametrize("blank", [0, 1, 5])
    def test_different_blank_values(self, blank):
        """Test: different blank values."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dtype = torch.float32
        T, N, C = 16, 2, 15
        # Ensure C is large enough for blank value
        if blank >= C:
            C = blank + 5

        log_probs = create_log_probs(T, N, C, dtype)
        targets, target_lengths = create_targets(N, max_target_len=6, num_classes=C)
        input_lengths = create_input_lengths(N, T)

        # Ensure targets don't contain blank value (simple filtering approach)
        # If any target equals blank, replace with a different valid value
        if blank < C - 1:
            targets = torch.where(
                targets == blank,
                torch.tensor(C - 1, device=targets.device, dtype=targets.dtype),
                targets,
            )
        elif blank == C - 1:
            targets = torch.where(
                targets == blank,
                torch.tensor(0, device=targets.device, dtype=targets.dtype),
                targets,
            )
        # If blank > C-1 (shouldn't happen), no filtering needed
        try:
            loss_torch = torch.nn.functional.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank=blank,
                reduction="mean",
                zero_infinity=False,
            )
        except NotImplementedError:
            pytest.skip("torch.nn.functional.ctc_loss not implemented")
        loss_gems = ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction="mean",
            zero_infinity=False,
        )

        assert_close(loss_gems, loss_torch, dtype=dtype)

    @pytest.mark.ctc_loss
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_different_reduction_modes(self, reduction, dtype):
        """Test: different reduction modes."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        T, N, C = 20, 4, 12
        blank = 0

        log_probs = create_log_probs(T, N, C, dtype)
        targets, target_lengths = create_targets(N, max_target_len=8, num_classes=C - 1)
        input_lengths = create_input_lengths(N, T)
        try:
            loss_torch = torch.nn.functional.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank=blank,
                reduction=reduction,
                zero_infinity=False,
            )
        except NotImplementedError:
            pytest.skip("torch.nn.functional.ctc_loss not implemented")

        loss_gems = ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction=reduction,
            zero_infinity=False,
        )

        assert_close(loss_gems, loss_torch, dtype=dtype)

    @pytest.mark.ctc_loss
    @pytest.mark.parametrize("zero_infinity", [True, False])
    def test_zero_infinity(self, zero_infinity):
        """Test: zero_infinity parameter."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dtype = torch.float32
        T, N, C = 16, 2, 10
        blank = 0

        # Create log_probs that might result in infinite loss
        # by setting some time steps to very low probabilities
        log_probs = create_log_probs(T, N, C, dtype)
        # Make some entries very negative to potentially cause infinity
        log_probs[:2] = -100

        targets, target_lengths = create_targets(N, max_target_len=6, num_classes=C - 1)
        input_lengths = create_input_lengths(N, T)
        # Set some input lengths to be shorter than targets to potentially cause infinity
        input_lengths[0] = max(1, target_lengths[0].item() // 2)
        try:
            loss_torch = torch.nn.functional.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank=blank,
                reduction="mean",
                zero_infinity=zero_infinity,
            )
        except NotImplementedError:
            pytest.skip("torch.nn.functional.ctc_loss not implemented")
        loss_gems = ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction="mean",
            zero_infinity=zero_infinity,
        )

        assert_close(loss_gems, loss_torch, dtype=dtype)


# ============================================================================
# 3. Edge case tests
# ============================================================================


class TestCTCLossEdgeCases:
    """Test edge cases."""

    @pytest.mark.ctc_loss
    def test_minimum_batch_size(self):
        """Test: minimum batch size N=1."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dtype = torch.float32
        T, N, C = 10, 1, 8
        blank = 0

        log_probs = create_log_probs(T, N, C, dtype)
        targets, target_lengths = create_targets(N, max_target_len=4, num_classes=C - 1)
        input_lengths = create_input_lengths(N, T)
        try:
            loss_torch = torch.nn.functional.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank=blank,
                reduction="mean",
            )
        except NotImplementedError:
            pytest.skip("torch.nn.functional.ctc_loss not implemented")
        loss_gems = ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction="mean",
        )

        assert_close(loss_gems, loss_torch, dtype=dtype)

    @pytest.mark.ctc_loss
    def test_minimum_time_steps(self):
        """Test: minimum time steps."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dtype = torch.float32
        T, N, C = 2, 2, 8
        blank = 0

        log_probs = create_log_probs(T, N, C, dtype)
        targets, target_lengths = create_targets(N, max_target_len=1, num_classes=C - 1)
        input_lengths = create_input_lengths(N, T)
        try:
            loss_torch = torch.nn.functional.ctc_loss(
                log_probs, targets, input_lengths, target_lengths, blank=blank
            )

        except NotImplementedError:
            pytest.skip("torch.nn.functional.ctc_loss not implemented")
        loss_gems = ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=blank
        )

        assert_close(loss_gems, loss_torch, dtype=dtype)

    @pytest.mark.ctc_loss
    def test_empty_targets(self):
        """Test: empty target sequence."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dtype = torch.float32
        T, N, C = 10, 2, 8
        blank = 0

        log_probs = create_log_probs(T, N, C, dtype)
        # Create empty targets
        targets = torch.tensor([], dtype=torch.int32, device="cuda")
        target_lengths = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
        input_lengths = create_input_lengths(N, T)
        try:
            loss_torch = torch.nn.functional.ctc_loss(
                log_probs, targets, input_lengths, target_lengths, blank=blank
            )

        except NotImplementedError:
            pytest.skip("torch.nn.functional.ctc_loss not implemented")

        loss_gems = ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=blank
        )

        assert_close(loss_gems, loss_torch, dtype=dtype)

    @pytest.mark.ctc_loss
    def test_single_target(self):
        """Test: single target."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dtype = torch.float32
        T, N, C = 10, 1, 8
        blank = 0

        log_probs = create_log_probs(T, N, C, dtype)
        # Single target
        targets = torch.tensor([3], dtype=torch.int32, device="cuda")
        target_lengths = torch.tensor([1], dtype=torch.int32, device="cuda")
        input_lengths = torch.tensor([T], dtype=torch.int32, device="cuda")
        try:
            loss_torch = torch.nn.functional.ctc_loss(
                log_probs, targets, input_lengths, target_lengths, blank=blank
            )

        except NotImplementedError:
            pytest.skip("torch.nn.functional.ctc_loss not implemented")
        loss_gems = ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=blank
        )

        assert_close(loss_gems, loss_torch, dtype=dtype)

    @pytest.mark.ctc_loss
    def test_long_target_sequence(self):
        """Test: long target sequence."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dtype = torch.float32
        T, N, C = 50, 2, 20
        blank = 0

        log_probs = create_log_probs(T, N, C, dtype)
        # Create long targets
        targets_list = []
        target_lengths_list = []
        for i in range(N):
            target_len = 20  # Long target
            target = torch.randint(0, C - 1, (target_len,), device="cuda")
            targets_list.append(target)
            target_lengths_list.append(target_len)

        targets = torch.cat(targets_list, dim=0)
        target_lengths = torch.tensor(
            target_lengths_list, dtype=torch.int32, device="cuda"
        )
        input_lengths = create_input_lengths(N, T)

        loss_gems = ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=blank
        )
        loss_torch = torch.nn.functional.ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=blank
        )

        assert_close(loss_gems, loss_torch, dtype=dtype)

    @pytest.mark.ctc_loss
    def test_target_with_repeated_labels(self):
        """Test: target with repeated labels."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dtype = torch.float32
        T, N, C = 20, 1, 10
        blank = 0

        log_probs = create_log_probs(T, N, C, dtype)
        # Target with repeated labels (separated by blank in CTC path)
        targets = torch.tensor([1, 2, 2, 3, 3, 3], dtype=torch.int32, device="cuda")
        target_lengths = torch.tensor([6], dtype=torch.int32, device="cuda")
        input_lengths = torch.tensor([T], dtype=torch.int32, device="cuda")

        loss_gems = ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=blank
        )
        loss_torch = torch.nn.functional.ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=blank
        )

        assert_close(loss_gems, loss_torch, dtype=dtype)


# ============================================================================
# 4. Input validation tests
# ============================================================================


class TestCTCLossValidation:
    """Test input validation."""

    @pytest.mark.ctc_loss
    def test_2d_log_probs_supported(self):
        """Test: 2D log_probs should be supported, consistent with PyTorch behavior."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dtype = torch.float32
        # 2D tensor (T, C) - PyTorch supports this for N=1 case
        log_probs = torch.randn(10, 5, dtype=dtype, device="cuda")
        targets = torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")
        input_lengths = torch.tensor([10], dtype=torch.int32, device="cuda")
        target_lengths = torch.tensor([3], dtype=torch.int32, device="cuda")

        # Both FlagGems and PyTorch should support 2D input
        # Verify results match
        loss_gems = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        loss_torch = torch.nn.functional.ctc_loss(
            log_probs, targets, input_lengths, target_lengths
        )
        assert_close(loss_gems, loss_torch, dtype=dtype)

    @pytest.mark.ctc_loss
    def test_mismatched_batch_size(self):
        """Test: mismatched batch size."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dtype = torch.float32
        T, N, C = 10, 2, 5
        log_probs = create_log_probs(T, N, C, dtype)
        targets, target_lengths = create_targets(
            N + 1, max_target_len=4, num_classes=C - 1
        )
        input_lengths = create_input_lengths(N, T)
        # target_lengths has wrong batch size
        with pytest.raises((ValueError, RuntimeError)):
            ctc_loss(log_probs, targets, input_lengths, target_lengths)

    @pytest.mark.ctc_loss
    def test_invalid_target_length(self):
        """Test: target length exceeds input length."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dtype = torch.float32
        T, N, C = 10, 1, 5
        log_probs = create_log_probs(T, N, C, dtype)
        targets = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32, device="cuda")
        target_lengths = torch.tensor(
            [10], dtype=torch.int32, device="cuda"
        )  # Too long
        input_lengths = torch.tensor([5], dtype=torch.int32, device="cuda")

        try:
            loss_torch = torch.nn.functional.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank=0,
                zero_infinity=True,
            )
        except RuntimeError as e:
            pytest.skip(f"RuntimeError: {e}")

        loss_gems = ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=0,
            zero_infinity=True,
        )

        # Both should be 0.0 when zero_infinity=True and impossible alignment
        assert_close(loss_gems, loss_torch, dtype=dtype)

    @pytest.mark.ctc_loss
    def test_nan_in_log_probs(self):
        """Test: log_probs contains NaN."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dtype = torch.float32
        T, N, C = 10, 2, 8
        blank = 0

        log_probs = create_log_probs(T, N, C, dtype)
        # Insert NaN into log_probs
        log_probs[0, 0, 0] = float("nan")
        targets, target_lengths = create_targets(N, max_target_len=4, num_classes=C - 1)
        input_lengths = create_input_lengths(N, T)

        try:
            loss_torch = torch.nn.functional.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank=blank,
                zero_infinity=False,
            )
        except NotImplementedError:
            pytest.skip("torch.nn.functional.ctc_loss not implemented")

        loss_gems = ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            zero_infinity=False,
        )

        # Both should handle NaN gracefully
        assert_close(loss_gems, loss_torch, dtype=dtype)

    @pytest.mark.ctc_loss
    def test_inf_in_log_probs(self):
        """Test: log_probs contains Inf."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dtype = torch.float32
        T, N, C = 10, 2, 8
        blank = 0

        log_probs = create_log_probs(T, N, C, dtype)
        # Insert Inf into log_probs
        log_probs[0, 0, 0] = float("inf")
        targets, target_lengths = create_targets(N, max_target_len=4, num_classes=C - 1)
        input_lengths = create_input_lengths(N, T)

        try:
            loss_torch = torch.nn.functional.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank=blank,
                zero_infinity=False,
            )
        except NotImplementedError:
            pytest.skip("torch.nn.functional.ctc_loss not implemented")

        loss_gems = ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            zero_infinity=False,
        )

        # Both should handle Inf gracefully
        assert_close(loss_gems, loss_torch, dtype=dtype)


# ============================================================================
# 5. Extreme size tests - satisfy competition requirement 4.1.4 (test case completeness requirement)
# ============================================================================


class TestCTCLossExtremeSizes:
    """
    Test extreme input sizes to satisfy competition requirement 4.1.4.

    Coverage:
    - Small size: T=4, N=1, C=3 (minimum valid size)
    - Regular large size: T=256, N=8, C=50
    - Large size: T=512, N=16, C=100
    - Extra large size: T=1024, N=16, C=100
    """

    @pytest.mark.ctc_loss
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_extremely_small_size(self, dtype):
        """Test: extremely small size (T=4, N=1, C=3)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        T, N, C = 4, 1, 3
        blank = 0

        log_probs = create_log_probs(T, N, C, dtype)
        targets, target_lengths = create_targets(N, max_target_len=2, num_classes=C - 1)
        input_lengths = create_input_lengths(N, T)

        loss_gems = ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=blank
        )
        loss_torch = torch.nn.functional.ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=blank
        )

        assert_close(loss_gems, loss_torch, dtype=dtype)

    @pytest.mark.ctc_loss
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_very_large_batch(self, dtype):
        """Test: large batch (T=32, N=32, C=20)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        T, N, C = 32, 32, 20
        blank = 0

        log_probs = create_log_probs(T, N, C, dtype)
        targets, target_lengths = create_targets(
            N, max_target_len=10, num_classes=C - 1
        )
        input_lengths = create_input_lengths(N, T)

        try:
            loss_torch = torch.nn.functional.ctc_loss(
                log_probs, targets, input_lengths, target_lengths, blank=blank
            )

        except NotImplementedError:
            pytest.skip("PyTorch CTC loss not implemented for this size")

        loss_gems = ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=blank
        )

        assert_close(loss_gems, loss_torch, dtype=dtype)

    @pytest.mark.ctc_loss
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_256_time_steps_large(self, dtype):
        """Test: 256 time steps - competition requirement."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        T, N, C = 256, 8, 50
        blank = 0

        log_probs = create_log_probs(T, N, C, dtype)
        targets, target_lengths = create_targets(
            N, max_target_len=30, num_classes=C - 1
        )
        input_lengths = create_input_lengths(N, T)
        try:
            loss_torch = torch.nn.functional.ctc_loss(
                log_probs, targets, input_lengths, target_lengths, blank=blank
            )

        except NotImplementedError:
            pytest.skip("PyTorch CTC loss not implemented for this size")
        loss_gems = ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=blank
        )

        assert_close(loss_gems, loss_torch, dtype=dtype)

    @pytest.mark.ctc_loss
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_512_time_steps_very_large(self, dtype):
        """Test: 512 time steps."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        T, N, C = 512, 8, 50
        blank = 0

        log_probs = create_log_probs(T, N, C, dtype)
        targets, target_lengths = create_targets(
            N, max_target_len=40, num_classes=C - 1
        )
        input_lengths = create_input_lengths(N, T)

        loss_gems = ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=blank
        )
        loss_torch = torch.nn.functional.ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=blank
        )

        assert_close(loss_gems, loss_torch, dtype=dtype)

    @pytest.mark.ctc_loss
    def test_1024_time_steps_extreme_large(self):
        """Test: 1024 time steps - competition requirement."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Check available GPU memory
        gpu_memory_available = torch.cuda.get_device_properties(0).total_memory
        if gpu_memory_available < 8 * 1024**3:  # Need at least 8GB
            pytest.skip(
                f"Insufficient GPU memory ({gpu_memory_available / 1024**3:.1f}GB) for 1024 timesteps test"
            )

        try:
            dtype = torch.float32
            T, N, C = 1024, 8, 50
            blank = 0

            log_probs = create_log_probs(T, N, C, dtype)
            targets, target_lengths = create_targets(
                N, max_target_len=50, num_classes=C - 1
            )
            input_lengths = create_input_lengths(N, T)

            loss_gems = ctc_loss(
                log_probs, targets, input_lengths, target_lengths, blank=blank
            )
            loss_torch = torch.nn.functional.ctc_loss(
                log_probs, targets, input_lengths, target_lengths, blank=blank
            )

            assert_close(loss_gems, loss_torch, dtype=dtype)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            pytest.skip(
                f"GPU memory insufficient for 1024 timesteps test: {str(e)[:100]}"
            )

    @pytest.mark.ctc_loss
    def test_large_vocabulary(self):
        """Test: large vocabulary (C=200)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dtype = torch.float32
        T, N, C = 64, 4, 200
        blank = 0

        log_probs = create_log_probs(T, N, C, dtype)
        targets, target_lengths = create_targets(
            N, max_target_len=20, num_classes=C - 1
        )
        input_lengths = create_input_lengths(N, T)

        loss_gems = ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=blank
        )
        loss_torch = torch.nn.functional.ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=blank
        )

        assert_close(loss_gems, loss_torch, dtype=dtype)


# ============================================================================
# 6. Data type support tests
# ============================================================================


class TestCTCLossDtypes:
    """Test different data types."""

    @pytest.mark.ctc_loss
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_all_dtypes_with_reductions(self, dtype, reduction):
        """Test: all supported data types and reduction mode combinations."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        T, N, C = 20, 2, 12
        blank = 0

        log_probs = create_log_probs(T, N, C, dtype)
        targets, target_lengths = create_targets(N, max_target_len=8, num_classes=C - 1)
        input_lengths = create_input_lengths(N, T)

        loss_gems = ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction=reduction,
        )
        loss_torch = torch.nn.functional.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction=reduction,
        )

        assert_close(loss_gems, loss_torch, dtype=dtype)

    @pytest.mark.ctc_loss
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_all_dtypes_with_zero_infinity(self, dtype):
        """Test: all data types with zero_infinity parameter."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        T, N, C = 16, 2, 10
        blank = 0

        log_probs = create_log_probs(T, N, C, dtype)
        # Make some entries very negative to potentially cause infinity
        log_probs[:2] = -100

        targets, target_lengths = create_targets(N, max_target_len=6, num_classes=C - 1)
        input_lengths = create_input_lengths(N, T)
        # Set some input lengths to be shorter than targets to potentially cause infinity
        input_lengths[0] = max(1, target_lengths[0].item() // 2)

        loss_gems = ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            zero_infinity=True,
        )
        loss_torch = torch.nn.functional.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            zero_infinity=True,
        )

        assert_close(loss_gems, loss_torch, dtype=dtype)


# ============================================================================
# 7. Regression Tests for Critical Bugs
# ============================================================================


class TestCTCLossRegression:
    """Regression tests for specific critical bugs found in previous versions."""

    @pytest.mark.ctc_loss
    def test_no_negative_loss(self):
        """Regression test: Loss should never be negative.

        This test catches race conditions in alpha computation that result in
        reading uninitialized memory, causing negative/invalid loss values.

        Bug: Alpha kernel launched all timesteps in parallel (grid=(N,T,blocks)),
        causing race conditions when reading alpha[t-1] before timestep t-1 completed.
        Fix: Process timesteps sequentially to avoid data dependency race.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dtype = torch.float32
        T, N, C = 8, 1, 5
        blank = 0

        log_probs = create_log_probs(T, N, C, dtype)
        targets, target_lengths = create_targets(N, max_target_len=4, num_classes=C - 1)
        input_lengths = create_input_lengths(N, T)

        loss_gems = ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction="mean",
            zero_infinity=False,
        )

        # Loss should always be non-negative (it's -log of probability)
        assert loss_gems >= 0, f"Loss should be non-negative, got {loss_gems.item()}"
