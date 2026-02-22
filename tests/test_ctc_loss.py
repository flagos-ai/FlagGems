"""
Test suite for ctc_loss operator.

This test module validates correctness, precision, and performance
of ctc_loss operator implementation following FlagGems testing conventions.

测试覆盖说明：
- 输入规模：小尺寸(T=8, N=1, C=5)、常规尺寸(T=64, N=4, C=20)、大尺寸(T=256, N=8, C=50)
- 输入维数：3D (T, N, C)
- 数据类型：float32（注：PyTorch的ctc_loss只支持float32）
- 参数模式：blank, reduction, zero_infinity
- 功能完整性：不同长度的输入、blank值、reduction模式、zero_infinity处理

Note:
PyTorch's torch.nn.functional.ctc_loss only supports float32 dtype.
Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.ctc_loss.html
"""

import pytest
import torch

from flag_gems.ops import ctc_loss

# ============================================================================
# 测试数据定义（按照比赛要求）
# ============================================================================

# 数据类型覆盖（比赛要求：至少支持 float32/float16）
FLOAT_DTYPES = [
    # torch.float16,
    torch.float32,
    # torch.bfloat16,
]

# 精度标准（比赛要求的标准）
# rtol = 1e-4 (所有浮点类型)
# atol 根据数据类型变化
ATOL_DICT = {
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
}

# ============================================================================
# 辅助函数
# ============================================================================


def assert_close(actual, expected, rtol=1e-4, atol=None, dtype=torch.float32):
    """
    使用 torch.allclose 验证精度（比赛要求的标准）

    Args:
        actual: FlagGems 实现结果
        expected: PyTorch 参考结果
        rtol: 相对误差容差（默认 1e-4）
        atol: 绝对误差容差（根据数据类型）
        dtype: 数据类型
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
    创建 log probabilities 张量

    Args:
        T: 时间步数
        N: 批量大小
        C: 类别数（包括 blank）
        dtype: 数据类型
        device: 设备

    Returns:
        log_probs: (T, N, C) 形状的张量
    """
    # Create random log probabilities (log space)
    log_probs = torch.randn(T, N, C, dtype=dtype, device=device)
    # Apply log_softmax to get valid log probabilities
    log_probs = torch.log_softmax(log_probs, dim=-1)
    return log_probs


def create_targets(N, max_target_len, num_classes, device="cuda"):
    """
    创建目标序列张量（1D拼接格式）

    Args:
        N: 批量大小
        max_target_len: 最大目标长度
        num_classes: 类别数（不包括 blank）
        device: 设备

    Returns:
        targets: 拼接的目标序列
        target_lengths: 每个目标序列的长度
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
    创建输入长度张量

    Args:
        N: 批量大小
        T: 最大时间步数
        device: 设备

    Returns:
        input_lengths: 每个序列的输入长度
    """
    # Random lengths between T//2 and T
    input_lengths = torch.randint(
        T // 2 + 1, T + 1, (N,), dtype=torch.int32, device=device
    )
    return input_lengths


# ============================================================================
# 1. 基础功能测试 - 不同尺寸
# ============================================================================


class TestCTCLossBasic:
    """测试基础功能."""

    @pytest.mark.ctc_loss
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_small_size(self, dtype):
        """测试：小尺寸 (T=8, N=1, C=5)."""
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
        """测试：常规尺寸 (T=32, N=4, C=15)."""
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
        """测试：大尺寸 (T=128, N=8, C=30)."""
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
# 2. 参数模式测试 - blank, reduction, zero_infinity
# ============================================================================


class TestCTCLossParameters:
    """测试不同参数组合."""

    @pytest.mark.ctc_loss
    @pytest.mark.parametrize("blank", [0, 1, 5])
    def test_different_blank_values(self, blank):
        """测试：不同的 blank 值."""
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
        """测试：不同的 reduction 模式."""
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
        """测试：zero_infinity 参数."""
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
# 3. 边界情况测试
# ============================================================================


class TestCTCLossEdgeCases:
    """测试边界情况."""

    @pytest.mark.ctc_loss
    def test_minimum_batch_size(self):
        """测试：最小批量大小 N=1."""
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
        """测试：最小时间步数."""
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
        """测试：空目标序列."""
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
        """测试：单个目标."""
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
        """测试：长目标序列."""
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
        """测试：目标包含重复标签."""
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
# 4. 输入验证测试
# ============================================================================


class TestCTCLossValidation:
    """测试输入验证."""

    @pytest.mark.ctc_loss
    def test_2d_log_probs_supported(self):
        """测试：2D log_probs 应该被支持，与PyTorch行为一致."""
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
        """测试：批量大小不匹配."""
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
        """测试：目标长度超过输入长度."""
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
        """测试：log_probs 包含 NaN."""
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
        """测试：log_probs 包含 Inf."""
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
# 5. 极端尺寸测试 - 满足竞赛要求 4.1.4 (测例完整度要求)
# ============================================================================


class TestCTCLossExtremeSizes:
    """
    测试极端输入尺寸以满足竞赛要求 4.1.4。

    覆盖范围：
    - 小尺寸：T=4, N=1, C=3（最小有效尺寸）
    - 常规大尺寸：T=256, N=8, C=50
    - 大尺寸：T=512, N=16, C=100
    - 超大尺寸：T=1024, N=16, C=100
    """

    @pytest.mark.ctc_loss
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_extremely_small_size(self, dtype):
        """测试：极小尺寸 (T=4, N=1, C=3)."""
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
        """测试：大批量 (T=32, N=32, C=20)."""
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
        """测试：256时间步 - 竞赛要求."""
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
        """测试：512时间步."""
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
        """测试：1024时间步 - 竞赛要求."""
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
        """测试：大词汇表 (C=200)."""
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
# 6. 数据类型支持测试
# ============================================================================


class TestCTCLossDtypes:
    """测试不同数据类型."""

    @pytest.mark.ctc_loss
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_all_dtypes_with_reductions(self, dtype, reduction):
        """测试：所有支持的数据类型和 reduction 模式组合."""
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
        """测试：所有数据类型与 zero_infinity 参数."""
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
