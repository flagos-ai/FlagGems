"""
Test suite for max_pool3d operator.

This test module validates the correctness, precision, and performance
of the max_pool3d operator implementation following FlagGems testing conventions.

测试覆盖说明：
- 输入规模：小尺寸（4×4×4）、常规尺寸（32×32×32）、大尺寸（128×128×128）
- 输入维数：4D (C,D,H,W)、5D (N,C,D,H,W)
- 数据类型：float16, float32, bfloat16
- 参数模式：kernel_size, stride, padding, dilation, ceil_mode, return_indices
- 功能完整性：基本池化、索引返回、4D/5D输入、批处理
"""

import pytest
import torch

from flag_gems.ops import max_pool3d

# ============================================================================
# 测试数据定义（按照比赛要求）
# ============================================================================

# 输入规模覆盖（比赛要求：小尺寸、常规尺寸、大尺寸）
POOL3D_SHAPES = [
    # 小尺寸
    (1, 1, 4, 4, 4),  # N=1, C=1, D=4, H=4, W=4
    (2, 3, 8, 8, 8),  # N=2, C=3, D=8, H=8, W=8
    # 常规尺寸
    (2, 16, 32, 32, 32),  # N=2, C=16, D=32, H=32, W=32
    (4, 64, 64, 64, 64),  # N=4, C=64, D=64, H=64, W=64
    # 大尺寸
    (2, 128, 128, 128, 128),  # N=2, C=128, D=128, H=128, W=128
    (4, 256, 256, 256, 256),  # N=4, C=256, D=256, H=256, W=256
]

# 数据类型覆盖（比赛要求：至少支持 float32/float16）
FLOAT_DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
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

    # 使用 torch.allclose 进行比较（比赛标准）
    assert torch.allclose(
        actual, expected, rtol=rtol, atol=atol, equal_nan=True
    ), f"Results don't match: max diff={(actual - expected).abs().max().item()}"


def create_tensor(shape, dtype, device="cuda"):
    """创建测试张量"""
    x = torch.randn(shape, dtype=dtype, device=device)
    return x


# ============================================================================
# 1. 输入规模覆盖测试（比赛要求：小、常规、大三类）
# ============================================================================


class TestMaxPool3DInputSize:
    """测试输入规模覆盖"""

    @pytest.mark.max_pool3d
    def test_size_very_small(self):
        """测试：极小尺寸 (1, 1, 4, 4, 4)"""
        x = create_tensor((1, 1, 4, 4, 4), torch.float32)
        y_gems = max_pool3d(x, kernel_size=2)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_size_small(self):
        """测试：小尺寸 (2, 3, 8, 8, 8)"""
        x = create_tensor((2, 3, 8, 8, 8), torch.float32)
        y_gems = max_pool3d(x, kernel_size=2)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_size_medium_32(self):
        """测试：常规尺寸 (2, 16, 32, 32, 32)"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=2)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_size_medium_64(self):
        """测试：常规尺寸 (4, 64, 64, 64, 64)"""
        x = create_tensor((4, 64, 64, 64, 64), torch.float32)
        y_gems = max_pool3d(x, kernel_size=3)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=3)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_size_large_128(self):
        """测试：大尺寸 (2, 128, 128, 128, 128)"""
        x = create_tensor((2, 128, 128, 128, 128), torch.float32)
        y_gems = max_pool3d(x, kernel_size=2)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_size_large_256(self):
        """测试：大尺寸 (2, 128, 64, 64, 64)"""
        x = create_tensor((2, 128, 64, 64, 64), torch.float32)
        y_gems = max_pool3d(x, kernel_size=3)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=3)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 2. 输入维数覆盖测试（比赛要求：覆盖全部合法维数）
# ============================================================================


class TestMaxPool3DInputDimensions:
    """测试输入维数覆盖：4D (C,D,H,W), 5D (N,C,D,H,W)"""

    @pytest.mark.max_pool3d
    def test_dim_4d_small(self):
        """测试：4D 张量 - 小尺寸 (C, D, H, W)"""
        x = create_tensor((1, 4, 4, 4), torch.float32)
        y_gems = max_pool3d(x, kernel_size=2)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)
        # 验证输出也是 4D
        assert y_gems.dim() == 4

    @pytest.mark.max_pool3d
    def test_dim_4d_medium(self):
        """测试：4D 张量 - 中等尺寸 (C, D, H, W)"""
        x = create_tensor((16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=2)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)
        # 验证输出也是 4D
        assert y_gems.dim() == 4

    @pytest.mark.max_pool3d
    def test_dim_5d_small(self):
        """测试：5D 张量 - 小尺寸 (N, C, D, H, W)"""
        x = create_tensor((2, 3, 8, 8, 8), torch.float32)
        y_gems = max_pool3d(x, kernel_size=2)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)
        # 验证输出也是 5D
        assert y_gems.dim() == 5

    @pytest.mark.max_pool3d
    def test_dim_5d_large(self):
        """测试：5D 张量 - 大尺寸 (N, C, D, H, W)"""
        x = create_tensor((4, 64, 64, 64, 64), torch.float32)
        y_gems = max_pool3d(x, kernel_size=3)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=3)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)
        # 验证输出也是 5D
        assert y_gems.dim() == 5


# ============================================================================
# 3. 数据类型覆盖测试（比赛要求：至少支持 float32/float16）
# ============================================================================


class TestMaxPool3DDataTypes:
    """测试数据类型覆盖：float16, float32, bfloat16"""

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_all_float_dtypes(self, dtype):
        """测试：所有浮点数据类型"""
        x = create_tensor((2, 16, 32, 32, 32), dtype)
        y_gems = max_pool3d(x, kernel_size=2)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)


# ============================================================================
# 4. 参数模式覆盖测试（比赛要求：默认值、边界值、特殊值）
# ============================================================================


class TestMaxPool3DParameterPatterns:
    """测试参数模式：kernel_size, stride, padding, dilation, ceil_mode"""

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize("kernel_size", [1, 2, 3, 4])
    def test_various_kernel_sizes(self, kernel_size):
        """测试：各种 kernel_size 值"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=kernel_size)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=kernel_size)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize("stride", [1, 2, 3])
    def test_various_strides(self, stride):
        """测试：各种 stride 值"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=3, stride=stride)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=3, stride=stride)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize("padding", [0, 1, 3])
    def test_various_padding(self, padding):
        """测试：各种 padding 值"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=7, padding=padding)
        # y_gems = torch.nn.functional.max_pool3d(x, kernel_size=7, padding=padding)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=7, padding=padding)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize("dilation", [1, 2])
    def test_various_dilation(self, dilation):
        """测试：各种 dilation 值"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=3, dilation=dilation)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=3, dilation=dilation)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize("ceil_mode", [False, True])
    def test_ceil_mode(self, ceil_mode):
        """测试：ceil_mode 参数"""
        x = create_tensor((2, 16, 31, 31, 31), torch.float32)
        y_gems = max_pool3d(x, kernel_size=3, stride=2, ceil_mode=ceil_mode)
        y_torch = torch.nn.functional.max_pool3d(
            x, kernel_size=3, stride=2, ceil_mode=ceil_mode
        )
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_tuple_kernel_size(self):
        """测试：tuple kernel_size 参数"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=(2, 3, 4))
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=(2, 3, 4))
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_tuple_stride(self):
        """测试：tuple stride 参数"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=3, stride=(1, 2, 1))
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=3, stride=(1, 2, 1))
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_tuple_padding(self):
        """测试：tuple padding 参数"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=3, padding=(0, 1, 0))
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=3, padding=(0, 1, 0))
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_tuple_dilation(self):
        """测试：tuple dilation 参数"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=3, dilation=(1, 2, 1))
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=3, dilation=(1, 2, 1))
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_default_stride(self):
        """测试：默认 stride（等于 kernel_size）"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=3)  # stride 默认为 kernel_size
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=3)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 5. 功能完整性测试（比赛要求：所有功能分支）
# ============================================================================


class TestMaxPool3DFunctionalCompleteness:
    """测试功能完整性：基本池化、索引返回、4D/5D输入、批处理"""

    @pytest.mark.max_pool3d
    def test_basic_pooling(self):
        """测试：基本 3D 池化操作"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=2)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_return_indices_false(self):
        """测试：return_indices=False（只返回输出）"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=2, return_indices=False)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2, return_indices=False)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)
        # 验证返回的是单个张量
        assert isinstance(y_gems, torch.Tensor)

    @pytest.mark.max_pool3d
    def test_return_indices_true(self):
        """测试：return_indices=True（返回输出和索引）"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems, idx_gems = max_pool3d(x, kernel_size=2, return_indices=True)
        y_torch, idx_torch = torch.nn.functional.max_pool3d(
            x, kernel_size=2, return_indices=True
        )
        # 验证输出值
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)
        # 验证索引（整数应该完全匹配）
        assert torch.equal(idx_gems, idx_torch)
        # 验证返回的是元组
        assert isinstance(y_gems, torch.Tensor)
        assert isinstance(idx_gems, torch.Tensor)

    @pytest.mark.max_pool3d
    def test_return_indices_4d_input(self):
        """测试：4D 输入 with return_indices=True"""
        x = create_tensor((16, 32, 32, 32), torch.float32)
        y_gems, idx_gems = max_pool3d(x, kernel_size=2, return_indices=True)
        y_torch, idx_torch = torch.nn.functional.max_pool3d(
            x, kernel_size=2, return_indices=True
        )
        # 验证输出值
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)
        # 验证索引
        assert torch.equal(idx_gems, idx_torch)
        # 验证输出维度
        assert y_gems.dim() == 4
        assert idx_gems.dim() == 4

    @pytest.mark.max_pool3d
    def test_batch_processing(self):
        """测试：批量处理（多张量）"""
        tensors = [
            create_tensor((2, 16, 32, 32, 32), torch.float32),
            create_tensor((4, 32, 64, 64, 64), torch.float32),
            create_tensor((1, 8, 16, 16, 16), torch.float32),
        ]
        for x in tensors:
            y_gems = max_pool3d(x, kernel_size=2)
            y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
            assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_consistency_with_pytorch(self):
        """测试：与 PyTorch 实现的一致性"""
        # 使用随机张量测试多次
        for _ in range(5):
            x = create_tensor((2, 16, 32, 32, 32), torch.float32)
            y_gems = max_pool3d(x, kernel_size=2)
            y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
            assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_empty_tensor(self):
        """测试：空张量处理"""
        x = torch.randn(0, 16, 32, 32, 32, dtype=torch.float32, device="cuda")
        y_gems = max_pool3d(x, kernel_size=2)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
        assert y_gems.shape == y_torch.shape
        assert y_gems.numel() == 0


# ============================================================================
# 6. 综合覆盖测试（参数化组合）
# ============================================================================


class TestMaxPool3DComprehensive:
    """综合测试：多维度组合覆盖"""

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize(
        "shape,kernel_size",
        [
            ((1, 1, 4, 4, 4), 2),
            ((2, 3, 8, 8, 8), 3),
            ((2, 16, 32, 32, 32), 2),
            ((4, 64, 64, 64, 64), 3),
        ],
    )
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_shape_dtype_combination(self, shape, kernel_size, dtype):
        """测试：形状和数据类型的组合"""
        x = create_tensor(shape, dtype)
        y_gems = max_pool3d(x, kernel_size=kernel_size)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=kernel_size)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize(
        "shape,kernel_size,stride,padding",
        [
            ((1, 1, 4, 4, 4), 2, 1, 0),
            ((2, 3, 8, 8, 8), 3, 2, 1),
            ((2, 16, 32, 32, 32), 2, 2, 1),
            ((4, 64, 64, 64, 64), 3, 3, 1),
        ],
    )
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_parameter_combinations(self, shape, kernel_size, stride, padding, dtype):
        """测试：参数组合"""
        x = create_tensor(shape, dtype)
        y_gems = max_pool3d(x, kernel_size=kernel_size, stride=stride, padding=padding)
        y_torch = torch.nn.functional.max_pool3d(
            x, kernel_size=kernel_size, stride=stride, padding=padding
        )
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize(
        "shape,dtype,kernel_size",
        [
            ((1, 1, 4, 4, 4), torch.float16, 2),
            ((2, 3, 8, 8, 8), torch.float32, 3),
            ((2, 16, 32, 32, 32), torch.bfloat16, 2),
            ((4, 64, 64, 64, 64), torch.float16, 3),
        ],
    )
    def test_typical_use_cases(self, shape, dtype, kernel_size):
        """测试：典型使用场景"""
        x = create_tensor(shape, dtype)
        y_gems = max_pool3d(x, kernel_size=kernel_size)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=kernel_size)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize(
        "shape,kernel_size,return_indices",
        [
            ((1, 1, 4, 4, 4), 2, False),
            ((2, 3, 8, 8, 8), 3, True),
            ((2, 16, 32, 32, 32), 2, False),
            ((4, 64, 64, 64, 64), 3, True),
        ],
    )
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_return_indices_combinations(
        self, shape, kernel_size, return_indices, dtype
    ):
        """测试：return_indices 参数组合"""
        x = create_tensor(shape, dtype)
        result_gems = max_pool3d(
            x, kernel_size=kernel_size, return_indices=return_indices
        )
        result_torch = torch.nn.functional.max_pool3d(
            x, kernel_size=kernel_size, return_indices=return_indices
        )

        if return_indices:
            y_gems, idx_gems = result_gems
            y_torch, idx_torch = result_torch
            atol = ATOL_DICT[dtype]
            assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)
            assert torch.equal(idx_gems, idx_torch)
        else:
            atol = ATOL_DICT[dtype]
            assert_close(result_gems, result_torch, rtol=1e-4, atol=atol, dtype=dtype)


# ============================================================================
# 7. 特殊场景测试
# ============================================================================


class TestMaxPool3DSpecialCases:
    """测试特殊场景"""

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_kernel_size_1(self, dtype):
        """测试：kernel_size=1（恒等操作）"""
        x = create_tensor((2, 16, 32, 32, 32), dtype)
        y_gems = max_pool3d(x, kernel_size=1)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=1)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)
        # kernel_size=1 时输出应该等于输入
        assert_close(y_gems, x, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_large_kernel(self, dtype):
        """测试：大 kernel（接近输入尺寸）"""
        x = create_tensor((2, 16, 16, 16, 16), dtype)
        y_gems = max_pool3d(x, kernel_size=8, padding=4)
        # y_gems = torch.nn.functional.max_pool3d(x, kernel_size=8, padding=4)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=8, padding=4)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_non_uniform_kernel(self, dtype):
        """测试：非均匀 kernel (2, 3, 4)"""
        x = create_tensor((2, 16, 32, 32, 32), dtype)
        y_gems = max_pool3d(x, kernel_size=(2, 3, 4))
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=(2, 3, 4))
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_asymmetric_stride_padding(self, dtype):
        """测试：非对称 stride 和 padding"""
        x = create_tensor((2, 16, 32, 32, 32), dtype)
        y_gems = max_pool3d(x, kernel_size=3, stride=(1, 2, 1), padding=(0, 1, 0))
        y_torch = torch.nn.functional.max_pool3d(
            x, kernel_size=3, stride=(1, 2, 1), padding=(0, 1, 0)
        )
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dilation_greater_than_1(self, dtype):
        """测试：dilation > 1（带空洞的池化）"""
        x = create_tensor((2, 16, 32, 32, 32), dtype)
        y_gems = max_pool3d(x, kernel_size=3, dilation=2)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=3, dilation=2)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)
