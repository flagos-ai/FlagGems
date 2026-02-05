"""
Test suite for cosh operator.

This test module validates the correctness, precision, and performance
of the cosh operator implementation following FlagGems testing conventions.

测试覆盖说明：
- 输入规模：1×1, 8×8, 64×64, 256×256, 1024×1024, 4096×4096
- 输入维数：1D, 2D, 3D, 4D
- 数据类型：float16, float32, float64, bfloat16
- 参数模式：默认值、边界值、特殊值、数值稳定性
- 功能完整性：基本计算、类型提升、批处理、对称性
"""

import pytest
import torch

from flag_gems.ops import cosh

# ============================================================================
# 测试数据定义（按照比赛要求）
# ============================================================================

# 输入规模覆盖（比赛要求：小尺寸、常规尺寸、大尺寸）
POINTWISE_SHAPES = [
    8,  # 小尺寸
    64,  # 小尺寸
    64 * 64,  # 常规尺寸
    256 * 256,  # 常规尺寸
    1024 * 1024,  # 大尺寸
    4096 * 4096,  # 大尺寸
]

# 数据类型覆盖（比赛要求：至少支持 float32/float16）
FLOAT_DTYPES = [
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
]

# 精度标准（比赛要求的标准）
# rtol = 1e-4 (所有浮点类型)
# atol 根据数据类型变化
ATOL_DICT = {
    torch.float16: 2.0e-3,  # Adjusted for float16 precision (2 × machine_epsilon)
    torch.float32: 1.3e-6,
    torch.float64: 1e-7,
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


class TestCoshInputSize:
    """测试输入规模覆盖"""

    @pytest.mark.cosh
    def test_size_very_small(self):
        """测试：1×1（极小尺寸）"""
        x = torch.tensor([[0.5]], device="cuda")
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_size_small(self):
        """测试：8×8（小尺寸）"""
        x = create_tensor((8, 8), torch.float32)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_size_medium_64(self):
        """测试：64×64（常规尺寸）"""
        x = create_tensor((64, 64), torch.float32)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_size_medium_256(self):
        """测试：256×256（常规尺寸）"""
        x = create_tensor((256, 256), torch.float32)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_size_large_1k(self):
        """测试：1024×1024（大尺寸）"""
        x = create_tensor((1024, 1024), torch.float32)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_size_large_4k(self):
        """测试：4096×4096（大尺寸）"""
        x = create_tensor((4096, 4096), torch.float32)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 2. 输入维数覆盖测试（比赛要求：覆盖全部合法维数）
# ============================================================================


class TestCoshInputDimensions:
    """测试输入维数覆盖：1D, 2D, 3D, 4D"""

    @pytest.mark.cosh
    def test_dim_1d_small(self):
        """测试：1D 张量（小尺寸）"""
        x = create_tensor((10,), torch.float32)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_dim_1d_large(self):
        """测试：1D 张量（大尺寸）"""
        x = create_tensor((10000,), torch.float32)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_dim_2d(self):
        """测试：2D 张量"""
        x = create_tensor((100, 100), torch.float32)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_dim_3d(self):
        """测试：3D 张量"""
        x = create_tensor((64, 64, 64), torch.float32)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_dim_4d_batch(self):
        """测试：4D 批量张量（batch × channel × height × width）"""
        x = create_tensor((16, 3, 128, 128), torch.float32)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 3. 数据类型覆盖测试（比赛要求：至少支持 float32/float16）
# ============================================================================


class TestCoshDataTypes:
    """测试数据类型覆盖：float16, float32, float64, bfloat16"""

    @pytest.mark.cosh
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_all_float_dtypes(self, dtype):
        """测试：所有浮点数据类型"""
        x = create_tensor((256, 256), dtype)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)


# ============================================================================
# 4. 参数模式覆盖测试（比赛要求：默认值、边界值、特殊值、数值稳定性）
# ============================================================================


class TestCoshParameterPatterns:
    """测试参数模式：默认值、边界值、特殊值、数值稳定性"""

    @pytest.mark.cosh
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_default_usage(self, dtype):
        """测试：默认参数使用"""
        x = create_tensor((100, 100), dtype)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.cosh
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_edge_case_zero(self, dtype):
        """测试：边界值 - cosh(0) = 1"""
        x = torch.zeros((10, 10), dtype=dtype, device="cuda")
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        # cosh(0) should be exactly 1
        expected = torch.ones((10, 10), dtype=dtype, device="cuda")
        assert_close(y_gems, expected, rtol=1e-4, atol=ATOL_DICT[dtype], dtype=dtype)
        assert_close(y_torch, expected, rtol=1e-4, atol=ATOL_DICT[dtype], dtype=dtype)

    @pytest.mark.cosh
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_edge_case_small_values(self, dtype):
        """测试：边界值 - 小数值"""
        x = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=dtype, device="cuda")
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.cosh
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_special_negative_values(self, dtype):
        """测试：特殊值 - 负数值（对称性测试）"""
        x = torch.tensor([-1.0, -2.0, -3.0, -10.0], dtype=dtype, device="cuda")
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.cosh
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_special_large_values(self, dtype):
        """测试：特殊值 - 大数值（数值稳定性测试）"""
        x = torch.tensor([10.0, 50.0, 100.0], dtype=dtype, device="cuda")
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.cosh
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_numerical_stability_extreme(self, dtype):
        """测试：数值稳定性 - 极端值（±100）"""
        x = torch.tensor(
            [0.0, 1.0, -1.0, 10.0, -10.0, 100.0, -100.0], dtype=dtype, device="cuda"
        )
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.cosh
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_symmetry_property(self, dtype):
        """测试：对称性 cosh(-x) = cosh(x)"""
        x = create_tensor((100, 100), dtype)
        y_pos = cosh(x)
        y_neg = cosh(-x)
        # cosh should be symmetric
        assert_close(y_pos, y_neg, rtol=0, atol=0, dtype=dtype)


# ============================================================================
# 5. 功能完整性测试（比赛要求：所有功能分支）
# ============================================================================


class TestCoshFunctionalCompleteness:
    """测试功能完整性：基本计算、批处理、对称性、一致性"""

    @pytest.mark.cosh
    def test_basic_computation(self):
        """测试：基本元素-wise 计算"""
        x = torch.tensor([0.0, 1.0, 2.0, 3.0], device="cuda")
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_symmetry_verification(self):
        """测试：对称性验证 cosh(-x) = cosh(x)"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
        y_pos = cosh(x)
        y_neg = cosh(-x)
        assert_close(y_pos, y_neg, rtol=0, atol=0, dtype=torch.float32)

    @pytest.mark.cosh
    def test_batch_processing(self):
        """测试：批量处理（多张量）"""
        tensors = [
            create_tensor((64, 64), torch.float32),
            create_tensor((128, 128), torch.float32),
            create_tensor((256, 256), torch.float32),
        ]
        for x in tensors:
            y_gems = cosh(x)
            y_torch = torch.cosh(x)
            assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_consistency_with_pytorch(self):
        """测试：与 PyTorch 实现的一致性"""
        # 使用随机张量测试多次
        for _ in range(10):
            x = create_tensor((100, 100), torch.float32)
            y_gems = cosh(x)
            y_torch = torch.cosh(x)
            assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_known_values(self):
        """测试：已知精确值"""
        # cosh(0) = 1
        # cosh(1) ≈ 1.543080634815244
        # cosh(2) ≈ 3.762195691083631
        x = torch.tensor([0.0, 1.0, 2.0], device="cuda")
        y = cosh(x)
        expected = torch.tensor(
            [1.0, 1.543080634815244, 3.762195691083631], device="cuda"
        )
        assert_close(y, expected, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 6. 综合覆盖测试（参数化组合）
# ============================================================================


class TestCoshComprehensive:
    """综合测试：多维度组合覆盖"""

    @pytest.mark.cosh
    @pytest.mark.parametrize("shape", [8, 64 * 64, 256 * 256, 1024 * 1024])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_shape_dtype_combination(self, shape, dtype):
        """测试：形状和数据类型的组合"""
        x = create_tensor((shape,), dtype)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.cosh
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((8, 8), torch.float16),
            ((64, 64), torch.float32),
            ((256, 256), torch.float64),
            ((1024, 1024), torch.bfloat16),
        ],
    )
    def test_typical_use_cases(self, shape, dtype):
        """测试：典型使用场景"""
        x = create_tensor(shape, dtype)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)
