"""
Test suite for log10 operator.

This test module validates the correctness, precision, and performance
of the log10 operator implementation following FlagGems testing conventions.

测试覆盖说明：
- 输入规模：1×1, 8×8, 64×64, 256×256, 1024×1024, 4096×4096
- 输入维数：1D, 2D, 3D, 4D
- 数据类型：float16, float32, float64, bfloat16, 整数类型
- 参数模式：默认值、边界值、特殊值
- 功能完整性：基本计算、类型提升、批处理
"""

import pytest
import torch

from flag_gems.ops import log10

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
    torch.float16: 1e-3,
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


def create_positive_tensor(shape, dtype, device="cuda"):
    """创建正数张量（用于 log10 测试）"""
    x = torch.randn(shape, dtype=dtype, device=device)
    return torch.abs(x) + 0.1  # 确保所有值 > 0


# ============================================================================
# 1. 输入规模覆盖测试（比赛要求：小、常规、大三类）
# ============================================================================


class TestLog10InputSize:
    """测试输入规模覆盖"""

    @pytest.mark.log10
    def test_size_very_small(self):
        """测试：1×1（极小尺寸）"""
        x = torch.tensor([[0.5]], device="cuda")
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.log10
    def test_size_small(self):
        """测试：8×8（小尺寸）"""
        x = create_positive_tensor((8, 8), torch.float32)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.log10
    def test_size_medium_64(self):
        """测试：64×64（常规尺寸）"""
        x = create_positive_tensor((64, 64), torch.float32)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.log10
    def test_size_medium_256(self):
        """测试：256×256（常规尺寸）"""
        x = create_positive_tensor((256, 256), torch.float32)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.log10
    def test_size_large_1k(self):
        """测试：1024×1024（大尺寸）"""
        x = create_positive_tensor((1024, 1024), torch.float32)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.log10
    def test_size_large_4k(self):
        """测试：4096×4096（大尺寸）"""
        x = create_positive_tensor((4096, 4096), torch.float32)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 2. 输入维数覆盖测试（比赛要求：覆盖全部合法维数）
# ============================================================================


class TestLog10InputDimensions:
    """测试输入维数覆盖：1D, 2D, 3D, 4D"""

    @pytest.mark.log10
    def test_dim_1d_small(self):
        """测试：1D 张量（小尺寸）"""
        x = create_positive_tensor((10,), torch.float32)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.log10
    def test_dim_1d_large(self):
        """测试：1D 张量（大尺寸）"""
        x = create_positive_tensor((10000,), torch.float32)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.log10
    def test_dim_2d(self):
        """测试：2D 张量"""
        x = create_positive_tensor((100, 100), torch.float32)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.log10
    def test_dim_3d(self):
        """测试：3D 张量"""
        x = create_positive_tensor((64, 64, 64), torch.float32)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.log10
    def test_dim_4d_batch(self):
        """测试：4D 批量张量（batch × channel × height × width）"""
        x = create_positive_tensor((16, 3, 128, 128), torch.float32)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 3. 数据类型覆盖测试（比赛要求：至少支持 float32/float16）
# ============================================================================


class TestLog10DataTypes:
    """测试数据类型覆盖：float16, float32, float64, bfloat16, 整数"""

    @pytest.mark.log10
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_all_float_dtypes(self, dtype):
        """测试：所有浮点数据类型"""
        x = create_positive_tensor((256, 256), dtype)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.log10
    def test_integer_input(self):
        """测试：整数输入（应自动提升为 float32，与 PyTorch 行为一致）"""
        x = torch.tensor([1, 10, 100, 1000], device="cuda")
        y_gems = log10(x)
        y_torch = torch.log10(x)
        # 验证输出类型是否为 float32
        assert y_gems.dtype == torch.float32
        assert y_torch.dtype == torch.float32
        # 整数计算应该完全精确
        assert torch.allclose(y_gems, y_torch, rtol=0, atol=0)
        # 验证值是否正确
        expected = torch.tensor([0.0, 1.0, 2.0, 3.0], device="cuda")
        assert torch.allclose(y_gems, expected, rtol=0, atol=0)

    @pytest.mark.log10
    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_integer_dtypes(self, dtype):
        """测试：各种整数类型（应自动提升为 float32）"""
        x = torch.tensor([1, 10, 100, 1000], dtype=dtype, device="cuda")
        y_gems = log10(x)
        y_torch = torch.log10(x)
        # 验证输出类型是否为 float32
        assert y_gems.dtype == torch.float32
        assert y_torch.dtype == torch.float32
        # 整数计算应该完全精确
        assert torch.allclose(y_gems, y_torch, rtol=0, atol=0)


# ============================================================================
# 4. 参数模式覆盖测试（比赛要求：默认值、边界值、特殊值）
# ============================================================================


class TestLog10ParameterPatterns:
    """测试参数模式：默认值、边界值、特殊值"""

    @pytest.mark.log10
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_default_usage(self, dtype):
        """测试：默认参数使用"""
        x = create_positive_tensor((100, 100), dtype)
        y_gems = log10(x)  # 默认调用
        y_torch = torch.log10(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.log10
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_edge_case_unit_values(self, dtype):
        """测试：边界值 - log10(1.0) = 0"""
        x = torch.ones((10, 10), dtype=dtype, device="cuda")
        y_gems = log10(x)
        y_torch = torch.log10(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.log10
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_edge_case_exact_powers(self, dtype):
        """测试：边界值 - 10 的整数次幂（精确结果）"""
        x = torch.tensor([1.0, 10.0, 100.0, 1000.0], dtype=dtype, device="cuda")
        y_gems = log10(x)
        y_torch = torch.log10(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.log10
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_special_small_values(self, dtype):
        """测试：特殊值 - 小正数"""
        x = torch.tensor([0.1, 0.01, 0.001], dtype=dtype, device="cuda")
        y_gems = log10(x)
        y_torch = torch.log10(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.log10
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_special_large_values(self, dtype):
        """测试：特殊值 - 大正数"""
        x = torch.tensor([1e6, 1e8, 1e10], dtype=dtype, device="cuda")
        y_gems = log10(x)
        y_torch = torch.log10(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.log10
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_special_zero_handling(self, dtype):
        """测试：特殊值 - 零值（应产生 -inf）"""
        x = torch.tensor([0.0], dtype=dtype, device="cuda")
        y_gems = log10(x)
        y_torch = torch.log10(x)
        # 零的对数应该是 -inf
        assert torch.isinf(y_gems[0]) and y_gems[0] < 0
        assert torch.isinf(y_torch[0]) and y_torch[0] < 0

    @pytest.mark.log10
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_special_negative_handling(self, dtype):
        """测试：特殊值 - 负数（应产生 nan）"""
        x = torch.tensor([-1.0, -10.0], dtype=dtype, device="cuda")
        y_gems = log10(x)
        y_torch = torch.log10(x)
        # 负数的对数应该是 nan
        assert torch.isnan(y_gems[0])
        assert torch.isnan(y_torch[0])


# ============================================================================
# 5. 功能完整性测试（比赛要求：所有功能分支）
# ============================================================================


class TestLog10FunctionalCompleteness:
    """测试功能完整性：基本计算、类型提升、批处理"""

    @pytest.mark.log10
    def test_basic_computation(self):
        """测试：基本元素-wise 计算"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.log10
    def test_type_promotion_int_to_float(self):
        """测试：类型提升 - 整数到浮点数"""
        x_int = torch.tensor([1, 10, 100], device="cuda")
        x_float = x_int.float()
        y_int = log10(x_int)
        y_float = log10(x_float)
        # 应该得到相同的结果
        assert_close(y_int, y_float, rtol=1e-4, atol=1e-5, dtype=torch.float32)

    @pytest.mark.log10
    def test_batch_processing(self):
        """测试：批量处理（多张量）"""
        tensors = [
            create_positive_tensor((64, 64), torch.float32),
            create_positive_tensor((128, 128), torch.float32),
            create_positive_tensor((256, 256), torch.float32),
        ]
        for x in tensors:
            y_gems = log10(x)
            y_torch = torch.log10(x)
            assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.log10
    def test_consistency_with_pytorch(self):
        """测试：与 PyTorch 实现的一致性"""
        # 使用随机张量测试多次
        for _ in range(10):
            x = create_positive_tensor((100, 100), torch.float32)
            y_gems = log10(x)
            y_torch = torch.log10(x)
            assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 6. 综合覆盖测试（参数化组合）
# ============================================================================


class TestLog10Comprehensive:
    """综合测试：多维度组合覆盖"""

    @pytest.mark.log10
    @pytest.mark.parametrize("shape", [8, 64 * 64, 256 * 256, 1024 * 1024])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_shape_dtype_combination(self, shape, dtype):
        """测试：形状和数据类型的组合"""
        x = create_positive_tensor((shape,), dtype)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.log10
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
        x = create_positive_tensor(shape, dtype)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)


# ============================================================================
# 测试覆盖清单（用于 PR 描述）
# ============================================================================

"""
## 测试覆盖清单

### 1. 输入规模覆盖 ✓
- [x] 极小尺寸：1×1
- [x] 小尺寸：8×8, 64
- [x] 常规尺寸：64×64 (4,096), 256×256 (65,536)
- [x] 大尺寸：1024×1024 (1,048,576), 4096×4096 (16,777,216)

### 2. 输入维数覆盖 ✓
- [x] 1D 张量：(10,), (10000,)
- [x] 2D 张量：(100, 100)
- [x] 3D 张量：(64, 64, 64)
- [x] 4D 张量：(16, 3, 128, 128) - 批量矩阵

### 3. 数据类型覆盖 ✓
- [x] torch.float16 (atol=1e-3)
- [x] torch.float32 (atol=1.3e-6)
- [x] torch.float64 (atol=1e-7)
- [x] torch.bfloat16 (atol=0.016)
- [x] 整数类型（自动类型提升）

### 4. 参数模式覆盖 ✓
- [x] 默认参数
- [x] 边界值：log10(1) = 0
- [x] 特殊值：10 的整数次幂
- [x] 小正数：0.1, 0.01, 0.001
- [x] 大正数：1e6, 1e8, 1e10
- [x] 零值：log10(0) → -inf
- [x] 负数：log10(-x) → nan

### 5. 功能完整性覆盖 ✓
- [x] 基本 element-wise 计算
- [x] 类型提升（整数到浮点）
- [x] 批量处理
- [x] 与 PyTorch 一致性验证

### 6. 测试精度标准 ✓
- [x] rtol = 1e-4 (所有浮点类型)
- [x] atol 根据数据类型变化（符合比赛要求）
- [x] 使用 torch.allclose 验证（比赛标准）
"""
