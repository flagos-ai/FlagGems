"""
Test suite for gcd operator.

This test module validates the correctness, precision, and performance
of the gcd operator implementation following FlagGems testing conventions.

测试覆盖说明：
- 输入规模：1×1, 8×8, 64×64, 256×256, 1024×1024, 4096×4096
- 输入维数：1D, 2D, 3D, 4D
- 数据类型：整数类型（int8, int16, int32, int64）
- 参数模式：默认值、边界值、特殊值、数学性质
- 功能完整性：基本计算、零值处理、负值处理、数学性质验证
"""

import pytest
import torch

from flag_gems.ops import gcd

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

# 整数类型覆盖（GCD 只支持整数）
INT_DTYPES = [
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
]

# 精度标准（GCD 是精确计算，不允许误差）
ATOL_DICT = {
    torch.int8: 0,
    torch.int16: 0,
    torch.int32: 0,
    torch.int64: 0,
}


# ============================================================================
# 辅助函数
# ============================================================================


def assert_equal(actual, expected, dtype=torch.int32):
    """
    使用 torch.equal 验证精确相等（GCD 必须完全匹配）

    Args:
        actual: FlagGems 实现结果
        expected: PyTorch 参考结果
        dtype: 数据类型
    """
    # GCD 必须精确匹配，不允许任何误差
    assert torch.equal(
        actual, expected
    ), f"GCD results don't match at {((actual != expected).sum().item())} positions"


def create_int_tensor(shape, dtype, device="cuda", low=1, high=1000):
    """创建随机整数测试张量"""
    # 根据数据类型调整范围，避免溢出
    dtype_limits = {
        torch.int8: 127,
        torch.int16: 32767,
        torch.int32: 2147483647,
        torch.int64: 9223372036854775807,
    }
    max_val = dtype_limits.get(dtype, 1000)
    adjusted_high = min(high, max_val)
    x = torch.randint(low, adjusted_high, shape, dtype=dtype, device=device)
    return x


# ============================================================================
# 1. 输入规模覆盖测试（比赛要求：小、常规、大三类）
# ============================================================================


class TestGcdInputSize:
    """测试输入规模覆盖"""

    @pytest.mark.gcd
    def test_size_very_small(self):
        """测试：1×1（极小尺寸）"""
        x = torch.tensor([[12]], device="cuda")
        y = torch.tensor([[8]], device="cuda")
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)

    @pytest.mark.gcd
    def test_size_small(self):
        """测试：8×8（小尺寸）"""
        x = create_int_tensor((8, 8), torch.int32)
        y = create_int_tensor((8, 8), torch.int32)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)

    @pytest.mark.gcd
    def test_size_medium_64(self):
        """测试：64×64（常规尺寸）"""
        x = create_int_tensor((64, 64), torch.int32)
        y = create_int_tensor((64, 64), torch.int32)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)

    @pytest.mark.gcd
    def test_size_medium_256(self):
        """测试：256×256（常规尺寸）"""
        x = create_int_tensor((256, 256), torch.int32)
        y = create_int_tensor((256, 256), torch.int32)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)

    @pytest.mark.gcd
    def test_size_large_1k(self):
        """测试：1024×1024（大尺寸）"""
        x = create_int_tensor((1024, 1024), torch.int32)
        y = create_int_tensor((1024, 1024), torch.int32)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)

    @pytest.mark.gcd
    def test_size_large_4k(self):
        """测试：4096×4096（大尺寸）"""
        x = create_int_tensor((4096, 4096), torch.int32)
        y = create_int_tensor((4096, 4096), torch.int32)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)


# ============================================================================
# 2. 输入维数覆盖测试（比赛要求：覆盖全部合法维数）
# ============================================================================


class TestGcdInputDimensions:
    """测试输入维数覆盖：1D, 2D, 3D, 4D"""

    @pytest.mark.gcd
    def test_dim_1d_small(self):
        """测试：1D 张量（小尺寸）"""
        x = create_int_tensor((10,), torch.int32)
        y = create_int_tensor((10,), torch.int32)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)

    @pytest.mark.gcd
    def test_dim_1d_large(self):
        """测试：1D 张量（大尺寸）"""
        x = create_int_tensor((10000,), torch.int32)
        y = create_int_tensor((10000,), torch.int32)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)

    @pytest.mark.gcd
    def test_dim_2d(self):
        """测试：2D 张量"""
        x = create_int_tensor((100, 100), torch.int32)
        y = create_int_tensor((100, 100), torch.int32)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)

    @pytest.mark.gcd
    def test_dim_3d(self):
        """测试：3D 张量"""
        x = create_int_tensor((64, 64, 64), torch.int32)
        y = create_int_tensor((64, 64, 64), torch.int32)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)

    @pytest.mark.gcd
    def test_dim_4d_batch(self):
        """测试：4D 批量张量（batch × channel × height × width）"""
        x = create_int_tensor((16, 3, 128, 128), torch.int32)
        y = create_int_tensor((16, 3, 128, 128), torch.int32)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)


# ============================================================================
# 3. 数据类型覆盖测试（比赛要求：支持所有整数类型）
# ============================================================================


class TestGcdDataTypes:
    """测试数据类型覆盖：int8, int16, int32, int64"""

    @pytest.mark.gcd
    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_all_int_dtypes(self, dtype):
        """测试：所有整数数据类型"""
        x = create_int_tensor((256, 256), dtype)
        y = create_int_tensor((256, 256), dtype)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch, dtype=dtype)


# ============================================================================
# 4. 参数模式覆盖测试（比赛要求：默认值、边界值、特殊值、数学性质）
# ============================================================================


class TestGcdParameterPatterns:
    """测试参数模式：默认值、边界值、特殊值、数学性质"""

    @pytest.mark.gcd
    def test_default_usage(self):
        """测试：默认参数使用"""
        x = create_int_tensor((100, 100), torch.int32)
        y = create_int_tensor((100, 100), torch.int32)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)

    @pytest.mark.gcd
    def test_edge_case_zero_values(self):
        """测试：边界值 - 零值处理"""
        # gcd(0, a) = a, gcd(a, 0) = a, gcd(0, 0) = 0
        x = torch.tensor([0, 5, 10, 0], device="cuda")
        y = torch.tensor([15, 0, 0, 0], device="cuda")
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        expected = torch.tensor([15, 5, 10, 0], device="cuda")
        assert_equal(z_gems, expected)
        assert_equal(z_torch, expected)

    @pytest.mark.gcd
    def test_special_negative_values(self):
        """测试：特殊值 - 负数值（gcd(a,b) = gcd(|a|,|b|)）"""
        x = torch.tensor([-12, -18, 24], device="cuda")
        y = torch.tensor([8, -12, -36], device="cuda")
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        expected = torch.tensor([4, 6, 12], device="cuda")
        assert_equal(z_gems, expected)
        assert_equal(z_torch, expected)

    @pytest.mark.gcd
    def test_special_prime_numbers(self):
        """测试：特殊值 - 质数（gcd = 1）"""
        x = torch.tensor([17, 19, 23], device="cuda")
        y = torch.tensor([13, 7, 11], device="cuda")
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        expected = torch.tensor([1, 1, 1], device="cuda")
        assert_equal(z_gems, expected)
        assert_equal(z_torch, expected)

    @pytest.mark.gcd
    def test_special_same_numbers(self):
        """测试：特殊值 - 相同数（gcd(a, a) = a）"""
        x = torch.tensor([42, 100, 255], device="cuda")
        y = torch.tensor([42, 100, 255], device="cuda")
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        expected = torch.tensor([42, 100, 255], device="cuda")
        assert_equal(z_gems, expected)
        assert_equal(z_torch, expected)

    @pytest.mark.gcd
    def test_special_large_numbers(self):
        """测试：特殊值 - 大数值"""
        x = torch.tensor([123456, 987654], device="cuda")
        y = torch.tensor([789012, 456789], device="cuda")
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)


# ============================================================================
# 5. 功能完整性测试（比赛要求：所有功能分支）
# ============================================================================


class TestGcdFunctionalCompleteness:
    """测试功能完整性：基本计算、零值处理、负值处理、数学性质"""

    @pytest.mark.gcd
    def test_basic_computation(self):
        """测试：基本元素-wise 计算"""
        x = torch.tensor([12, 18, 24, 35], device="cuda")
        y = torch.tensor([8, 12, 36, 49], device="cuda")
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        expected = torch.tensor([4, 6, 12, 7], device="cuda")
        assert_equal(z_gems, expected)
        assert_equal(z_torch, expected)

    @pytest.mark.gcd
    def test_commutative_property(self):
        """测试：交换律 gcd(a, b) = gcd(b, a)"""
        x = create_int_tensor((100, 100), torch.int32)
        y = create_int_tensor((100, 100), torch.int32)
        z_ab = gcd(x, y)
        z_ba = gcd(y, x)
        assert_equal(z_ab, z_ba)

    @pytest.mark.gcd
    def test_batch_processing(self):
        """测试：批量处理（多张量）"""
        tensors_x = [
            create_int_tensor((64, 64), torch.int32),
            create_int_tensor((128, 128), torch.int32),
            create_int_tensor((256, 256), torch.int32),
        ]
        tensors_y = [
            create_int_tensor((64, 64), torch.int32),
            create_int_tensor((128, 128), torch.int32),
            create_int_tensor((256, 256), torch.int32),
        ]
        for x, y in zip(tensors_x, tensors_y):
            z_gems = gcd(x, y)
            z_torch = torch.gcd(x, y)
            assert_equal(z_gems, z_torch)

    @pytest.mark.gcd
    def test_consistency_with_pytorch(self):
        """测试：与 PyTorch 实现的一致性"""
        # 使用随机张量测试多次
        for _ in range(10):
            x = create_int_tensor((100, 100), torch.int32)
            y = create_int_tensor((100, 100), torch.int32)
            z_gems = gcd(x, y)
            z_torch = torch.gcd(x, y)
            assert_equal(z_gems, z_torch)


# ============================================================================
# 6. 综合覆盖测试（参数化组合）
# ============================================================================


class TestGcdComprehensive:
    """综合测试：多维度组合覆盖"""

    @pytest.mark.gcd
    @pytest.mark.parametrize("shape", [8, 64 * 64, 256 * 256, 1024 * 1024])
    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_shape_dtype_combination(self, shape, dtype):
        """测试：形状和数据类型的组合"""
        x = create_int_tensor((shape,), dtype)
        y = create_int_tensor((shape,), dtype)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch, dtype=dtype)

    @pytest.mark.gcd
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((8, 8), torch.int8),
            ((64, 64), torch.int16),
            ((256, 256), torch.int32),
            ((1024, 1024), torch.int64),
        ],
    )
    def test_typical_use_cases(self, shape, dtype):
        """测试：典型使用场景"""
        x = create_int_tensor(shape, dtype)
        y = create_int_tensor(shape, dtype)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch, dtype=dtype)


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
- [x] torch.int8 (atol=0)
- [x] torch.int16 (atol=0)
- [x] torch.int32 (atol=0)
- [x] torch.int64 (atol=0)

### 4. 参数模式覆盖 ✓
- [x] 默认参数
- [x] 边界值：零值处理 gcd(0, a) = a
- [x] 特殊值：负数值 gcd(a, b) = gcd(|a|, |b|)
- [x] 特殊值：质数（gcd = 1）
- [x] 特殊值：相同数 gcd(a, a) = a
- [x] 特殊值：大数值
- [x] 数学性质：交换律 gcd(a, b) = gcd(b, a)

### 5. 功能完整性覆盖 ✓
- [x] 基本 element-wise 计算
- [x] 零值处理
- [x] 负值处理（绝对值）
- [x] 批量处理
- [x] 与 PyTorch 一致性验证
- [x] 交换律验证

### 6. 测试精度标准 ✓
- [x] 精确匹配（GCD 是整数运算，不允许误差）
- [x] 使用 torch.equal 验证
- [x] atol = 0 (所有整数类型)

### 7. 数学性质测试 ✓
- [x] 零值性质：gcd(0, a) = a
- [x] 负值性质：gcd(a, b) = gcd(|a|, |b|)
- [x] 交换律：gcd(a, b) = gcd(b, a)
- [x] 自反性：gcd(a, a) = a
"""
