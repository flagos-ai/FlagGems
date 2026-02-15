"""
Test suite for logaddexp operator.

This test module validates the correctness, precision, and performance
of the logaddexp operator implementation following FlagGems testing conventions.

测试覆盖说明：
- 输入规模：1×1, 8×8, 64×64, 256×256, 1024×1024, 4096×4096
- 输入维数：1D, 2D, 3D, 4D
- 数据类型：float16, float32, float64, bfloat16, 整数类型
- 参数模式：默认值、边界值、特殊值（极值、零值、数值稳定性）
- 功能完整性：基本计算、类型提升、批处理、数值稳定性

算子公式：logaddexp(x, y) = log(exp(x) + exp(y))
数值稳定实现：max(x, y) + log(1 + exp(-|x - y|))
"""

import pytest
import torch

from flag_gems.ops import logaddexp

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


def create_tensor(shape, dtype, device="cuda"):
    """创建张量（用于 logaddexp 测试）"""
    x = torch.randn(shape, dtype=dtype, device=device)
    return x


# ============================================================================
# 1. 输入规模覆盖测试（比赛要求：小、常规、大三类）
# ============================================================================


class TestLogAddExpInputSize:
    """测试输入规模覆盖"""

    @pytest.mark.logaddexp
    def test_size_very_small(self):
        """测试：1×1（极小尺寸）"""
        x = torch.tensor([[0.5]], device="cuda")
        y = torch.tensor([[0.3]], device="cuda")
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.logaddexp
    def test_size_small(self):
        """测试：8×8（小尺寸）"""
        x = create_tensor((8, 8), torch.float32)
        y = create_tensor((8, 8), torch.float32)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.logaddexp
    def test_size_medium_64(self):
        """测试：64×64（常规尺寸）"""
        x = create_tensor((64, 64), torch.float32)
        y = create_tensor((64, 64), torch.float32)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.logaddexp
    def test_size_medium_256(self):
        """测试：256×256（常规尺寸）"""
        x = create_tensor((256, 256), torch.float32)
        y = create_tensor((256, 256), torch.float32)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.logaddexp
    def test_size_large_1k(self):
        """测试：1024×1024（大尺寸）"""
        x = create_tensor((1024, 1024), torch.float32)
        y = create_tensor((1024, 1024), torch.float32)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.logaddexp
    def test_size_large_4k(self):
        """测试：4096×4096（大尺寸）"""
        x = create_tensor((4096, 4096), torch.float32)
        y = create_tensor((4096, 4096), torch.float32)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 2. 输入维数覆盖测试（比赛要求：覆盖全部合法维数）
# ============================================================================


class TestLogAddExpInputDimensions:
    """测试输入维数覆盖：1D, 2D, 3D, 4D"""

    @pytest.mark.logaddexp
    def test_dim_1d_small(self):
        """测试：1D 张量（小尺寸）"""
        x = create_tensor((10,), torch.float32)
        y = create_tensor((10,), torch.float32)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.logaddexp
    def test_dim_1d_large(self):
        """测试：1D 张量（大尺寸）"""
        x = create_tensor((10000,), torch.float32)
        y = create_tensor((10000,), torch.float32)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.logaddexp
    def test_dim_2d(self):
        """测试：2D 张量"""
        x = create_tensor((100, 100), torch.float32)
        y = create_tensor((100, 100), torch.float32)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.logaddexp
    def test_dim_3d(self):
        """测试：3D 张量"""
        x = create_tensor((64, 64, 64), torch.float32)
        y = create_tensor((64, 64, 64), torch.float32)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.logaddexp
    def test_dim_4d_batch(self):
        """测试：4D 批量张量（batch × channel × height × width）"""
        x = create_tensor((16, 3, 128, 128), torch.float32)
        y = create_tensor((16, 3, 128, 128), torch.float32)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 3. 数据类型覆盖测试（比赛要求：至少支持 float32/float16）
# ============================================================================


class TestLogAddExpDataTypes:
    """测试数据类型覆盖：float16, float32, float64, bfloat16, 整数"""

    @pytest.mark.logaddexp
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_all_float_dtypes(self, dtype):
        """测试：所有浮点数据类型"""
        x = create_tensor((256, 256), dtype)
        y = create_tensor((256, 256), dtype)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        atol = ATOL_DICT[dtype]
        assert_close(z_gems, z_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.logaddexp
    def test_integer_input_not_supported(self):
        """测试：整数输入应抛出错误（与 PyTorch 行为一致）

        PyTorch 的 logaddexp 不支持整数输入，会抛出 NotImplementedError。
        我们的实现应该保持一致的行为。
        """
        x = torch.tensor([1, 10, 100, 1000], device="cuda")
        y = torch.tensor([2, 20, 200, 2000], device="cuda")

        # PyTorch 抛出 NotImplementedError
        with pytest.raises(NotImplementedError, match="not implemented for 'Long'"):
            torch.logaddexp(x, y)

        # 我们的实现应该也抛出相同的错误
        with pytest.raises((NotImplementedError, TypeError)):
            logaddexp(x, y)


# ============================================================================
# 4. 参数模式覆盖测试（比赛要求：默认值、边界值、特殊值、数值稳定性）
# ============================================================================


class TestLogAddExpParameterPatterns:
    """测试参数模式：默认值、边界值、特殊值、数值稳定性"""

    @pytest.mark.logaddexp
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_default_usage(self, dtype):
        """测试：默认参数使用"""
        x = create_tensor((100, 100), dtype)
        y = create_tensor((100, 100), dtype)
        z_gems = logaddexp(x, y)  # 默认调用
        z_torch = torch.logaddexp(x, y)
        atol = ATOL_DICT[dtype]
        assert_close(z_gems, z_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.logaddexp
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_edge_case_equal_values(self, dtype):
        """测试：边界值 - 相等值 log(exp(x) + exp(x)) = log(2*exp(x)) = x + log(2)"""
        x = torch.ones((10, 10), dtype=dtype, device="cuda")
        y = torch.ones((10, 10), dtype=dtype, device="cuda")
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        atol = ATOL_DICT[dtype]
        assert_close(z_gems, z_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.logaddexp
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_edge_case_one_value_zero(self, dtype):
        """测试：边界值 - 一个值为 0"""
        x = torch.zeros((10, 10), dtype=dtype, device="cuda")
        y = torch.ones((10, 10), dtype=dtype, device="cuda")
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        atol = ATOL_DICT[dtype]
        assert_close(z_gems, z_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.logaddexp
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_special_small_values(self, dtype):
        """测试：特殊值 - 小负数（数值稳定性关键）"""
        x = torch.tensor([-100.0, -50.0, -10.0], dtype=dtype, device="cuda")
        y = torch.tensor([-90.0, -60.0, -5.0], dtype=dtype, device="cuda")
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        atol = ATOL_DICT[dtype]
        assert_close(z_gems, z_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.logaddexp
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_special_large_values(self, dtype):
        """测试：特殊值 - 大正数"""
        x = torch.tensor([100.0, 500.0, 1000.0], dtype=dtype, device="cuda")
        y = torch.tensor([200.0, 600.0, 900.0], dtype=dtype, device="cuda")
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        atol = ATOL_DICT[dtype]
        assert_close(z_gems, z_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.logaddexp
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_special_extreme_values(self, dtype):
        """测试：极端值 - 数值稳定性关键测试"""
        # 测试非常大的值，验证数值稳定性
        x = torch.tensor([1000.0, -1000.0], dtype=dtype, device="cuda")
        y = torch.tensor([1001.0, -999.0], dtype=dtype, device="cuda")
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        atol = ATOL_DICT[dtype]
        assert_close(z_gems, z_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.logaddexp
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_numerical_stability(self, dtype):
        """测试：数值稳定性 - 混合正负值"""
        # log(exp(1000) + exp(-1000)) ≈ 1000
        # log(exp(-1000) + exp(1000)) ≈ 1000
        x = torch.tensor([1000.0, -1000.0, 0.0, 500.0], dtype=dtype, device="cuda")
        y = torch.tensor([1001.0, -999.0, 1.0, 500.0], dtype=dtype, device="cuda")
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        atol = ATOL_DICT[dtype]
        assert_close(z_gems, z_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.logaddexp
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_symmetry_property(self, dtype):
        """测试：对称性 logaddexp(x, y) = logaddexp(y, x)"""
        x = create_tensor((100, 100), dtype)
        y = create_tensor((100, 100), dtype)
        z_gems = logaddexp(x, y)
        z_gems_swapped = logaddexp(y, x)
        assert torch.allclose(z_gems, z_gems_swapped, rtol=0, atol=0)


# ============================================================================
# 5. 功能完整性测试（比赛要求：所有功能分支）
# ============================================================================


class TestLogAddExpFunctionalCompleteness:
    """测试功能完整性：基本计算、类型提升、批处理、对称性"""

    @pytest.mark.logaddexp
    def test_basic_computation(self):
        """测试：基本元素-wise 计算"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
        y = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0], device="cuda")
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.logaddexp
    def test_dtype_consistency(self):
        """测试：浮点类型一致性"""
        # 验证不同浮点类型的行为一致性
        x_f32 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="cuda")
        y_f32 = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32, device="cuda")
        x_f64 = x_f32.to(torch.float64)
        y_f64 = y_f32.to(torch.float64)

        z_f32 = logaddexp(x_f32, y_f32)
        z_f64 = logaddexp(x_f64, y_f64)

        # 转换后应该接近（考虑精度差异）
        assert torch.allclose(z_f32.to(torch.float64), z_f64, rtol=1e-5, atol=1e-6)

    @pytest.mark.logaddexp
    def test_batch_processing(self):
        """测试：批量处理（多张量）"""
        tensors_x = [
            create_tensor((64, 64), torch.float32),
            create_tensor((128, 128), torch.float32),
            create_tensor((256, 256), torch.float32),
        ]
        tensors_y = [
            create_tensor((64, 64), torch.float32),
            create_tensor((128, 128), torch.float32),
            create_tensor((256, 256), torch.float32),
        ]
        for x, y in zip(tensors_x, tensors_y):
            z_gems = logaddexp(x, y)
            z_torch = torch.logaddexp(x, y)
            assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.logaddexp
    def test_consistency_with_pytorch(self):
        """测试：与 PyTorch 实现的一致性"""
        # 使用随机张量测试多次
        for _ in range(10):
            x = create_tensor((100, 100), torch.float32)
            y = create_tensor((100, 100), torch.float32)
            z_gems = logaddexp(x, y)
            z_torch = torch.logaddexp(x, y)
            assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 6. 综合覆盖测试（参数化组合）
# ============================================================================


class TestLogAddExpComprehensive:
    """综合测试：多维度组合覆盖"""

    @pytest.mark.logaddexp
    @pytest.mark.parametrize("shape", [8, 64 * 64, 256 * 256, 1024 * 1024])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_shape_dtype_combination(self, shape, dtype):
        """测试：形状和数据类型的组合"""
        x = create_tensor((shape,), dtype)
        y = create_tensor((shape,), dtype)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        atol = ATOL_DICT[dtype]
        assert_close(z_gems, z_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.logaddexp
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
        y = create_tensor(shape, dtype)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        atol = ATOL_DICT[dtype]
        assert_close(z_gems, z_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.logaddexp
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((8, 8), torch.float16),
            ((64, 64), torch.float32),
            ((256, 256), torch.float64),
            ((1024, 1024), torch.bfloat16),
        ],
    )
    def test_commutative_property(self, shape, dtype):
        """测试：交换律 logaddexp(x, y) = logaddexp(y, x)"""
        x = create_tensor(shape, dtype)
        y = create_tensor(shape, dtype)
        z_xy = logaddexp(x, y)
        z_yx = logaddexp(y, x)
        assert_close(z_xy, z_yx, rtol=0, atol=0, dtype=dtype)


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
- [x] 整数类型（抛出 NotImplementedError，与 PyTorch 一致）

### 4. 参数模式覆盖 ✓
- [x] 默认参数
- [x] 边界值：相等值 logaddexp(x, x) = x + log(2)
- [x] 边界值：一个为 0
- [x] 特殊值：小负数（数值稳定性测试）
- [x] 特殊值：大正数
- [x] 极端值：±1000（数值稳定性关键测试）
- [x] 数值稳定性：混合正负值
- [x] 对称性：logaddexp(x, y) = logaddexp(y, x)

### 5. 功能完整性覆盖 ✓
- [x] 基本 element-wise 计算
- [x] 整数输入错误处理（NotImplementedError）
- [x] 批量处理
- [x] 与 PyTorch 一致性验证
- [x] 交换律验证

### 6. 测试精度标准 ✓
- [x] rtol = 1e-4 (所有浮点类型)
- [x] atol 根据数据类型变化（符合比赛要求）
- [x] 使用 torch.allclose 验证（比赛标准）

### 7. 数值稳定性测试 ✓
- [x] 极端值测试（±1000）
- [x] 混合正负值测试
- [x] 大值相加精度保持
"""
