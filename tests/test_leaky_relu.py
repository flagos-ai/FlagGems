"""
Test suite for LeakyReLU operator.

This test module validates the correctness, precision, and performance
of the LeakyReLU operator implementation following FlagGems testing conventions.

测试覆盖说明：
- 输入规模：1×1, 8×8, 64×64, 256×256, 1024×1024, 4096×4096
- 输入维数：1D, 2D, 3D, 4D, 5D
- 数据类型：float16, float32
- 参数模式：默认值、边界值、特殊值、不同负斜率
- 功能完整性：正值分支、负值分支、零值处理、批处理
"""

import pytest
import torch

from flag_gems.ops import leaky_relu

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
]

# 精度标准（比赛要求的标准）
# rtol = 1e-4 (所有浮点类型)
# atol 根据数据类型变化
ATOL_DICT = {
    torch.float16: 1e-5,
    torch.float32: 1e-6,
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
        atol = ATOL_DICT.get(dtype, 1e-6)

    # 使用 torch.allclose 进行比较（比赛标准）
    assert torch.allclose(
        actual, expected, rtol=rtol, atol=atol, equal_nan=True
    ), f"Results don't match: max diff={(actual - expected).abs().max().item()}"


def create_tensor(shape, dtype, device="cuda"):
    """创建测试张量（包含正负值）"""
    x = torch.randn(shape, dtype=dtype, device=device)
    return x


# ============================================================================
# 1. 输入规模覆盖测试（比赛要求：小、常规、大三类）
# ============================================================================


class TestLeakyReLUInputSize:
    """测试输入规模覆盖"""

    @pytest.mark.leaky_relu
    def test_size_very_small(self):
        """测试：1×1（极小尺寸）"""
        x = torch.tensor([[0.5]], device="cuda")
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_size_small(self):
        """测试：8×8（小尺寸）"""
        x = create_tensor((8, 8), torch.float32)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_size_medium_64(self):
        """测试：64×64（常规尺寸）"""
        x = create_tensor((64, 64), torch.float32)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_size_medium_256(self):
        """测试：256×256（常规尺寸）"""
        x = create_tensor((256, 256), torch.float32)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_size_large_1k(self):
        """测试：1024×1024（大尺寸）"""
        x = create_tensor((1024, 1024), torch.float32)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_size_large_4k(self):
        """测试：4096×4096（大尺寸）"""
        x = create_tensor((4096, 4096), torch.float32)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)


# ============================================================================
# 2. 输入维数覆盖测试（比赛要求：覆盖全部合法维数）
# ============================================================================


class TestLeakyReLUInputDimensions:
    """测试输入维数覆盖：1D, 2D, 3D, 4D, 5D"""

    @pytest.mark.leaky_relu
    def test_dim_1d_small(self):
        """测试：1D 张量（小尺寸）"""
        x = create_tensor((10,), torch.float32)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_dim_1d_large(self):
        """测试：1D 张量（大尺寸）"""
        x = create_tensor((10000,), torch.float32)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_dim_2d(self):
        """测试：2D 张量"""
        x = create_tensor((100, 100), torch.float32)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_dim_3d(self):
        """测试：3D 张量"""
        x = create_tensor((64, 64, 64), torch.float32)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_dim_4d_batch(self):
        """测试：4D 批量张量（batch × channel × height × width）"""
        x = create_tensor((16, 3, 128, 128), torch.float32)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_dim_5d(self):
        """测试：5D 张量"""
        x = create_tensor((2, 3, 4, 5, 6), torch.float32)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)


# ============================================================================
# 3. 数据类型覆盖测试（比赛要求：至少支持 float32/float16）
# ============================================================================


class TestLeakyReLUDataTypes:
    """测试数据类型覆盖：float16, float32"""

    @pytest.mark.leaky_relu
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_all_float_dtypes(self, dtype):
        """测试：所有浮点数据类型"""
        x = create_tensor((256, 256), dtype)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)


# ============================================================================
# 4. 参数模式覆盖测试（比赛要求：默认值、边界值、特殊值）
# ============================================================================


class TestLeakyReLUParameterPatterns:
    """测试参数模式：默认值、边界值、特殊值、不同负斜率"""

    @pytest.mark.leaky_relu
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_default_usage(self, dtype):
        """测试：默认参数使用"""
        x = create_tensor((100, 100), dtype)
        y_gems = leaky_relu(x)  # 默认 negative_slope=0.01
        y_torch = torch.nn.functional.leaky_relu(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.leaky_relu
    @pytest.mark.parametrize("negative_slope", [0.001, 0.01, 0.1, 0.5, 1.0, 2.0])
    def test_various_slopes(self, negative_slope):
        """测试：各种负斜率值"""
        x = create_tensor((100, 100), torch.float32)
        y_gems = leaky_relu(x, negative_slope=negative_slope)
        y_torch = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_edge_case_zero(self, dtype):
        """测试：边界值 - LeakyReLU(0) = 0"""
        x = torch.zeros((10, 10), dtype=dtype, device="cuda")
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)
        # 验证零映射到零
        assert torch.allclose(y_gems, torch.zeros_like(y_gems))

    @pytest.mark.leaky_relu
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_edge_case_positive(self, dtype):
        """测试：边界值 - 正值保持不变"""
        x = torch.ones((10, 10), dtype=dtype, device="cuda")
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)
        # 验证正值不变
        assert torch.allclose(y_gems, x)

    @pytest.mark.leaky_relu
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_special_small_values(self, dtype):
        """测试：特殊值 - 小正数/负数"""
        x = torch.tensor(
            [0.1, 0.01, 0.001, -0.1, -0.01, -0.001], dtype=dtype, device="cuda"
        )
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.leaky_relu
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_special_large_values(self, dtype):
        """测试：特殊值 - 大正数/负数"""
        x = torch.tensor(
            [1e6, 1e8, 1e10, -1e6, -1e8, -1e10], dtype=dtype, device="cuda"
        )
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.leaky_relu
    def test_zero_slope(self):
        """测试：特殊参数 - zero_slope=0（等价于 ReLU）"""
        x = create_tensor((100, 100), torch.float32)
        y_gems = leaky_relu(x, negative_slope=0.0)
        y_torch = torch.nn.functional.leaky_relu(x, negative_slope=0.0)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)
        # 验证所有负值变为 0
        assert torch.all(y_gems[x < 0] == 0)

    @pytest.mark.leaky_relu
    def test_unit_slope(self):
        """测试：特殊参数 - unit_slope=1（等价于恒等函数）"""
        x = create_tensor((100, 100), torch.float32)
        y_gems = leaky_relu(x, negative_slope=1.0)
        y_torch = torch.nn.functional.leaky_relu(x, negative_slope=1.0)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)
        # 验证输出等于输入
        assert_close(y_gems, x, rtol=1e-4, atol=1e-6, dtype=torch.float32)


# ============================================================================
# 5. 功能完整性测试（比赛要求：所有功能分支）
# ============================================================================


class TestLeakyReLUFunctionalCompleteness:
    """测试功能完整性：正值分支、负值分支、零值处理、批处理"""

    @pytest.mark.leaky_relu
    def test_positive_values(self):
        """测试：正值分支 - 输出等于输入"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)
        # 验证正值不变
        assert torch.allclose(y_gems, x)

    @pytest.mark.leaky_relu
    def test_negative_values(self):
        """测试：负值分支 - 输出等于输入乘以负斜率"""
        x = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0], device="cuda")
        slope = 0.1
        y_gems = leaky_relu(x, negative_slope=slope)
        y_torch = torch.nn.functional.leaky_relu(x, negative_slope=slope)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)
        # 验证负值被缩放
        expected_values = x * slope
        assert_close(y_gems, expected_values, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_zero_values(self):
        """测试：零值处理 - 输出为零"""
        x = torch.tensor([0.0, 0.0, 0.0], device="cuda")
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)
        # 验证零保持为零
        assert torch.allclose(y_gems, torch.zeros_like(x))

    @pytest.mark.leaky_relu
    def test_mixed_values(self):
        """测试：混合值 - 正值、负值、零值"""
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device="cuda")
        slope = 0.1
        y_gems = leaky_relu(x, negative_slope=slope)
        y_torch = torch.nn.functional.leaky_relu(x, negative_slope=slope)
        expected_values = torch.tensor([-0.2, -0.1, 0.0, 1.0, 2.0], device="cuda")
        assert_close(y_gems, expected_values, rtol=1e-4, atol=1e-6, dtype=torch.float32)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_batch_processing(self):
        """测试：批量处理（多张量）"""
        tensors = [
            create_tensor((64, 64), torch.float32),
            create_tensor((128, 128), torch.float32),
            create_tensor((256, 256), torch.float32),
        ]
        for x in tensors:
            y_gems = leaky_relu(x)
            y_torch = torch.nn.functional.leaky_relu(x)
            assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_consistency_with_pytorch(self):
        """测试：与 PyTorch 实现的一致性"""
        # 使用随机张量测试多次
        for _ in range(10):
            x = create_tensor((100, 100), torch.float32)
            y_gems = leaky_relu(x)
            y_torch = torch.nn.functional.leaky_relu(x)
            assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_empty_tensor(self):
        """测试：空张量处理"""
        x = torch.randn(0, 10, device="cuda")
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert y_gems.shape == y_torch.shape
        assert y_gems.numel() == 0


# ============================================================================
# 6. 综合覆盖测试（参数化组合）
# ============================================================================


class TestLeakyReLUComprehensive:
    """综合测试：多维度组合覆盖"""

    @pytest.mark.leaky_relu
    @pytest.mark.parametrize("shape", [8, 64 * 64, 256 * 256, 1024 * 1024])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_shape_dtype_combination(self, shape, dtype):
        """测试：形状和数据类型的组合"""
        x = create_tensor((shape,), dtype)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.leaky_relu
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((8, 8), torch.float16),
            ((64, 64), torch.float32),
            ((256, 256), torch.float16),
            ((1024, 1024), torch.float32),
        ],
    )
    def test_typical_use_cases(self, shape, dtype):
        """测试：典型使用场景"""
        x = create_tensor(shape, dtype)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.leaky_relu
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((10,), torch.float32),
            ((10, 10), torch.float32),
            ((10, 10, 10), torch.float32),
            ((2, 3, 4, 5), torch.float32),
            ((2, 2, 2, 2, 2), torch.float32),
        ],
    )
    def test_various_shapes_and_dtypes(self, shape, dtype):
        """测试：各种形状和数据类型的组合"""
        x = create_tensor(shape, dtype)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=dtype)

    @pytest.mark.leaky_relu
    @pytest.mark.parametrize(
        "shape,slope",
        [
            ((100, 100), 0.001),
            ((100, 100), 0.01),
            ((100, 100), 0.1),
            ((100, 100), 1.0),
        ],
    )
    def test_shape_slope_combination(self, shape, slope):
        """测试：形状和负斜率的组合"""
        x = create_tensor(shape, torch.float32)
        y_gems = leaky_relu(x, negative_slope=slope)
        y_torch = torch.nn.functional.leaky_relu(x, negative_slope=slope)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)


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
- [x] 5D 张量：(2, 3, 4, 5, 6)

### 3. 数据类型覆盖 ✓
- [x] torch.float16 (atol=1e-5)
- [x] torch.float32 (atol=1e-6)

### 4. 参数模式覆盖 ✓
- [x] 默认参数：negative_slope=0.01
- [x] 边界值：LeakyReLU(0) = 0, LeakyReLU(1) = 1
- [x] 特殊值：小值（0.001~0.1）
- [x] 特殊值：大值（1e6~1e10）
- [x] 特殊参数：zero_slope=0（等价于 ReLU）
- [x] 特殊参数：unit_slope=1（等价于恒等函数）
- [x] 不同负斜率：0.001, 0.01, 0.1, 0.5, 1.0, 2.0

### 5. 功能完整性覆盖 ✓
- [x] 正值分支：输出 = 输入
- [x] 负值分支：输出 = negative_slope * 输入
- [x] 零值处理：输出 = 0
- [x] 混合值处理：同时包含正、负、零值
- [x] 批量处理
- [x] 空张量处理
- [x] 与 PyTorch 一致性验证

### 6. 测试精度标准 ✓
- [x] rtol = 1e-4 (所有浮点类型)
- [x] atol 根据数据类型变化（符合比赛要求）
- [x] 使用 torch.allclose 验证（比赛标准）

### 7. 数学性质验证 ✓
- [x] 正值不变性：LeakyReLU(x) = x for x >= 0
- [x] 负值缩放：LeakyReLU(x) = negative_slope * x for x < 0
- [x] 零值保持：LeakyReLU(0) = 0
- [x] 线性性：在正负区间都是线性的
"""
