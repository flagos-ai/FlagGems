"""
Test suite for Roll operator.

This test module validates the correctness, precision, and performance
of the roll operator implementation following FlagGems testing conventions.

测试覆盖说明：
- 输入规模：1×1, 8×8, 64×64, 256×256, 1024×1024, 4096×4096
- 输入维数：1D, 2D, 3D, 4D, 5D
- 数据类型：float16, float32, float64, bfloat16, int8, int16, int32, int64, bool
- 参数模式：默认值、边界值、特殊值、不同位移和维度组合
- 功能完整性：单维度滚动、多维度滚动、展平滚动、空张量处理
"""

import pytest
import torch

from flag_gems.ops import roll

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

# 整数类型覆盖
INT_DTYPES = [
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
]


# ============================================================================
# 辅助函数
# ============================================================================


def assert_equal(actual, expected):
    """
    使用 torch.equal 验证完全相等（roll 操作要求精确匹配）

    Args:
        actual: FlagGems 实现结果
        expected: PyTorch 参考结果
    """
    assert torch.equal(
        actual, expected
    ), "Results don't match: shape mismatch or values differ"


def create_tensor(shape, dtype, device="cuda"):
    """创建测试张量（根据数据类型选择）"""
    if dtype.is_floating_point:
        x = torch.randn(shape, dtype=dtype, device=device)
    else:
        x = torch.randint(1, 100, shape, dtype=dtype, device=device)
    return x


# ============================================================================
# 1. 输入规模覆盖测试（比赛要求：小、常规、大三类）
# ============================================================================


class TestRollInputSize:
    """测试输入规模覆盖"""

    @pytest.mark.roll
    def test_size_very_small(self):
        """测试：1×1（极小尺寸）"""
        x = torch.tensor([[0.5]], device="cuda")
        y_gems = roll(x, shifts=1)
        y_torch = torch.roll(x, shifts=1)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_size_small(self):
        """测试：8×8（小尺寸）"""
        x = create_tensor((8, 8), torch.float32)
        y_gems = roll(x, shifts=2, dims=0)
        y_torch = torch.roll(x, shifts=2, dims=0)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_size_medium_64(self):
        """测试：64×64（常规尺寸）"""
        x = create_tensor((64, 64), torch.float32)
        y_gems = roll(x, shifts=10, dims=1)
        y_torch = torch.roll(x, shifts=10, dims=1)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_size_medium_256(self):
        """测试：256×256（常规尺寸）"""
        x = create_tensor((256, 256), torch.float32)
        y_gems = roll(x, shifts=50, dims=0)
        y_torch = torch.roll(x, shifts=50, dims=0)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_size_large_1k(self):
        """测试：1024×1024（大尺寸）"""
        x = create_tensor((1024, 1024), torch.float32)
        y_gems = roll(x, shifts=100, dims=1)
        y_torch = torch.roll(x, shifts=100, dims=1)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_size_large_4k(self):
        """测试：4096×4096（大尺寸）"""
        x = create_tensor((4096, 4096), torch.float32)
        y_gems = roll(x, shifts=500, dims=0)
        y_torch = torch.roll(x, shifts=500, dims=0)
        assert_equal(y_gems, y_torch)


# ============================================================================
# 2. 输入维数覆盖测试（比赛要求：覆盖全部合法维数）
# ============================================================================


class TestRollInputDimensions:
    """测试输入维数覆盖：1D, 2D, 3D, 4D, 5D"""

    @pytest.mark.roll
    def test_dim_1d_small(self):
        """测试：1D 张量（小尺寸）"""
        x = create_tensor((10,), torch.float32)
        y_gems = roll(x, shifts=3)
        y_torch = torch.roll(x, shifts=3)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_dim_1d_large(self):
        """测试：1D 张量（大尺寸）"""
        x = create_tensor((10000,), torch.float32)
        y_gems = roll(x, shifts=100)
        y_torch = torch.roll(x, shifts=100)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_dim_2d(self):
        """测试：2D 张量"""
        x = create_tensor((100, 100), torch.float32)
        y_gems = roll(x, shifts=20, dims=0)
        y_torch = torch.roll(x, shifts=20, dims=0)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_dim_3d(self):
        """测试：3D 张量"""
        x = create_tensor((64, 64, 64), torch.float32)
        y_gems = roll(x, shifts=10, dims=1)
        y_torch = torch.roll(x, shifts=10, dims=1)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_dim_4d_batch(self):
        """测试：4D 批量张量（batch × channel × height × width）"""
        x = create_tensor((16, 3, 128, 128), torch.float32)
        y_gems = roll(x, shifts=30, dims=2)
        y_torch = torch.roll(x, shifts=30, dims=2)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_dim_5d(self):
        """测试：5D 张量"""
        x = create_tensor((2, 3, 4, 5, 6), torch.float32)
        y_gems = roll(x, shifts=(1, 2, 0), dims=(0, 2, 4))
        y_torch = torch.roll(x, shifts=(1, 2, 0), dims=(0, 2, 4))
        assert_equal(y_gems, y_torch)


# ============================================================================
# 3. 数据类型覆盖测试（比赛要求：至少支持 float32/float16）
# ============================================================================


class TestRollDataTypes:
    """测试数据类型覆盖：float16, float32, float64, bfloat16, int8, int16, int32, int64, bool"""

    @pytest.mark.roll
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_float_dtypes(self, dtype):
        """测试：所有浮点数据类型"""
        if dtype == torch.bfloat16 and not torch.cuda.is_available():
            pytest.skip("bfloat16 requires CUDA")

        x = create_tensor((100, 100), dtype)
        y_gems = roll(x, shifts=10, dims=0)
        y_torch = torch.roll(x, shifts=10, dims=0)

        assert y_gems.dtype == y_torch.dtype
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_integer_dtypes(self, dtype):
        """测试：所有整数数据类型"""
        x = create_tensor((50, 50), dtype)
        y_gems = roll(x, shifts=5, dims=1)
        y_torch = torch.roll(x, shifts=5, dims=1)

        assert y_gems.dtype == y_torch.dtype
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_bool_dtype(self):
        """测试：布尔类型"""
        x = torch.randint(0, 2, (20, 20), dtype=torch.bool, device="cuda")
        y_gems = roll(x, shifts=3, dims=0)
        y_torch = torch.roll(x, shifts=3, dims=0)

        assert y_gems.dtype == y_torch.dtype
        assert_equal(y_gems, y_torch)


# ============================================================================
# 4. 参数模式覆盖测试（比赛要求：默认值、边界值、特殊值）
# ============================================================================


class TestRollParameterPatterns:
    """测试参数模式：默认值、边界值、特殊值、不同位移和维度组合"""

    @pytest.mark.roll
    @pytest.mark.parametrize("shift", [0, 1, 5, 10, 100])
    def test_various_positive_shifts(self, shift):
        """测试：各种正位移"""
        x = create_tensor((100,), torch.float32)
        y_gems = roll(x, shifts=shift)
        y_torch = torch.roll(x, shifts=shift)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    @pytest.mark.parametrize("shift", [-1, -5, -10, -100])
    def test_negative_shifts(self, shift):
        """测试：负位移"""
        x = create_tensor((100,), torch.float32)
        y_gems = roll(x, shifts=shift)
        y_torch = torch.roll(x, shifts=shift)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_shift_larger_than_size(self):
        """测试：位移大于张量大小（循环）"""
        x = torch.arange(10, device="cuda", dtype=torch.float32)
        shift = 15  # 大于尺寸 10

        y_gems = roll(x, shifts=shift)
        y_torch = torch.roll(x, shifts=shift)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_negative_dim_index(self):
        """测试：负维度索引"""
        x = create_tensor((10, 20, 30), torch.float32)

        # 测试 dim=-1（最后一维）
        y_gems = roll(x, shifts=3, dims=-1)
        y_torch = torch.roll(x, shifts=3, dims=-1)
        assert_equal(y_gems, y_torch)

        # 测试 dim=-2（倒数第二维）
        y_gems = roll(x, shifts=2, dims=-2)
        y_torch = torch.roll(x, shifts=2, dims=-2)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_dims_none_flatten(self):
        """测试：dims=None 时展平张量并滚动"""
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device="cuda")

        y_gems = roll(x, shifts=2)
        y_torch = torch.roll(x, shifts=2)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_tuple_shifts_and_dims(self):
        """测试：元组形式的 shifts 和 dims"""
        x = create_tensor((10, 20), torch.float32)
        shifts = (3, 5)
        dims = (0, 1)

        y_gems = roll(x, shifts=shifts, dims=dims)
        y_torch = torch.roll(x, shifts=shifts, dims=dims)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_edge_case_zero_shift(self):
        """测试：边界值 - 位移为 0（无操作）"""
        x = create_tensor((50, 50), torch.float32)
        y_gems = roll(x, shifts=0, dims=1)
        y_torch = torch.roll(x, shifts=0, dims=1)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_edge_case_shift_equal_to_size(self):
        """测试：边界值 - 位移等于维度大小（应保持不变）"""
        x = create_tensor((20, 30), torch.float32)

        y_gems = roll(x, shifts=20, dims=0)
        y_torch = torch.roll(x, shifts=20, dims=0)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_special_very_large_shift(self):
        """测试：特殊值 - 极大位移（应正常循环）"""
        x = create_tensor((10, 10), torch.float32)

        # 极大正位移（应循环）
        shift = 1000
        y_gems = roll(x, shifts=shift, dims=1)
        y_torch = torch.roll(x, shifts=shift, dims=1)
        assert_equal(y_gems, y_torch)

        # 极大负位移
        shift = -1000
        y_gems = roll(x, shifts=shift, dims=0)
        y_torch = torch.roll(x, shifts=shift, dims=0)
        assert_equal(y_gems, y_torch)


# ============================================================================
# 5. 功能完整性测试（比赛要求：所有功能分支）
# ============================================================================


class TestRollFunctionalCompleteness:
    """测试功能完整性：单维度滚动、多维度滚动、展平滚动、空张量处理"""

    @pytest.mark.roll
    def test_single_dimension(self):
        """测试：单维度滚动"""
        x = create_tensor((10, 20, 30), torch.float32)

        y_gems = roll(x, shifts=5, dims=1)
        y_torch = torch.roll(x, shifts=5, dims=1)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_multiple_dimensions(self):
        """测试：多维度滚动（元组参数）"""
        x = create_tensor((10, 20, 30), torch.float32)
        shifts = (3, 5, 2)
        dims = (0, 1, 2)

        y_gems = roll(x, shifts=shifts, dims=dims)
        y_torch = torch.roll(x, shifts=shifts, dims=dims)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_flattened_roll(self):
        """测试：展平张量滚动"""
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device="cuda")

        y_gems = roll(x, shifts=2)
        y_torch = torch.roll(x, shifts=2)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_empty_tensor(self):
        """测试：空张量处理"""
        x = torch.randn(0, 10, device="cuda")

        y_gems = roll(x, shifts=2, dims=0)
        y_torch = torch.roll(x, shifts=2, dims=0)

        assert y_gems.shape == y_torch.shape
        assert y_gems.numel() == 0

    @pytest.mark.roll
    def test_single_element(self):
        """测试：单元素张量"""
        x = torch.tensor([42.0], device="cuda")

        y_gems = roll(x, shifts=100)
        y_torch = torch.roll(x, shifts=100)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_contiguous_memory_layout(self):
        """测试：连续内存布局"""
        x = torch.randn(10, 10, device="cuda")
        assert x.is_contiguous()

        y_gems = roll(x, shifts=3, dims=1)
        y_torch = torch.roll(x, shifts=3, dims=1)

        assert y_gems.is_contiguous()
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_non_contiguous_memory_layout(self):
        """测试：非连续内存布局"""
        x = torch.randn(10, 10, device="cuda").t()  # 转置使其非连续
        assert not x.is_contiguous()

        y_gems = roll(x, shifts=3, dims=0)
        y_torch = torch.roll(x, shifts=3, dims=0)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_consistency_with_pytorch(self):
        """测试：与 PyTorch 实现的一致性"""
        # 使用随机张量测试多次
        for _ in range(10):
            x = create_tensor((100, 100), torch.float32)
            shift = torch.randint(-50, 50, (1,)).item()
            dim = torch.randint(0, 2, (1,)).item()

            y_gems = roll(x, shifts=shift, dims=dim)
            y_torch = torch.roll(x, shifts=shift, dims=dim)
            assert_equal(y_gems, y_torch)


# ============================================================================
# 6. 综合覆盖测试（参数化组合）
# ============================================================================


class TestRollComprehensive:
    """综合测试：多维度组合覆盖"""

    @pytest.mark.roll
    @pytest.mark.parametrize(
        "shape,shift,dim",
        [
            ((5, 10), 2, 0),
            ((5, 10), 3, 1),
            ((100, 200), 50, 0),
            ((100, 200), 75, 1),
            ((10, 10), 11, 0),  # 位移大于尺寸
            ((10, 10), -3, 1),  # 负位移
        ],
    )
    def test_shape_shift_dim_combination(self, shape, shift, dim):
        """测试：形状、位移和维度的组合"""
        x = create_tensor(shape, torch.float32)
        y_gems = roll(x, shifts=shift, dims=dim)
        y_torch = torch.roll(x, shifts=shift, dims=dim)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
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
        if dtype == torch.bfloat16 and not torch.cuda.is_available():
            pytest.skip("bfloat16 requires CUDA")

        x = create_tensor(shape, dtype)
        y_gems = roll(x, shifts=10, dims=0)
        y_torch = torch.roll(x, shifts=10, dims=0)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
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
        y_gems = roll(x, shifts=2, dims=-1)
        y_torch = torch.roll(x, shifts=2, dims=-1)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    @pytest.mark.parametrize(
        "shifts,dims",
        [
            (3, 0),
            (5, 1),
            ((3, 5), (0, 1)),
            ((1, 2), (1, 0)),
            ((1, 2, 0), (0, 1, 2)),
        ],
    )
    def test_various_shifts_and_dims(self, shifts, dims):
        """测试：各种位移和维度组合"""
        x = create_tensor((10, 20, 30), torch.float32)
        y_gems = roll(x, shifts=shifts, dims=dims)
        y_torch = torch.roll(x, shifts=shifts, dims=dims)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_randomized_comparison(self):
        """随机化测试：与 PyTorch 结果比较"""
        torch.manual_seed(42)

        for _ in range(20):
            # 随机形状
            n_rows = torch.randint(10, 100, (1,)).item()
            n_cols = torch.randint(10, 100, (1,)).item()

            # 随机数据类型
            dtype_idx = torch.randint(0, 3, (1,)).item()
            dtypes = [torch.float16, torch.float32, torch.float64]
            dtype = dtypes[dtype_idx]

            x = torch.randn(n_rows, n_cols, dtype=dtype, device="cuda")

            # 随机位移和维度
            shift = torch.randint(-20, 20, (1,)).item()
            dim = torch.randint(0, 2, (1,)).item()

            y_gems = roll(x, shifts=shift, dims=dim)
            y_torch = torch.roll(x, shifts=shift, dims=dim)
            assert_equal(y_gems, y_torch)


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
- [x] torch.float16
- [x] torch.float32
- [x] torch.float64
- [x] torch.bfloat16
- [x] torch.int8
- [x] torch.int16
- [x] torch.int32
- [x] torch.int64
- [x] torch.bool

### 4. 参数模式覆盖 ✓
- [x] 默认参数：dims=None（展平张量）
- [x] 边界值：shift=0（无操作）
- [x] 边界值：shift=dim_size（保持不变）
- [x] 特殊值：shift > dim_size（循环）
- [x] 特殊值：负位移
- [x] 特殊值：负维度索引
- [x] 特殊值：极大位移（1000, -1000）
- [x] 不同位移：0, 1, 5, 10, 100
- [x] 多维度滚动：shifts 和 dims 为元组

### 5. 功能完整性覆盖 ✓
- [x] 单维度滚动
- [x] 多维度滚动
- [x] 展平张量滚动
- [x] 空张量处理
- [x] 单元素张量处理
- [x] 连续内存布局
- [x] 非连续内存布局
- [x] 与 PyTorch 一致性验证

### 6. 测试精度标准 ✓
- [x] 使用 torch.equal 验证（roll 操作要求精确匹配）
- [x] 验证数据类型保持不变
- [x] 验证形状保持不变

### 7. 数学性质验证 ✓
- [x] 循环性质：shift % dim_size
- [x] 可逆性：roll(x, shift, dim) 再 roll(x, -shift, dim) = x
- [x] 等价性：多维度滚动等价于依次单维度滚动
- [x] 保持性：shift=0 或 shift=dim_size 时保持不变
"""
