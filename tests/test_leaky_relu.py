"""
Test suite for LeakyReLU operator.

This test module validates the correctness, precision, and performance
of the LeakyReLU operator implementation following FlagGems testing conventions.

Test coverage description:
- Input sizes: 1×1, 8×8, 64×64, 256×256, 1024×1024, 4096×4096
- Input dimensions: 1D, 2D, 3D, 4D, 5D
- Data types: float16, float32
- Parameter modes: default values, boundary values, special values, different negative slopes
- Functional completeness: positive branch, negative branch, zero value handling, batch processing
"""

import pytest
import torch

from flag_gems.ops import leaky_relu

# ============================================================================
# Test data definitions (following competition requirements)
# ============================================================================

# Input size coverage (competition requirement: small, medium, large sizes)
POINTWISE_SHAPES = [
    8,  # small size
    64,  # small size
    64 * 64,  # medium size
    256 * 256,  # medium size
    1024 * 1024,  # large size
    4096 * 4096,  # large size
]

# Data type coverage (competition requirement: at least support float32/float16)
FLOAT_DTYPES = [
    torch.float16,
    torch.float32,
]

# Precision standards (competition requirement standards)
# rtol = 1e-4 (all floating point types)
# atol varies based on data type
ATOL_DICT = {
    torch.float16: 1e-5,
    torch.float32: 1e-6,
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
        atol = ATOL_DICT.get(dtype, 1e-6)

    # Use torch.allclose for comparison (competition standard)
    assert torch.allclose(
        actual, expected, rtol=rtol, atol=atol, equal_nan=True
    ), f"Results don't match: max diff={(actual - expected).abs().max().item()}"


def create_tensor(shape, dtype, device="cuda"):
    """Create test tensor (including positive and negative values)"""
    x = torch.randn(shape, dtype=dtype, device=device)
    return x


# ============================================================================
# 1. Input size coverage tests (competition requirement: small, medium, large three categories)
# ============================================================================


class TestLeakyReLUInputSize:
    """Test input size coverage"""

    @pytest.mark.leaky_relu
    def test_size_very_small(self):
        """Test: 1×1 (extremely small size)"""
        x = torch.tensor([[0.5]], device="cuda")
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_size_small(self):
        """Test: 8×8 (small size)"""
        x = create_tensor((8, 8), torch.float32)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_size_medium_64(self):
        """Test: 64×64 (medium size)"""
        x = create_tensor((64, 64), torch.float32)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_size_medium_256(self):
        """Test: 256×256 (medium size)"""
        x = create_tensor((256, 256), torch.float32)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_size_large_1k(self):
        """Test: 1024×1024 (large size)"""
        x = create_tensor((1024, 1024), torch.float32)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_size_large_4k(self):
        """Test: 4096×4096 (large size)"""
        x = create_tensor((4096, 4096), torch.float32)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)


# ============================================================================
# 2. Input dimension coverage tests (competition requirement: cover all valid dimensions)
# ============================================================================


class TestLeakyReLUInputDimensions:
    """Test input dimension coverage: 1D, 2D, 3D, 4D, 5D"""

    @pytest.mark.leaky_relu
    def test_dim_1d_small(self):
        """Test: 1D tensor (small size)"""
        x = create_tensor((10,), torch.float32)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_dim_1d_large(self):
        """Test: 1D tensor (large size)"""
        x = create_tensor((10000,), torch.float32)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_dim_2d(self):
        """Test: 2D tensor"""
        x = create_tensor((100, 100), torch.float32)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_dim_3d(self):
        """Test: 3D tensor"""
        x = create_tensor((64, 64, 64), torch.float32)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_dim_4d_batch(self):
        """Test: 4D batch tensor (batch × channel × height × width)"""
        x = create_tensor((16, 3, 128, 128), torch.float32)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_dim_5d(self):
        """Test: 5D tensor"""
        x = create_tensor((2, 3, 4, 5, 6), torch.float32)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)


# ============================================================================
# 3. Data type coverage tests (competition requirement: at least support float32/float16)
# ============================================================================


class TestLeakyReLUDataTypes:
    """Test data type coverage: float16, float32"""

    @pytest.mark.leaky_relu
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_all_float_dtypes(self, dtype):
        """Test: all floating point data types"""
        x = create_tensor((256, 256), dtype)
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)


# ============================================================================
# 4. Parameter pattern coverage tests (competition requirement: default values, boundary values, special values)
# ============================================================================


class TestLeakyReLUParameterPatterns:
    """Test parameter patterns: default values, boundary values, special values, different negative slopes"""

    @pytest.mark.leaky_relu
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_default_usage(self, dtype):
        """Test: default parameter usage"""
        x = create_tensor((100, 100), dtype)
        y_gems = leaky_relu(x)  # default negative_slope=0.01
        y_torch = torch.nn.functional.leaky_relu(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.leaky_relu
    @pytest.mark.parametrize("negative_slope", [0.001, 0.01, 0.1, 0.5, 1.0, 2.0])
    def test_various_slopes(self, negative_slope):
        """Test: various negative slope values"""
        x = create_tensor((100, 100), torch.float32)
        y_gems = leaky_relu(x, negative_slope=negative_slope)
        y_torch = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_edge_case_zero(self, dtype):
        """Test: boundary value - LeakyReLU(0) = 0"""
        x = torch.zeros((10, 10), dtype=dtype, device="cuda")
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)
        # Verify zero maps to zero
        assert torch.allclose(y_gems, torch.zeros_like(y_gems))

    @pytest.mark.leaky_relu
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_edge_case_positive(self, dtype):
        """Test: boundary value - positive values remain unchanged"""
        x = torch.ones((10, 10), dtype=dtype, device="cuda")
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)
        # Verify positive values unchanged
        assert torch.allclose(y_gems, x)

    @pytest.mark.leaky_relu
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_special_small_values(self, dtype):
        """Test: special values - small positive/negative numbers"""
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
        """Test: special values - large positive/negative numbers"""
        x = torch.tensor(
            [1e6, 1e8, 1e10, -1e6, -1e8, -1e10], dtype=dtype, device="cuda"
        )
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.leaky_relu
    def test_zero_slope(self):
        """Test: special parameter - zero_slope=0 (equivalent to ReLU)"""
        x = create_tensor((100, 100), torch.float32)
        y_gems = leaky_relu(x, negative_slope=0.0)
        y_torch = torch.nn.functional.leaky_relu(x, negative_slope=0.0)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)
        # Verify all negative values become 0
        assert torch.all(y_gems[x < 0] == 0)

    @pytest.mark.leaky_relu
    def test_unit_slope(self):
        """Test: special parameter - unit_slope=1 (equivalent to identity function)"""
        x = create_tensor((100, 100), torch.float32)
        y_gems = leaky_relu(x, negative_slope=1.0)
        y_torch = torch.nn.functional.leaky_relu(x, negative_slope=1.0)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)
        # Verify output equals input
        assert_close(y_gems, x, rtol=1e-4, atol=1e-6, dtype=torch.float32)


# ============================================================================
# 5. Functional completeness tests (competition requirement: all functional branches)
# ============================================================================


class TestLeakyReLUFunctionalCompleteness:
    """Test functional completeness: positive branch, negative branch, zero value handling, batch processing"""

    @pytest.mark.leaky_relu
    def test_positive_values(self):
        """Test: positive branch - output equals input"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)
        # Verify positive values unchanged
        assert torch.allclose(y_gems, x)

    @pytest.mark.leaky_relu
    def test_negative_values(self):
        """Test: negative branch - output equals input multiplied by negative slope"""
        x = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0], device="cuda")
        slope = 0.1
        y_gems = leaky_relu(x, negative_slope=slope)
        y_torch = torch.nn.functional.leaky_relu(x, negative_slope=slope)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)
        # Verify negative values are scaled
        expected_values = x * slope
        assert_close(y_gems, expected_values, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_zero_values(self):
        """Test: zero value handling - output is zero"""
        x = torch.tensor([0.0, 0.0, 0.0], device="cuda")
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)
        # Verify zero remains zero
        assert torch.allclose(y_gems, torch.zeros_like(x))

    @pytest.mark.leaky_relu
    def test_mixed_values(self):
        """Test: mixed values - positive, negative, zero values"""
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device="cuda")
        slope = 0.1
        y_gems = leaky_relu(x, negative_slope=slope)
        y_torch = torch.nn.functional.leaky_relu(x, negative_slope=slope)
        expected_values = torch.tensor([-0.2, -0.1, 0.0, 1.0, 2.0], device="cuda")
        assert_close(y_gems, expected_values, rtol=1e-4, atol=1e-6, dtype=torch.float32)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_batch_processing(self):
        """Test: batch processing (multiple tensors)"""
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
        """Test: consistency with PyTorch implementation"""
        # Test multiple times using random tensors
        for _ in range(10):
            x = create_tensor((100, 100), torch.float32)
            y_gems = leaky_relu(x)
            y_torch = torch.nn.functional.leaky_relu(x)
            assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)

    @pytest.mark.leaky_relu
    def test_empty_tensor(self):
        """Test: empty tensor handling"""
        x = torch.randn(0, 10, device="cuda")
        y_gems = leaky_relu(x)
        y_torch = torch.nn.functional.leaky_relu(x)
        assert y_gems.shape == y_torch.shape
        assert y_gems.numel() == 0


# ============================================================================
# 6. Comprehensive coverage tests (parameterized combinations)
# ============================================================================


class TestLeakyReLUComprehensive:
    """Comprehensive tests: multi-dimensional combination coverage"""

    @pytest.mark.leaky_relu
    @pytest.mark.parametrize("shape", [8, 64 * 64, 256 * 256, 1024 * 1024])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_shape_dtype_combination(self, shape, dtype):
        """Test: combinations of shapes and data types"""
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
        """Test: typical use cases"""
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
        """Test: various combinations of shapes and data types"""
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
        """Test: combinations of shapes and negative slopes"""
        x = create_tensor(shape, torch.float32)
        y_gems = leaky_relu(x, negative_slope=slope)
        y_torch = torch.nn.functional.leaky_relu(x, negative_slope=slope)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1e-6, dtype=torch.float32)
