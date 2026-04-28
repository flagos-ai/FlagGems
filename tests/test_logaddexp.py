<<<<<<< feature/FlagGemsOperatorDevelopmentCompetition
"""
Test suite for logaddexp operator.

This test module validates the correctness, precision, and performance
of the logaddexp operator implementation following FlagGems testing conventions.

Test coverage description:
- Input sizes: 1×1, 8×8, 64×64, 256×256, 1024×1024, 4096×4096
- Input dimensions: 1D, 2D, 3D, 4D
- Data types: float16, float32, float64, bfloat16, integer types
- Parameter modes: default values, boundary values, special values (extreme values, zero values, numerical stability)
- Functional completeness: basic computation, type promotion, batch processing, numerical stability

Operator formula: logaddexp(x, y) = log(exp(x) + exp(y))
Numerically stable implementation: max(x, y) + log(1 + exp(-|x - y|))
"""

import pytest
import torch

from flag_gems.ops import logaddexp

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
    torch.float64,
    torch.bfloat16,
]

# Precision standards (competition requirement standards)
# rtol = 1e-4 (all floating point types)
# atol varies based on data type
ATOL_DICT = {
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.float64: 1e-7,
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

    # Use torch.allclose for comparison (competition standard)
    assert torch.allclose(
        actual, expected, rtol=rtol, atol=atol, equal_nan=True
    ), f"Results don't match: max diff={(actual - expected).abs().max().item()}"


def create_tensor(shape, dtype, device="cuda"):
    """Create tensor (for logaddexp testing)"""
    x = torch.randn(shape, dtype=dtype, device=device)
    return x


# ============================================================================
# 1. Input size coverage tests (competition requirement: small, medium, large three categories)
# ============================================================================


class TestLogAddExpInputSize:
    """Test input size coverage"""

    @pytest.mark.logaddexp
    def test_size_very_small(self):
        """Test: 1×1 (extremely small size)"""
        x = torch.tensor([[0.5]], device="cuda")
        y = torch.tensor([[0.3]], device="cuda")
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.logaddexp
    def test_size_small(self):
        """Test: 8×8 (small size)"""
        x = create_tensor((8, 8), torch.float32)
        y = create_tensor((8, 8), torch.float32)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.logaddexp
    def test_size_medium_64(self):
        """Test: 64×64 (medium size)"""
        x = create_tensor((64, 64), torch.float32)
        y = create_tensor((64, 64), torch.float32)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.logaddexp
    def test_size_medium_256(self):
        """Test: 256×256 (medium size)"""
        x = create_tensor((256, 256), torch.float32)
        y = create_tensor((256, 256), torch.float32)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.logaddexp
    def test_size_large_1k(self):
        """Test: 1024×1024 (large size)"""
        x = create_tensor((1024, 1024), torch.float32)
        y = create_tensor((1024, 1024), torch.float32)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.logaddexp
    def test_size_large_4k(self):
        """Test: 4096×4096 (large size)"""
        x = create_tensor((4096, 4096), torch.float32)
        y = create_tensor((4096, 4096), torch.float32)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 2. Input dimension coverage tests (competition requirement: cover all valid dimensions)
# ============================================================================


class TestLogAddExpInputDimensions:
    """Test input dimension coverage: 1D, 2D, 3D, 4D"""

    @pytest.mark.logaddexp
    def test_dim_1d_small(self):
        """Test: 1D tensor (small size)"""
        x = create_tensor((10,), torch.float32)
        y = create_tensor((10,), torch.float32)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.logaddexp
    def test_dim_1d_large(self):
        """Test: 1D tensor (large size)"""
        x = create_tensor((10000,), torch.float32)
        y = create_tensor((10000,), torch.float32)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.logaddexp
    def test_dim_2d(self):
        """Test: 2D tensor"""
        x = create_tensor((100, 100), torch.float32)
        y = create_tensor((100, 100), torch.float32)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.logaddexp
    def test_dim_3d(self):
        """Test: 3D tensor"""
        x = create_tensor((64, 64, 64), torch.float32)
        y = create_tensor((64, 64, 64), torch.float32)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.logaddexp
    def test_dim_4d_batch(self):
        """Test: 4D batch tensor (batch × channel × height × width)"""
        x = create_tensor((16, 3, 128, 128), torch.float32)
        y = create_tensor((16, 3, 128, 128), torch.float32)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 3. Data type coverage tests (competition requirement: at least support float32/float16)
# ============================================================================


class TestLogAddExpDataTypes:
    """Test data type coverage: float16, float32, float64, bfloat16, integer"""

    @pytest.mark.logaddexp
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_all_float_dtypes(self, dtype):
        """Test: all floating point data types"""
        x = create_tensor((256, 256), dtype)
        y = create_tensor((256, 256), dtype)
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        atol = ATOL_DICT[dtype]
        assert_close(z_gems, z_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.logaddexp
    def test_integer_input_not_supported(self):
        """Test: integer input should raise error (consistent with PyTorch behavior)

        PyTorch's logaddexp does not support integer input and raises NotImplementedError.
        Our implementation should maintain consistent behavior.
        """
        x = torch.tensor([1, 10, 100, 1000], device="cuda")
        y = torch.tensor([2, 20, 200, 2000], device="cuda")

        # PyTorch raises NotImplementedError
        with pytest.raises(NotImplementedError, match="not implemented for 'Long'"):
            torch.logaddexp(x, y)

        # Our implementation should also raise the same error
        with pytest.raises((NotImplementedError, TypeError)):
            logaddexp(x, y)


# ============================================================================
# 4. Parameter pattern coverage tests (competition requirement: default values, boundary values,
# special values, numerical stability)
# ============================================================================


class TestLogAddExpParameterPatterns:
    """Test parameter patterns: default values, boundary values, special values, numerical stability"""

    @pytest.mark.logaddexp
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_default_usage(self, dtype):
        """Test: default parameter usage"""
        x = create_tensor((100, 100), dtype)
        y = create_tensor((100, 100), dtype)
        z_gems = logaddexp(x, y)  # Default call
        z_torch = torch.logaddexp(x, y)
        atol = ATOL_DICT[dtype]
        assert_close(z_gems, z_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.logaddexp
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_edge_case_equal_values(self, dtype):
        """Test: boundary value - equal values log(exp(x) + exp(x)) = log(2*exp(x)) = x + log(2)"""
        x = torch.ones((10, 10), dtype=dtype, device="cuda")
        y = torch.ones((10, 10), dtype=dtype, device="cuda")
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        atol = ATOL_DICT[dtype]
        assert_close(z_gems, z_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.logaddexp
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_edge_case_one_value_zero(self, dtype):
        """Test: boundary value - one value is 0"""
        x = torch.zeros((10, 10), dtype=dtype, device="cuda")
        y = torch.ones((10, 10), dtype=dtype, device="cuda")
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        atol = ATOL_DICT[dtype]
        assert_close(z_gems, z_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.logaddexp
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_special_small_values(self, dtype):
        """Test: special values - small negative numbers (critical for numerical stability)"""
        x = torch.tensor([-100.0, -50.0, -10.0], dtype=dtype, device="cuda")
        y = torch.tensor([-90.0, -60.0, -5.0], dtype=dtype, device="cuda")
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        atol = ATOL_DICT[dtype]
        assert_close(z_gems, z_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.logaddexp
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_special_large_values(self, dtype):
        """Test: special values - large positive numbers"""
        x = torch.tensor([100.0, 500.0, 1000.0], dtype=dtype, device="cuda")
        y = torch.tensor([200.0, 600.0, 900.0], dtype=dtype, device="cuda")
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        atol = ATOL_DICT[dtype]
        assert_close(z_gems, z_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.logaddexp
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_special_extreme_values(self, dtype):
        """Test: extreme values - critical numerical stability test"""
        # Test very large values to verify numerical stability
        x = torch.tensor([1000.0, -1000.0], dtype=dtype, device="cuda")
        y = torch.tensor([1001.0, -999.0], dtype=dtype, device="cuda")
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        atol = ATOL_DICT[dtype]
        assert_close(z_gems, z_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.logaddexp
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_numerical_stability(self, dtype):
        """Test: numerical stability - mixed positive and negative values"""
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
        """Test: symmetry property logaddexp(x, y) = logaddexp(y, x)"""
        x = create_tensor((100, 100), dtype)
        y = create_tensor((100, 100), dtype)
        z_gems = logaddexp(x, y)
        z_gems_swapped = logaddexp(y, x)
        assert torch.allclose(z_gems, z_gems_swapped, rtol=0, atol=0)


# ============================================================================
# 5. Functional completeness tests (competition requirement: all functional branches)
# ============================================================================


class TestLogAddExpFunctionalCompleteness:
    """Test functional completeness: basic computation, type promotion, batch processing, symmetry"""

    @pytest.mark.logaddexp
    def test_basic_computation(self):
        """Test: basic element-wise computation"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
        y = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0], device="cuda")
        z_gems = logaddexp(x, y)
        z_torch = torch.logaddexp(x, y)
        assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.logaddexp
    def test_dtype_consistency(self):
        """Test: floating point type consistency"""
        # Verify consistency across different floating point types
        x_f32 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="cuda")
        y_f32 = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32, device="cuda")
        x_f64 = x_f32.to(torch.float64)
        y_f64 = y_f32.to(torch.float64)

        z_f32 = logaddexp(x_f32, y_f32)
        z_f64 = logaddexp(x_f64, y_f64)

        # After conversion should be close (considering precision differences)
        assert torch.allclose(z_f32.to(torch.float64), z_f64, rtol=1e-5, atol=1e-6)

    @pytest.mark.logaddexp
    def test_batch_processing(self):
        """Test: batch processing (multiple tensors)"""
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
        """Test: consistency with PyTorch implementation"""
        # Test multiple times using random tensors
        for _ in range(10):
            x = create_tensor((100, 100), torch.float32)
            y = create_tensor((100, 100), torch.float32)
            z_gems = logaddexp(x, y)
            z_torch = torch.logaddexp(x, y)
            assert_close(z_gems, z_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 6. Comprehensive coverage tests (parameterized combinations)
# ============================================================================


class TestLogAddExpComprehensive:
    """Comprehensive tests: multi-dimensional combination coverage"""

    @pytest.mark.logaddexp
    @pytest.mark.parametrize("shape", [8, 64 * 64, 256 * 256, 1024 * 1024])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_shape_dtype_combination(self, shape, dtype):
        """Test: combinations of shapes and data types"""
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
        """Test: typical use cases"""
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
        """Test: commutative property logaddexp(x, y) = logaddexp(y, x)"""
        x = create_tensor(shape, dtype)
        y = create_tensor(shape, dtype)
        z_xy = logaddexp(x, y)
        z_yx = logaddexp(y, x)
        assert_close(z_xy, z_yx, rtol=0, atol=0, dtype=dtype)
=======
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.logaddexp
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.logaddexp(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = torch.logaddexp(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.logaddexp_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out_buf = torch.empty(shape, dtype=ref_x.dtype, device=ref_x.device)
    ref_out = torch.ops.aten.logaddexp.out(ref_x, ref_y, out=ref_out_buf)

    res_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.logaddexp.out(x, y, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype)
>>>>>>> master
