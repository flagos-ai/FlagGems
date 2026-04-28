<<<<<<< feature/FlagGemsOperatorDevelopmentCompetition
"""
Test suite for log10 operator.

This test module validates the correctness, precision, and performance
of the log10 operator implementation following FlagGems testing conventions.

Test coverage description:
- Input sizes: 1×1, 8×8, 64×64, 256×256, 1024×1024, 4096×4096
- Input dimensions: 1D, 2D, 3D, 4D
- Data types: float16, float32, float64, bfloat16, integer types
- Parameter modes: default values, boundary values, special values
- Functional completeness: basic computation, type promotion, batch processing
"""

import pytest
import torch

from flag_gems.ops import log10

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


def create_positive_tensor(shape, dtype, device="cuda"):
    """Create positive tensor (for log10 testing)"""
    x = torch.randn(shape, dtype=dtype, device=device)
    return torch.abs(x) + 0.1  # Ensure all values > 0


# ============================================================================
# 1. Input size coverage tests (competition requirement: small, medium, large three categories)
# ============================================================================


class TestLog10InputSize:
    """Test input size coverage"""

    @pytest.mark.log10
    def test_size_very_small(self):
        """Test: 1×1 (extremely small size)"""
        x = torch.tensor([[0.5]], device="cuda")
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.log10
    def test_size_small(self):
        """Test: 8×8 (small size)"""
        x = create_positive_tensor((8, 8), torch.float32)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.log10
    def test_size_medium_64(self):
        """Test: 64×64 (medium size)"""
        x = create_positive_tensor((64, 64), torch.float32)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.log10
    def test_size_medium_256(self):
        """Test: 256×256 (medium size)"""
        x = create_positive_tensor((256, 256), torch.float32)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.log10
    def test_size_large_1k(self):
        """Test: 1024×1024 (large size)"""
        x = create_positive_tensor((1024, 1024), torch.float32)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.log10
    def test_size_large_4k(self):
        """Test: 4096×4096 (large size)"""
        x = create_positive_tensor((4096, 4096), torch.float32)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 2. Input dimension coverage tests (competition requirement: cover all valid dimensions)
# ============================================================================


class TestLog10InputDimensions:
    """Test input dimension coverage: 1D, 2D, 3D, 4D"""

    @pytest.mark.log10
    def test_dim_1d_small(self):
        """Test: 1D tensor (small size)"""
        x = create_positive_tensor((10,), torch.float32)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.log10
    def test_dim_1d_large(self):
        """Test: 1D tensor (large size)"""
        x = create_positive_tensor((10000,), torch.float32)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.log10
    def test_dim_2d(self):
        """Test: 2D tensor"""
        x = create_positive_tensor((100, 100), torch.float32)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.log10
    def test_dim_3d(self):
        """Test: 3D tensor"""
        x = create_positive_tensor((64, 64, 64), torch.float32)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.log10
    def test_dim_4d_batch(self):
        """Test: 4D batch tensor (batch × channel × height × width)"""
        x = create_positive_tensor((16, 3, 128, 128), torch.float32)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 3. Data type coverage tests (competition requirement: at least support float32/float16)
# ============================================================================


class TestLog10DataTypes:
    """Test data type coverage: float16, float32, float64, bfloat16, integer"""

    @pytest.mark.log10
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_all_float_dtypes(self, dtype):
        """Test: all floating point data types"""
        x = create_positive_tensor((256, 256), dtype)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.log10
    def test_integer_input(self):
        """Test: integer input (should automatically promote to float32, consistent with PyTorch behavior)"""
        x = torch.tensor([1, 10, 100, 1000], device="cuda")
        y_gems = log10(x)
        y_torch = torch.log10(x)
        # Verify output type is float32
        assert y_gems.dtype == torch.float32
        assert y_torch.dtype == torch.float32
        # Integer computation should be completely precise
        assert torch.allclose(y_gems, y_torch, rtol=0, atol=0)
        # Verify values are correct
        expected = torch.tensor([0.0, 1.0, 2.0, 3.0], device="cuda")
        assert torch.allclose(y_gems, expected, rtol=0, atol=0)

    @pytest.mark.log10
    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_integer_dtypes(self, dtype):
        """Test: various integer types (should automatically promote to float32)"""
        x = torch.tensor([1, 10, 100, 1000], dtype=dtype, device="cuda")
        y_gems = log10(x)
        y_torch = torch.log10(x)
        # Verify output type is float32
        assert y_gems.dtype == torch.float32
        assert y_torch.dtype == torch.float32
        # Integer computation should be completely precise
        assert torch.allclose(y_gems, y_torch, rtol=0, atol=0)


# ============================================================================
# 4. Parameter pattern coverage tests (competition requirement: default values, boundary values, special values)
# ============================================================================


class TestLog10ParameterPatterns:
    """Test parameter patterns: default values, boundary values, special values"""

    @pytest.mark.log10
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_default_usage(self, dtype):
        """Test: default parameter usage"""
        x = create_positive_tensor((100, 100), dtype)
        y_gems = log10(x)  # Default call
        y_torch = torch.log10(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.log10
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_edge_case_unit_values(self, dtype):
        """Test: boundary value - log10(1.0) = 0"""
        x = torch.ones((10, 10), dtype=dtype, device="cuda")
        y_gems = log10(x)
        y_torch = torch.log10(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.log10
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_edge_case_exact_powers(self, dtype):
        """Test: boundary value - integer powers of 10 (exact results)"""
        x = torch.tensor([1.0, 10.0, 100.0, 1000.0], dtype=dtype, device="cuda")
        y_gems = log10(x)
        y_torch = torch.log10(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.log10
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_special_small_values(self, dtype):
        """Test: special values - small positive numbers"""
        x = torch.tensor([0.1, 0.01, 0.001], dtype=dtype, device="cuda")
        y_gems = log10(x)
        y_torch = torch.log10(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.log10
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_special_large_values(self, dtype):
        """Test: special values - large positive numbers"""
        x = torch.tensor([1e6, 1e8, 1e10], dtype=dtype, device="cuda")
        y_gems = log10(x)
        y_torch = torch.log10(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.log10
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_special_zero_handling(self, dtype):
        """Test: special values - zero value (should produce -inf)"""
        x = torch.tensor([0.0], dtype=dtype, device="cuda")
        y_gems = log10(x)
        y_torch = torch.log10(x)
        # Log of zero should be -inf
        assert torch.isinf(y_gems[0]) and y_gems[0] < 0
        assert torch.isinf(y_torch[0]) and y_torch[0] < 0

    @pytest.mark.log10
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_special_negative_handling(self, dtype):
        """Test: special values - negative numbers (should produce nan)"""
        x = torch.tensor([-1.0, -10.0], dtype=dtype, device="cuda")
        y_gems = log10(x)
        y_torch = torch.log10(x)
        # Log of negative numbers should be nan
        assert torch.isnan(y_gems[0])
        assert torch.isnan(y_torch[0])


# ============================================================================
# 5. Functional completeness tests (competition requirement: all functional branches)
# ============================================================================


class TestLog10FunctionalCompleteness:
    """Test functional completeness: basic computation, type promotion, batch processing"""

    @pytest.mark.log10
    def test_basic_computation(self):
        """Test: basic element-wise computation"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
        y_gems = log10(x)
        y_torch = torch.log10(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.log10
    def test_type_promotion_int_to_float(self):
        """Test: type promotion - integer to floating point"""
        x_int = torch.tensor([1, 10, 100], device="cuda")
        x_float = x_int.float()
        y_int = log10(x_int)
        y_float = log10(x_float)
        # Should get the same result
        assert_close(y_int, y_float, rtol=1e-4, atol=1e-5, dtype=torch.float32)

    @pytest.mark.log10
    def test_batch_processing(self):
        """Test: batch processing (multiple tensors)"""
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
        """Test: consistency with PyTorch implementation"""
        # Test multiple times using random tensors
        for _ in range(10):
            x = create_positive_tensor((100, 100), torch.float32)
            y_gems = log10(x)
            y_torch = torch.log10(x)
            assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 6. Comprehensive coverage tests (parameterized combinations)
# ============================================================================


class TestLog10Comprehensive:
    """Comprehensive tests: multi-dimensional combination coverage"""

    @pytest.mark.log10
    @pytest.mark.parametrize("shape", [8, 64 * 64, 256 * 256, 1024 * 1024])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_shape_dtype_combination(self, shape, dtype):
        """Test: combinations of shapes and data types"""
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
        """Test: typical use cases"""
        x = create_positive_tensor(shape, dtype)
        y_gems = log10(x)
        y_torch = torch.log10(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)
=======
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.log10
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.log10_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_out(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.log10(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.empty_like(inp)
        torch.log10(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_special_values(dtype):
    inp = torch.tensor(
        [0.0, -0.0, 1.0, 10.0, -1.0, float("inf"), float("-inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_empty(dtype):
    shapes = ((0,), (4, 0), (2, 0, 3))
    for shape in shapes:
        inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp, True)

        ref_out = torch.log10(ref_inp)
        with flag_gems.use_gems():
            res_out = torch.log10(inp)

        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_noncontiguous(dtype):
    inp = torch.rand((32, 64), dtype=dtype, device=flag_gems.device).transpose(0, 1)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_log10_int_promotes_to_float(dtype):
    inp = torch.randint(1, 100, (128, 64), dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    utils.gems_assert_close(res_out, ref_out, torch.float32)
>>>>>>> master
