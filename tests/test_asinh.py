<<<<<<< feature/FlagGemsOperatorDevelopmentCompetition
"""
Test suite for Asinh operator.

This test module validates the correctness, precision, and performance
of the asinh operator implementation following FlagGems testing conventions.

Test coverage description:
- Input sizes: 1×1, 8×8, 64×64, 256×256, 1024×1024, 4096×4096
- Input dimensions: 1D, 2D, 3D, 4D
- Data types: float16, float32, float64, bfloat16
- Parameter modes: default values, boundary values, special values
- Functional completeness: basic computation, odd function property, batch processing
"""

import pytest
import torch

from flag_gems.ops import asinh

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
    """Create test tensor (including positive and negative values)"""
    x = torch.randn(shape, dtype=dtype, device=device) * 10
    return x


# ============================================================================
# 1. Input size coverage tests (competition requirement: small, medium, large three categories)
# ============================================================================


class TestAsinhInputSize:
    """Test input size coverage"""

    @pytest.mark.asinh
    def test_size_very_small(self):
        """Test: 1×1 (extremely small size)"""
        x = torch.tensor([[0.5]], device="cuda")
        y_gems = asinh(x)
        y_torch = torch.asinh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.asinh
    def test_size_small(self):
        """Test: 8×8 (small size)"""
        x = create_tensor((8, 8), torch.float32)
        y_gems = asinh(x)
        y_torch = torch.asinh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.asinh
    def test_size_medium_64(self):
        """Test: 64×64 (medium size)"""
        x = create_tensor((64, 64), torch.float32)
        y_gems = asinh(x)
        y_torch = torch.asinh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.asinh
    def test_size_medium_256(self):
        """Test: 256×256 (medium size)"""
        x = create_tensor((256, 256), torch.float32)
        y_gems = asinh(x)
        y_torch = torch.asinh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.asinh
    def test_size_large_1k(self):
        """Test: 1024×1024 (large size)"""
        x = create_tensor((1024, 1024), torch.float32)
        y_gems = asinh(x)
        y_torch = torch.asinh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.asinh
    def test_size_large_4k(self):
        """Test: 4096×4096 (large size)"""
        x = create_tensor((4096, 4096), torch.float32)
        y_gems = asinh(x)
        y_torch = torch.asinh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 2. Input dimension coverage tests (competition requirement: cover all valid dimensions)
# ============================================================================


class TestAsinhInputDimensions:
    """Test input dimension coverage: 1D, 2D, 3D, 4D"""

    @pytest.mark.asinh
    def test_dim_1d_small(self):
        """Test: 1D tensor (small size)"""
        x = create_tensor((10,), torch.float32)
        y_gems = asinh(x)
        y_torch = torch.asinh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.asinh
    def test_dim_1d_large(self):
        """Test: 1D tensor (large size)"""
        x = create_tensor((10000,), torch.float32)
        y_gems = asinh(x)
        y_torch = torch.asinh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.asinh
    def test_dim_2d(self):
        """Test: 2D tensor"""
        x = create_tensor((100, 100), torch.float32)
        y_gems = asinh(x)
        y_torch = torch.asinh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.asinh
    def test_dim_3d(self):
        """Test: 3D tensor"""
        x = create_tensor((64, 64, 64), torch.float32)
        y_gems = asinh(x)
        y_torch = torch.asinh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.asinh
    def test_dim_4d_batch(self):
        """Test: 4D batch tensor (batch × channel × height × width)"""
        x = create_tensor((16, 3, 128, 128), torch.float32)
        y_gems = asinh(x)
        y_torch = torch.asinh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 3. Data type coverage tests (competition requirement: at least support float32/float16)
# ============================================================================


class TestAsinhDataTypes:
    """Test data type coverage: float16, float32, float64, bfloat16"""

    @pytest.mark.asinh
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_all_float_dtypes(self, dtype):
        """Test: all floating point data types"""
        x = create_tensor((256, 256), dtype)
        y_gems = asinh(x)
        y_torch = torch.asinh(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)


# ============================================================================
# 4. Parameter pattern coverage tests (competition requirement: default values, boundary values, special values)
# ============================================================================


class TestAsinhParameterPatterns:
    """Test parameter patterns: default values, boundary values, special values"""

    @pytest.mark.asinh
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_default_usage(self, dtype):
        """Test: default parameter usage"""
        x = create_tensor((100, 100), dtype)
        y_gems = asinh(x)  # 默认调用
        y_torch = torch.asinh(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.asinh
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_edge_case_zero(self, dtype):
        """Test: boundary value - asinh(0) = 0"""
        x = torch.zeros((10, 10), dtype=dtype, device="cuda")
        y_gems = asinh(x)
        y_torch = torch.asinh(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)
        # Verify zero maps to zero
        assert torch.allclose(y_gems, torch.zeros_like(y_gems))

    @pytest.mark.asinh
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_edge_case_ones(self, dtype):
        """Test: boundary value - asinh(1.0)"""
        x = torch.ones((10, 10), dtype=dtype, device="cuda")
        y_gems = asinh(x)
        y_torch = torch.asinh(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.asinh
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_special_small_values(self, dtype):
        """Test: special values - small positive/negative numbers (asinh(x) ≈ x)"""
        x = torch.tensor(
            [0.1, 0.01, 0.001, -0.1, -0.01, -0.001], dtype=dtype, device="cuda"
        )
        y_gems = asinh(x)
        y_torch = torch.asinh(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.asinh
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_special_large_values(self, dtype):
        """Test: special values - large positive/negative numbers"""
        x = torch.tensor(
            [1e6, 1e8, 1e10, -1e6, -1e8, -1e10], dtype=dtype, device="cuda"
        )
        y_gems = asinh(x)
        y_torch = torch.asinh(x)
        # For very large values, use more relaxed tolerance
        atol = ATOL_DICT[dtype] * 10
        assert_close(y_gems, y_torch, rtol=1e-3, atol=atol, dtype=dtype)

    @pytest.mark.asinh
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_special_infinity(self, dtype):
        """Test: special values - infinity"""
        x = torch.tensor([float("inf"), float("-inf")], dtype=dtype, device="cuda")
        y_gems = asinh(x)
        y_torch = torch.asinh(x)
        # asinh(inf) = inf, asinh(-inf) = -inf
        assert torch.isinf(y_gems[0]) and y_gems[0] > 0
        assert torch.isinf(y_gems[1]) and y_gems[1] < 0
        assert torch.isinf(y_torch[0]) and y_torch[0] > 0
        assert torch.isinf(y_torch[1]) and y_torch[1] < 0

    @pytest.mark.asinh
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_special_nan(self, dtype):
        """Test: special values - NaN"""
        x = torch.tensor([float("nan")], dtype=dtype, device="cuda")
        y_gems = asinh(x)
        y_torch = torch.asinh(x)
        # NaN should remain NaN
        assert torch.isnan(y_gems[0])
        assert torch.isnan(y_torch[0])


# ============================================================================
# 5. Functional completeness tests (competition requirement: all functional branches)
# ============================================================================


class TestAsinhFunctionalCompleteness:
    """Test functional completeness: basic computation, odd function property, batch processing"""

    @pytest.mark.asinh
    def test_basic_computation(self):
        """Test: basic element-wise computation"""
        x = torch.tensor([0.0, 0.5, 1.0, 2.0, 10.0], device="cuda")
        y_gems = asinh(x)
        y_torch = torch.asinh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.asinh
    def test_odd_function_property(self):
        """Test: odd function property - asinh(-x) = -asinh(x)"""
        x = torch.randn(100, 100, device="cuda") * 10
        y_pos = asinh(x)
        y_neg = asinh(-x)
        # Verify odd function property
        assert_close(y_neg, -y_pos, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.asinh
    def test_monotonic_increase(self):
        """Test: monotonic increasing property"""
        x = torch.linspace(-100, 100, 1000, device="cuda")
        y = asinh(x)
        # Verify monotonic increasing
        assert torch.all(y[1:] >= y[:-1])

    @pytest.mark.asinh
    def test_definition_formula(self):
        """Test: definition formula - asinh(x) = ln(x + sqrt(x^2 + 1))"""
        x = torch.randn(100, device="cuda") * 10
        y_asinh = asinh(x)
        y_formula = torch.log(x + torch.sqrt(x * x + 1))
        assert_close(y_asinh, y_formula, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.asinh
    def test_batch_processing(self):
        """Test: batch processing (multiple tensors)"""
        tensors = [
            create_tensor((64, 64), torch.float32),
            create_tensor((128, 128), torch.float32),
            create_tensor((256, 256), torch.float32),
        ]
        for x in tensors:
            y_gems = asinh(x)
            y_torch = torch.asinh(x)
            assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.asinh
    def test_consistency_with_pytorch(self):
        """Test: consistency with PyTorch implementation"""
        # Test multiple times using random tensors
        for _ in range(10):
            x = create_tensor((100, 100), torch.float32)
            y_gems = asinh(x)
            y_torch = torch.asinh(x)
            assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 6. Comprehensive coverage tests (parameterized combinations)
# ============================================================================


class TestAsinhComprehensive:
    """Comprehensive tests: multi-dimensional combination coverage"""

    @pytest.mark.asinh
    @pytest.mark.parametrize("shape", [8, 64 * 64, 256 * 256, 1024 * 1024])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_shape_dtype_combination(self, shape, dtype):
        """Test: combinations of shapes and data types"""
        x = create_tensor((shape,), dtype)
        y_gems = asinh(x)
        y_torch = torch.asinh(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.asinh
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
        y_gems = asinh(x)
        y_torch = torch.asinh(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.asinh
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((10,), torch.float32),
            ((10, 10), torch.float32),
            ((10, 10, 10), torch.float32),
            ((2, 3, 4, 5), torch.float32),
        ],
    )
    def test_various_shapes_and_dtypes(self, shape, dtype):
        """Test: various combinations of shapes and data types"""
        x = create_tensor(shape, dtype)
        y_gems = asinh(x)
        y_torch = torch.asinh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=dtype)
=======
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.asinh
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_accuracy_asinh(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.asinh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.asinh(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.asinh
@pytest.mark.parametrize(
    "shape",
    [(1, 1), (8, 8), (64, 64), (256, 256), (1024, 1024), (4096, 4096)],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_asinh_various_sizes(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.asinh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.asinh(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.asinh
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_asinh_edge_cases(dtype):
    vals = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        10.0,
        -10.0,
        float("inf"),
        float("-inf"),
        float("nan"),
    ]
    inp = torch.tensor(vals, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.asinh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.asinh(inp)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.asinh
def test_accuracy_asinh_empty_tensor():
    inp = torch.empty(0, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.asinh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.asinh(inp)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.asinh
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_asinh_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.empty_like(ref_inp)
    torch.asinh(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.empty_like(inp)
        torch.asinh(inp, out=res_out)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.asinh_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asinh_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.asinh_()
    with flag_gems.use_gems():
        res_out = inp.asinh_()

    utils.gems_assert_close(res_out, ref_out, dtype)
>>>>>>> master
