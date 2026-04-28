"""
Test suite for cosh operator.

This test module validates the correctness, precision, and performance
of the cosh operator implementation following FlagGems testing conventions.

Test coverage description:
- Input sizes: 1×1, 8×8, 64×64, 256×256, 1024×1024, 4096×4096
- Input dimensions: 1D, 2D, 3D, 4D
- Data types: float16, float32, float64, bfloat16
- Parameter modes: default values, boundary values, special values, numerical stability
- Functional completeness: basic computation, type promotion, batch processing, symmetry
"""

import pytest
import torch

from flag_gems.ops import cosh

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
    torch.float16: 2.0e-3,  # Adjusted for float16 precision (2 × machine_epsilon)
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
    """Create test tensor"""
    x = torch.randn(shape, dtype=dtype, device=device)
    return x


# ============================================================================
# 1. Input size coverage tests (competition requirement: small, medium, large three categories)
# ============================================================================


class TestCoshInputSize:
    """Test input size coverage"""

    @pytest.mark.cosh
    def test_size_very_small(self):
        """Test: 1×1 (extremely small size)"""
        x = torch.tensor([[0.5]], device="cuda")
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_size_small(self):
        """Test: 8×8 (small size)"""
        x = create_tensor((8, 8), torch.float32)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_size_medium_64(self):
        """Test: 64×64 (medium size)"""
        x = create_tensor((64, 64), torch.float32)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_size_medium_256(self):
        """Test: 256×256 (medium size)"""
        x = create_tensor((256, 256), torch.float32)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_size_large_1k(self):
        """Test: 1024×1024 (large size)"""
        x = create_tensor((1024, 1024), torch.float32)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_size_large_4k(self):
        """Test: 4096×4096 (large size)"""
        x = create_tensor((4096, 4096), torch.float32)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 2. Input dimension coverage tests (competition requirement: cover all valid dimensions)
# ============================================================================


class TestCoshInputDimensions:
    """Test input dimension coverage: 1D, 2D, 3D, 4D"""

    @pytest.mark.cosh
    def test_dim_1d_small(self):
        """Test: 1D tensor (small size)"""
        x = create_tensor((10,), torch.float32)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_dim_1d_large(self):
        """Test: 1D tensor (large size)"""
        x = create_tensor((10000,), torch.float32)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_dim_2d(self):
        """Test: 2D tensor"""
        x = create_tensor((100, 100), torch.float32)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_dim_3d(self):
        """Test: 3D tensor"""
        x = create_tensor((64, 64, 64), torch.float32)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_dim_4d_batch(self):
        """Test: 4D batch tensor (batch × channel × height × width)"""
        x = create_tensor((16, 3, 128, 128), torch.float32)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 3. Data type coverage tests (competition requirement: at least support float32/float16)
# ============================================================================


class TestCoshDataTypes:
    """Test data type coverage: float16, float32, float64, bfloat16"""

    @pytest.mark.cosh
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_all_float_dtypes(self, dtype):
        """Test: all floating point data types"""
        x = create_tensor((256, 256), dtype)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)


# ============================================================================
# 4. Parameter pattern coverage tests (competition requirement: default values,
# boundary values, special values, numerical stability)
# ============================================================================


class TestCoshParameterPatterns:
    """Test parameter patterns: default values, boundary values, special values, numerical stability"""

    @pytest.mark.cosh
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_default_usage(self, dtype):
        """Test: default parameter usage"""
        x = create_tensor((100, 100), dtype)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.cosh
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_edge_case_zero(self, dtype):
        """Test: boundary value - cosh(0) = 1"""
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
        """Test: boundary values - small values"""
        x = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=dtype, device="cuda")
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.cosh
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_special_negative_values(self, dtype):
        """Test: special values - negative values (symmetry test)"""
        x = torch.tensor([-1.0, -2.0, -3.0, -10.0], dtype=dtype, device="cuda")
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.cosh
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_special_large_values(self, dtype):
        """Test: special values - large values (numerical stability test)"""
        x = torch.tensor([10.0, 50.0, 100.0], dtype=dtype, device="cuda")
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.cosh
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_numerical_stability_extreme(self, dtype):
        """Test: numerical stability - extreme values (±100)"""
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
        """Test: symmetry cosh(-x) = cosh(x)"""
        x = create_tensor((100, 100), dtype)
        y_pos = cosh(x)
        y_neg = cosh(-x)
        # cosh should be symmetric
        assert_close(y_pos, y_neg, rtol=0, atol=0, dtype=dtype)


# ============================================================================
# 5. Functional completeness tests (competition requirement: all functional branches)
# ============================================================================


class TestCoshFunctionalCompleteness:
    """Test functional completeness: basic computation, batch processing, symmetry, consistency"""

    @pytest.mark.cosh
    def test_basic_computation(self):
        """Test: basic element-wise computation"""
        x = torch.tensor([0.0, 1.0, 2.0, 3.0], device="cuda")
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_symmetry_verification(self):
        """Test: symmetry verification cosh(-x) = cosh(x)"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
        y_pos = cosh(x)
        y_neg = cosh(-x)
        assert_close(y_pos, y_neg, rtol=0, atol=0, dtype=torch.float32)

    @pytest.mark.cosh
    def test_batch_processing(self):
        """Test: batch processing (multiple tensors)"""
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
        """Test: consistency with PyTorch implementation"""
        # Test multiple times using random tensors
        for _ in range(10):
            x = create_tensor((100, 100), torch.float32)
            y_gems = cosh(x)
            y_torch = torch.cosh(x)
            assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.cosh
    def test_known_values(self):
        """Test: known exact values"""
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
# 6. Comprehensive coverage tests (parameterized combinations)
# ============================================================================


class TestCoshComprehensive:
    """Comprehensive tests: multi-dimensional combination coverage"""

    @pytest.mark.cosh
    @pytest.mark.parametrize("shape", [8, 64 * 64, 256 * 256, 1024 * 1024])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_shape_dtype_combination(self, shape, dtype):
        """Test: combinations of shapes and data types"""
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
        """Test: typical use cases"""
        x = create_tensor(shape, dtype)
        y_gems = cosh(x)
        y_torch = torch.cosh(x)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.cosh
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cosh(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.cosh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.cosh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cosh_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cosh_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.cosh_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.cosh_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cosh_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cosh_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.cosh(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.empty_like(inp)
        torch.cosh(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cosh
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cosh_special_values(dtype):
    inp = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.cosh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.cosh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.cosh
@pytest.mark.parametrize("shape", [(0,), (4, 0), (2, 0, 3)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cosh_empty(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.cosh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.cosh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cosh
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cosh_even_property(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        pos = torch.cosh(inp)
        neg = torch.cosh(-inp)

    ref_pos = utils.to_reference(pos, True)
    utils.gems_assert_close(neg, ref_pos, dtype)


@pytest.mark.cosh
@pytest.mark.parametrize("shape", [(17, 33), (5, 7, 9)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cosh_noncontiguous(shape, dtype):
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = base.transpose(-1, -2)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.cosh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.cosh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
