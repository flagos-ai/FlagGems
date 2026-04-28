<<<<<<< feature/FlagGemsOperatorDevelopmentCompetition
"""
Test suite for tril operator.

This test module validates the correctness and functionality
of the tril operator implementation following FlagGems testing conventions.

Test coverage description:
- Input sizes: small, regular, large
- Input dimensions: 2D, 3D, 4D (batch matrices)
- Data types: float16, float32, float64, bfloat16
- Parameter patterns: diagonal parameter (negative, zero, positive values)
- Functional completeness: square matrices, rectangular matrices, batch processing, edge cases
"""

import pytest
import torch

from flag_gems.ops import tril

# ============================================================================
# Test data definitions (following competition requirements)
# ============================================================================

# Input size coverage (competition requirement: small, regular, large sizes)
MATRIX_SHAPES = [
    (8, 8),  # small size
    (16, 16),  # small size
    (64, 64),  # regular size
    (256, 256),  # regular size
    (1024, 1024),  # large size
    (4096, 4096),  # large size
]

# Rectangular shapes (rows ≠ columns)
RECTANGULAR_SHAPES = [
    (32, 64),  # wide matrix
    (64, 32),  # tall matrix
    (16, 128),  # wider
    (128, 16),  # taller
]

# Data type coverage (competition requirement: at least support float32/float16)
FLOAT_DTYPES = [
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
]

# Precision standards (competition requirement standards)
ATOL_DICT = {
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.float64: 1e-7,
    torch.bfloat16: 0.016,
}


# ============================================================================
# Helper functions
# ============================================================================


def assert_equal(actual, expected, msg=""):
    """
    Verify exact equality using torch.equal (tril must match exactly)

    Args:
        actual: FlagGems implementation result
        expected: PyTorch reference result
        msg: error message
    """
    assert torch.equal(
        actual, expected
    ), f"tril results don't match{': ' + msg if msg else ''}"


def create_matrix(shape, dtype, device="cuda"):
    """Create random matrix"""
    x = torch.randn(shape, dtype=dtype, device=device)
    return x


# ============================================================================
# 1. Input size coverage tests (competition requirement: small, regular, large categories)
# ============================================================================


class TestTrilInputSize:
    """Test input size coverage"""

    @pytest.mark.tril
    def test_size_very_small(self):
        """Test: 2×2 (extremely small size)"""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
        y_gems = tril(x)
        y_torch = torch.tril(x)
        assert_equal(y_gems, y_torch)

    @pytest.mark.tril
    @pytest.mark.parametrize("shape", MATRIX_SHAPES)
    def test_various_sizes(self, shape):
        """Test: various matrix sizes"""
        x = create_matrix(shape, torch.float32)
        y_gems = tril(x)
        y_torch = torch.tril(x)
        assert_equal(y_gems, y_torch)


# ============================================================================
# 2. Input dimension coverage tests (competition requirement: cover all valid dimensions)
# ============================================================================


class TestTrilInputDimensions:
    """Test input dimension coverage: 2D, 3D, 4D"""

    @pytest.mark.tril
    def test_dim_2d_square(self):
        """Test: 2D square matrix"""
        x = create_matrix((100, 100), torch.float32)
        y_gems = tril(x)
        y_torch = torch.tril(x)
        assert_equal(y_gems, y_torch)

    @pytest.mark.tril
    def test_dim_2d_rectangular(self):
        """Test: 2D rectangular matrix"""
        x = create_matrix((50, 100), torch.float32)
        y_gems = tril(x)
        y_torch = torch.tril(x)
        assert_equal(y_gems, y_torch)

    @pytest.mark.tril
    def test_dim_3d_batch(self):
        """Test: 3D batch matrix (batch, M, N)"""
        x = create_matrix((10, 64, 64), torch.float32)
        y_gems = tril(x)
        y_torch = torch.tril(x)
        assert_equal(y_gems, y_torch)

    @pytest.mark.tril
    def test_dim_4d_batch(self):
        """Test: 4D batch matrix (batch, channel, M, N)"""
        x = create_matrix((4, 3, 32, 32), torch.float32)
        y_gems = tril(x)
        y_torch = torch.tril(x)
        assert_equal(y_gems, y_torch)


# ============================================================================
# 3. Data type coverage tests (competition requirement: at least support float32/float16)
# ============================================================================


class TestTrilDataTypes:
    """Test data type coverage: float16, float32, float64, bfloat16"""

    @pytest.mark.tril
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_all_float_dtypes(self, dtype):
        """Test: all floating point data types"""
        x = create_matrix((64, 64), dtype)
        y_gems = tril(x)
        y_torch = torch.tril(x)
        assert_equal(y_gems, y_torch)


# ============================================================================
# 4. Parameter pattern coverage tests (diagonal parameter)
# ============================================================================


class TestTrilParameterPatterns:
    """Test diagonal parameter: default value, boundary value, special value"""

    @pytest.mark.tril
    def test_default_diagonal(self):
        """Test: default diagonal=0 (main diagonal)"""
        x = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            device="cuda",
            dtype=torch.float32,
        )
        y_gems = tril(x)
        y_torch = torch.tril(x)
        expected = torch.tensor(
            [[1, 0, 0, 0], [5, 6, 0, 0], [9, 10, 11, 0]], device="cuda"
        )
        assert_equal(y_gems, expected)
        assert_equal(y_torch, expected)

    @pytest.mark.tril
    def test_diagonal_positive(self):
        """Test: diagonal > 0 (above main diagonal)"""
        x = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            device="cuda",
            dtype=torch.float32,
        )

        # diagonal = 1
        y_gems = tril(x, diagonal=1)
        y_torch = torch.tril(x, diagonal=1)
        expected = torch.tensor(
            [[1, 2, 0, 0], [5, 6, 7, 0], [9, 10, 11, 12]], device="cuda"
        )
        assert_equal(y_gems, expected)
        assert_equal(y_torch, expected)

        # diagonal = 2
        y_gems = tril(x, diagonal=2)
        y_torch = torch.tril(x, diagonal=2)
        expected = torch.tensor(
            [[1, 2, 3, 0], [5, 6, 7, 8], [9, 10, 11, 12]], device="cuda"
        )
        assert_equal(y_gems, expected)
        assert_equal(y_torch, expected)

    @pytest.mark.tril
    def test_diagonal_negative(self):
        """Test: diagonal < 0 (below main diagonal)"""
        x = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            device="cuda",
            dtype=torch.float32,
        )

        # diagonal = -1
        y_gems = tril(x, diagonal=-1)
        y_torch = torch.tril(x, diagonal=-1)
        expected = torch.tensor(
            [[0, 0, 0, 0], [5, 0, 0, 0], [9, 10, 0, 0]], device="cuda"
        )
        assert_equal(y_gems, expected)
        assert_equal(y_torch, expected)

        # diagonal = -2
        y_gems = tril(x, diagonal=-2)
        y_torch = torch.tril(x, diagonal=-2)
        expected = torch.tensor(
            [[0, 0, 0, 0], [0, 0, 0, 0], [9, 0, 0, 0]], device="cuda"
        )
        assert_equal(y_gems, expected)
        assert_equal(y_torch, expected)

    @pytest.mark.tril
    @pytest.mark.parametrize("diagonal", [-2, -1, 0, 1, 2])
    def test_diagonal_range(self, diagonal):
        """Test: diagonal parameter range"""
        x = create_matrix((32, 32), torch.float32)
        y_gems = tril(x, diagonal=diagonal)
        y_torch = torch.tril(x, diagonal=diagonal)
        assert_equal(y_gems, y_torch)


# ============================================================================
# 5. Functional completeness tests (competition requirement: all functional branches)
# ============================================================================


class TestTrilFunctionalCompleteness:
    """Test functional completeness: square matrices, rectangular matrices, batch processing, edge cases"""

    @pytest.mark.tril
    def test_square_matrix(self):
        """Test: square matrix (M = N)"""
        x = create_matrix((64, 64), torch.float32)
        y_gems = tril(x)
        y_torch = torch.tril(x)
        assert_equal(y_gems, y_torch)

    @pytest.mark.tril
    @pytest.mark.parametrize("shape", RECTANGULAR_SHAPES)
    def test_rectangular_matrix(self, shape):
        """Test: rectangular matrix (M ≠ N)"""
        x = create_matrix(shape, torch.float32)
        y_gems = tril(x)
        y_torch = torch.tril(x)
        assert_equal(y_gems, y_torch)

    @pytest.mark.tril
    def test_tall_matrix(self):
        """Test: tall matrix (rows > columns)"""
        x = create_matrix((100, 50), torch.float32)
        y_gems = tril(x)
        y_torch = torch.tril(x)
        assert_equal(y_gems, y_torch)

    @pytest.mark.tril
    def test_wide_matrix(self):
        """Test: wide matrix (rows < columns)"""
        x = create_matrix((50, 100), torch.float32)
        y_gems = tril(x)
        y_torch = torch.tril(x)
        assert_equal(y_gems, y_torch)

    @pytest.mark.tril
    def test_batch_processing_2d(self):
        """Test: 2D batch processing (batch, M, N)"""
        x = create_matrix((10, 32, 32), torch.float32)
        y_gems = tril(x)
        y_torch = torch.tril(x)
        assert_equal(y_gems, y_torch)

    @pytest.mark.tril
    def test_batch_processing_3d(self):
        """Test: 3D batch processing (batch, channel, M, N)"""
        x = create_matrix((4, 3, 16, 16), torch.float32)
        y_gems = tril(x)
        y_torch = torch.tril(x)
        assert_equal(y_gems, y_torch)

    @pytest.mark.tril
    def test_consistency_with_pytorch(self):
        """Test: consistency with PyTorch implementation"""
        # Test multiple random matrices
        for _ in range(10):
            x = create_matrix((64, 64), torch.float32)
            y_gems = tril(x)
            y_torch = torch.tril(x)
            assert_equal(y_gems, y_torch)

    @pytest.mark.tril
    def test_special_matrices(self):
        """Test: special matrices"""
        # Zero matrix
        x_zero = torch.zeros((10, 10), device="cuda")
        y_gems = tril(x_zero)
        y_torch = torch.tril(x_zero)
        assert_equal(y_gems, y_torch)

        # Identity matrix
        x_eye = torch.eye(10, device="cuda")
        y_gems = tril(x_eye)
        y_torch = torch.tril(x_eye)
        assert_equal(y_gems, y_torch)

        # All-ones matrix
        x_ones = torch.ones((10, 10), device="cuda")
        y_gems = tril(x_ones)
        y_torch = torch.tril(x_ones)
        assert_equal(y_gems, y_torch)


# ============================================================================
# 6. Comprehensive coverage tests (parameterized combinations)
# ============================================================================


class TestTrilComprehensive:
    """Comprehensive tests: multi-dimensional combination coverage"""

    @pytest.mark.tril
    @pytest.mark.parametrize("shape", [(8, 8), (32, 32), (64, 64), (128, 128)])
    @pytest.mark.parametrize("diagonal", [-1, 0, 1])
    def test_shape_diagonal_combination(self, shape, diagonal):
        """Test: combination of shape and diagonal parameters"""
        x = create_matrix(shape, torch.float32)
        y_gems = tril(x, diagonal=diagonal)
        y_torch = torch.tril(x, diagonal=diagonal)
        assert_equal(y_gems, y_torch)

    @pytest.mark.tril
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((16, 16), torch.float16),
            ((32, 32), torch.float32),
            ((64, 64), torch.float64),
            ((128, 128), torch.bfloat16),
        ],
    )
    def test_typical_use_cases(self, shape, dtype):
        """Test: typical use cases"""
        x = create_matrix(shape, dtype)
        y_gems = tril(x)
        y_torch = torch.tril(x)
        assert_equal(y_gems, y_torch)

    @pytest.mark.tril
    @pytest.mark.parametrize("shape", RECTANGULAR_SHAPES)
    @pytest.mark.parametrize("diagonal", [-1, 0, 1, 2])
    def test_rectangular_with_diagonal(self, shape, diagonal):
        """Test: rectangular matrix with diagonal parameter combination"""
        x = create_matrix(shape, torch.float32)
        y_gems = tril(x, diagonal=diagonal)
        y_torch = torch.tril(x, diagonal=diagonal)
        assert_equal(y_gems, y_torch)
=======
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SHAPE_DIAGONAL = list(zip(utils.POINTWISE_SHAPES, [-2, -2, -1, 0, 1, 3]))


@pytest.mark.tril
@pytest.mark.parametrize("shape, diagonal", SHAPE_DIAGONAL)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tril(shape, diagonal, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = utils.unsqueeze_tensor(inp, 2)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)

    utils.gems_assert_close(res_out, ref_out, dtype)
>>>>>>> master
