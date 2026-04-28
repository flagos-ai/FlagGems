<<<<<<< feature/FlagGemsOperatorDevelopmentCompetition
"""
Test suite for Roll operator.

This test module validates the correctness, precision, and performance
of the roll operator implementation following FlagGems testing conventions.

Test coverage description:
- Input sizes: 1×1, 8×8, 64×64, 256×256, 1024×1024, 4096×4096
- Input dimensions: 1D, 2D, 3D, 4D, 5D
- Data types: float16, float32, float64, bfloat16, int8, int16, int32, int64, bool
- Parameter modes: default values, boundary values, special values, different shift and dimension combinations
- Functional completeness: single-dimension rolling, multi-dimension rolling, flattened rolling, empty tensor handling
"""

import pytest
import torch

from flag_gems.ops import roll

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

# Integer type coverage
INT_DTYPES = [
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
]


# ============================================================================
# Helper functions
# ============================================================================


def assert_equal(actual, expected):
    """
    Verify exact equality using torch.equal (roll operation requires exact match)

    Args:
        actual: FlagGems implementation result
        expected: PyTorch reference result
    """
    assert torch.equal(
        actual, expected
    ), "Results don't match: shape mismatch or values differ"


def create_tensor(shape, dtype, device="cuda"):
    """Create test tensor (based on data type)"""
    if dtype.is_floating_point:
        x = torch.randn(shape, dtype=dtype, device=device)
    else:
        x = torch.randint(1, 100, shape, dtype=dtype, device=device)
    return x


# ============================================================================
# 1. Input size coverage tests (competition requirement: small, medium, large three categories)
# ============================================================================


class TestRollInputSize:
    """Test input size coverage"""

    @pytest.mark.roll
    def test_size_very_small(self):
        """Test: 1×1 (extremely small size)"""
        x = torch.tensor([[0.5]], device="cuda")
        y_gems = roll(x, shifts=1)
        y_torch = torch.roll(x, shifts=1)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_size_small(self):
        """Test: 8×8 (small size)"""
        x = create_tensor((8, 8), torch.float32)
        y_gems = roll(x, shifts=2, dims=0)
        y_torch = torch.roll(x, shifts=2, dims=0)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_size_medium_64(self):
        """Test: 64×64 (medium size)"""
        x = create_tensor((64, 64), torch.float32)
        y_gems = roll(x, shifts=10, dims=1)
        y_torch = torch.roll(x, shifts=10, dims=1)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_size_medium_256(self):
        """Test: 256×256 (medium size)"""
        x = create_tensor((256, 256), torch.float32)
        y_gems = roll(x, shifts=50, dims=0)
        y_torch = torch.roll(x, shifts=50, dims=0)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_size_large_1k(self):
        """Test: 1024×1024 (large size)"""
        x = create_tensor((1024, 1024), torch.float32)
        y_gems = roll(x, shifts=100, dims=1)
        y_torch = torch.roll(x, shifts=100, dims=1)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_size_large_4k(self):
        """Test: 4096×4096 (large size)"""
        x = create_tensor((4096, 4096), torch.float32)
        y_gems = roll(x, shifts=500, dims=0)
        y_torch = torch.roll(x, shifts=500, dims=0)
        assert_equal(y_gems, y_torch)


# ============================================================================
# 2. Input dimension coverage tests (competition requirement: cover all valid dimensions)
# ============================================================================


class TestRollInputDimensions:
    """Test input dimension coverage: 1D, 2D, 3D, 4D, 5D"""

    @pytest.mark.roll
    def test_dim_1d_small(self):
        """Test: 1D tensor (small size)"""
        x = create_tensor((10,), torch.float32)
        y_gems = roll(x, shifts=3)
        y_torch = torch.roll(x, shifts=3)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_dim_1d_large(self):
        """Test: 1D tensor (large size)"""
        x = create_tensor((10000,), torch.float32)
        y_gems = roll(x, shifts=100)
        y_torch = torch.roll(x, shifts=100)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_dim_2d(self):
        """Test: 2D tensor"""
        x = create_tensor((100, 100), torch.float32)
        y_gems = roll(x, shifts=20, dims=0)
        y_torch = torch.roll(x, shifts=20, dims=0)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_dim_3d(self):
        """Test: 3D tensor"""
        x = create_tensor((64, 64, 64), torch.float32)
        y_gems = roll(x, shifts=10, dims=1)
        y_torch = torch.roll(x, shifts=10, dims=1)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_dim_4d_batch(self):
        """Test: 4D batch tensor (batch × channel × height × width)"""
        x = create_tensor((16, 3, 128, 128), torch.float32)
        y_gems = roll(x, shifts=30, dims=2)
        y_torch = torch.roll(x, shifts=30, dims=2)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_dim_5d(self):
        """Test: 5D tensor"""
        x = create_tensor((2, 3, 4, 5, 6), torch.float32)
        y_gems = roll(x, shifts=(1, 2, 0), dims=(0, 2, 4))
        y_torch = torch.roll(x, shifts=(1, 2, 0), dims=(0, 2, 4))
        assert_equal(y_gems, y_torch)


# ============================================================================
# 3. Data type coverage tests (competition requirement: at least support float32/float16)
# ============================================================================


class TestRollDataTypes:
    """Test data type coverage: float16, float32, float64, bfloat16, int8, int16, int32, int64, bool"""

    @pytest.mark.roll
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_float_dtypes(self, dtype):
        """Test: all floating point data types"""
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
        """Test: all integer data types"""
        x = create_tensor((50, 50), dtype)
        y_gems = roll(x, shifts=5, dims=1)
        y_torch = torch.roll(x, shifts=5, dims=1)

        assert y_gems.dtype == y_torch.dtype
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_bool_dtype(self):
        """Test: boolean type"""
        x = torch.randint(0, 2, (20, 20), dtype=torch.bool, device="cuda")
        y_gems = roll(x, shifts=3, dims=0)
        y_torch = torch.roll(x, shifts=3, dims=0)

        assert y_gems.dtype == y_torch.dtype
        assert_equal(y_gems, y_torch)


# ============================================================================
# 4. Parameter pattern coverage tests (competition requirement: default values, boundary values, special values)
# ============================================================================


class TestRollParameterPatterns:
    """Test parameter patterns: default values, boundary values, special values,
    different shift and dimension combinations"""

    @pytest.mark.roll
    @pytest.mark.parametrize("shift", [0, 1, 5, 10, 100])
    def test_various_positive_shifts(self, shift):
        """Test: various positive shifts"""
        x = create_tensor((100,), torch.float32)
        y_gems = roll(x, shifts=shift)
        y_torch = torch.roll(x, shifts=shift)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    @pytest.mark.parametrize("shift", [-1, -5, -10, -100])
    def test_negative_shifts(self, shift):
        """Test: negative shifts"""
        x = create_tensor((100,), torch.float32)
        y_gems = roll(x, shifts=shift)
        y_torch = torch.roll(x, shifts=shift)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_shift_larger_than_size(self):
        """Test: shift larger than tensor size (circular)"""
        x = torch.arange(10, device="cuda", dtype=torch.float32)
        shift = 15  # larger than size 10

        y_gems = roll(x, shifts=shift)
        y_torch = torch.roll(x, shifts=shift)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_negative_dim_index(self):
        """Test: negative dimension index"""
        x = create_tensor((10, 20, 30), torch.float32)

        # Test dim=-1 (last dimension)
        y_gems = roll(x, shifts=3, dims=-1)
        y_torch = torch.roll(x, shifts=3, dims=-1)
        assert_equal(y_gems, y_torch)

        # Test dim=-2 (second to last dimension)
        y_gems = roll(x, shifts=2, dims=-2)
        y_torch = torch.roll(x, shifts=2, dims=-2)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_dims_none_flatten(self):
        """Test: flatten tensor and roll when dims=None"""
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device="cuda")

        y_gems = roll(x, shifts=2)
        y_torch = torch.roll(x, shifts=2)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_tuple_shifts_and_dims(self):
        """Test: tuple form of shifts and dims"""
        x = create_tensor((10, 20), torch.float32)
        shifts = (3, 5)
        dims = (0, 1)

        y_gems = roll(x, shifts=shifts, dims=dims)
        y_torch = torch.roll(x, shifts=shifts, dims=dims)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_edge_case_zero_shift(self):
        """Test: boundary value - shift=0 (no operation)"""
        x = create_tensor((50, 50), torch.float32)
        y_gems = roll(x, shifts=0, dims=1)
        y_torch = torch.roll(x, shifts=0, dims=1)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_edge_case_shift_equal_to_size(self):
        """Test: boundary value - shift equals dimension size (should remain unchanged)"""
        x = create_tensor((20, 30), torch.float32)

        y_gems = roll(x, shifts=20, dims=0)
        y_torch = torch.roll(x, shifts=20, dims=0)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_special_very_large_shift(self):
        """Test: special value - very large shift (should cycle normally)"""
        x = create_tensor((10, 10), torch.float32)

        # Very large positive shift (should cycle)
        shift = 1000
        y_gems = roll(x, shifts=shift, dims=1)
        y_torch = torch.roll(x, shifts=shift, dims=1)
        assert_equal(y_gems, y_torch)

        # Very large negative shift
        shift = -1000
        y_gems = roll(x, shifts=shift, dims=0)
        y_torch = torch.roll(x, shifts=shift, dims=0)
        assert_equal(y_gems, y_torch)


# ============================================================================
# 5. Functional completeness tests (competition requirement: all functional branches)
# ============================================================================


class TestRollFunctionalCompleteness:
    """Test functional completeness: single-dimension rolling, multi-dimension rolling,
    flattened rolling, empty tensor handling"""

    @pytest.mark.roll
    def test_single_dimension(self):
        """Test: single-dimension rolling"""
        x = create_tensor((10, 20, 30), torch.float32)

        y_gems = roll(x, shifts=5, dims=1)
        y_torch = torch.roll(x, shifts=5, dims=1)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_multiple_dimensions(self):
        """Test: multi-dimension rolling (tuple parameters)"""
        x = create_tensor((10, 20, 30), torch.float32)
        shifts = (3, 5, 2)
        dims = (0, 1, 2)

        y_gems = roll(x, shifts=shifts, dims=dims)
        y_torch = torch.roll(x, shifts=shifts, dims=dims)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_flattened_roll(self):
        """Test: flattened tensor rolling"""
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device="cuda")

        y_gems = roll(x, shifts=2)
        y_torch = torch.roll(x, shifts=2)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_empty_tensor(self):
        """Test: empty tensor handling"""
        x = torch.randn(0, 10, device="cuda")

        y_gems = roll(x, shifts=2, dims=0)
        y_torch = torch.roll(x, shifts=2, dims=0)

        assert y_gems.shape == y_torch.shape
        assert y_gems.numel() == 0

    @pytest.mark.roll
    def test_single_element(self):
        """Test: single-element tensor"""
        x = torch.tensor([42.0], device="cuda")

        y_gems = roll(x, shifts=100)
        y_torch = torch.roll(x, shifts=100)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_contiguous_memory_layout(self):
        """Test: contiguous memory layout"""
        x = torch.randn(10, 10, device="cuda")
        assert x.is_contiguous()

        y_gems = roll(x, shifts=3, dims=1)
        y_torch = torch.roll(x, shifts=3, dims=1)

        assert y_gems.is_contiguous()
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_non_contiguous_memory_layout(self):
        """Test: non-contiguous memory layout"""
        x = torch.randn(
            10, 10, device="cuda"
        ).t()  # transpose to make it non-contiguous
        assert not x.is_contiguous()

        y_gems = roll(x, shifts=3, dims=0)
        y_torch = torch.roll(x, shifts=3, dims=0)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_consistency_with_pytorch(self):
        """Test: consistency with PyTorch implementation"""
        # Test multiple times with random tensors
        for _ in range(10):
            x = create_tensor((100, 100), torch.float32)
            shift = torch.randint(-50, 50, (1,)).item()
            dim = torch.randint(0, 2, (1,)).item()

            y_gems = roll(x, shifts=shift, dims=dim)
            y_torch = torch.roll(x, shifts=shift, dims=dim)
            assert_equal(y_gems, y_torch)


# ============================================================================
# 6. Comprehensive coverage tests (parameterized combinations)
# ============================================================================


class TestRollComprehensive:
    """Comprehensive tests: multi-dimensional combination coverage"""

    @pytest.mark.roll
    @pytest.mark.parametrize(
        "shape,shift,dim",
        [
            ((5, 10), 2, 0),
            ((5, 10), 3, 1),
            ((100, 200), 50, 0),
            ((100, 200), 75, 1),
            ((10, 10), 11, 0),  # shift larger than size
            ((10, 10), -3, 1),  # negative shift
        ],
    )
    def test_shape_shift_dim_combination(self, shape, shift, dim):
        """Test: combination of shape, shift, and dimension"""
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
        """Test: typical use cases"""
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
        """Test: various combinations of shapes and data types"""
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
        """Test: various shift and dimension combinations"""
        x = create_tensor((10, 20, 30), torch.float32)
        y_gems = roll(x, shifts=shifts, dims=dims)
        y_torch = torch.roll(x, shifts=shifts, dims=dims)
        assert_equal(y_gems, y_torch)

    @pytest.mark.roll
    def test_randomized_comparison(self):
        """Randomized test: compare with PyTorch results"""
        torch.manual_seed(42)

        for _ in range(20):
            # Random shape
            n_rows = torch.randint(10, 100, (1,)).item()
            n_cols = torch.randint(10, 100, (1,)).item()

            # Random data type
            dtype_idx = torch.randint(0, 3, (1,)).item()
            dtypes = [torch.float16, torch.float32, torch.float64]
            dtype = dtypes[dtype_idx]

            x = torch.randn(n_rows, n_cols, dtype=dtype, device="cuda")

            # Random shift and dimension
            shift = torch.randint(-20, 20, (1,)).item()
            dim = torch.randint(0, 2, (1,)).item()

            y_gems = roll(x, shifts=shift, dims=dim)
            y_torch = torch.roll(x, shifts=shift, dims=dim)
            assert_equal(y_gems, y_torch)
=======
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

ROLL_SHIFTS_DIMS = [
    (1, 0),
    (-1, 0),
    (2, -1),
    (3, 1),
]


@pytest.mark.roll
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES)
@pytest.mark.parametrize("shifts_dims", ROLL_SHIFTS_DIMS)
def test_roll_single_dim(shape, dtype, shifts_dims):
    shifts, dims = shifts_dims
    ndim = len(shape)
    # Adjust dims if it's out of range for this shape
    if dims >= ndim or dims < -ndim:
        pytest.skip(f"dims {dims} out of range for shape {shape}")

    if dtype in utils.ALL_FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-1000, 1000, shape, device=flag_gems.device).to(dtype)
    ref_inp = utils.to_reference(inp, False)

    ref_out = torch.roll(ref_inp, shifts, dims)
    with flag_gems.use_gems():
        res_out = torch.roll(inp, shifts, dims)

    utils.gems_assert_equal(res_out, ref_out)


ROLL_MULTI_DIMS = [
    ((1, 2), (0, 1)),
    ((-1, 1), (0, -1)),
    ((2, -2), (-2, -1)),
]


@pytest.mark.roll
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("shifts_dims", ROLL_MULTI_DIMS)
def test_roll_multi_dims(shape, dtype, shifts_dims):
    shifts, dims = shifts_dims
    ndim = len(shape)
    # Check all dims are valid for this shape
    for d in dims:
        if d >= ndim or d < -ndim:
            pytest.skip(f"dims {d} out of range for shape {shape}")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    ref_out = torch.roll(ref_inp, shifts, dims)
    with flag_gems.use_gems():
        res_out = torch.roll(inp, shifts, dims)

    utils.gems_assert_equal(res_out, ref_out)


ROLL_FLATTEN_SHIFTS = [1, -1, 5, -3]


@pytest.mark.roll
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("shifts", ROLL_FLATTEN_SHIFTS)
def test_roll_flatten(shape, dtype, shifts):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    # Roll without specifying dims (flatten case)
    ref_out = torch.roll(ref_inp, shifts)
    with flag_gems.use_gems():
        res_out = torch.roll(inp, shifts)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.roll
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_roll_with_non_dense_input(shape, dtype):
    if len(shape) < 2:
        pytest.skip("Need at least 2D for non-dense test")

    shape_dilated = tuple(item * 2 for item in shape)
    inp = torch.randn(shape_dilated, dtype=dtype, device=flag_gems.device)[::2, ::2]
    ref_inp = utils.to_reference(inp, False)

    shifts = 2
    dims = 0

    ref_out = torch.roll(ref_inp, shifts, dims)
    with flag_gems.use_gems():
        res_out = torch.roll(inp, shifts, dims)

    utils.gems_assert_equal(res_out, ref_out)
>>>>>>> master
