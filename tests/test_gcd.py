<<<<<<< feature/FlagGemsOperatorDevelopmentCompetition
"""
Test suite for gcd operator.

This test module validates the correctness, precision, and performance
of the gcd operator implementation following FlagGems testing conventions.

Test coverage description:
- Input sizes: 1×1, 8×8, 64×64, 256×256, 1024×1024, 4096×4096
- Input dimensions: 1D, 2D, 3D, 4D
- Data types: integer types (int8, int16, int32, int64)
- Parameter modes: default values, boundary values, special values, mathematical properties
- Functional completeness: basic computation, zero value handling, negative value handling,
mathematical property verification
"""

import pytest
import torch

from flag_gems.ops import gcd

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

# Integer type coverage (GCD only supports integers)
INT_DTYPES = [
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
]

# Precision standards (GCD is exact calculation, no error allowed)
ATOL_DICT = {
    torch.int8: 0,
    torch.int16: 0,
    torch.int32: 0,
    torch.int64: 0,
}


# ============================================================================
# Helper functions
# ============================================================================


def assert_equal(actual, expected, dtype=torch.int32):
    """
    Verify exact equality using torch.equal (GCD must match exactly)

    Args:
        actual: FlagGems implementation result
        expected: PyTorch reference result
        dtype: Data type
    """
    # GCD must match exactly, no error allowed
    assert torch.equal(
        actual, expected
    ), f"GCD results don't match at {((actual != expected).sum().item())} positions"


def create_int_tensor(shape, dtype, device="cuda", low=1, high=1000):
    """Create random integer test tensor"""
    # Adjust range based on data type to avoid overflow
    dtype_limits = {
        torch.int8: 127,
        torch.int16: 32767,
        torch.int32: 2147483647,
        torch.int64: 9223372036854775807,
    }
    max_val = dtype_limits.get(dtype, 1000)
    adjusted_high = min(high, max_val)
    x = torch.randint(low, adjusted_high, shape, dtype=dtype, device=device)
    return x


# ============================================================================
# 1. Input size coverage tests (competition requirement: small, medium, large three categories)
# ============================================================================


class TestGcdInputSize:
    """Test input size coverage"""

    @pytest.mark.gcd
    def test_size_very_small(self):
        """Test: 1×1 (extremely small size)"""
        x = torch.tensor([[12]], device="cuda")
        y = torch.tensor([[8]], device="cuda")
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)

    @pytest.mark.gcd
    def test_size_small(self):
        """Test: 8×8 (small size)"""
        x = create_int_tensor((8, 8), torch.int32)
        y = create_int_tensor((8, 8), torch.int32)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)

    @pytest.mark.gcd
    def test_size_medium_64(self):
        """Test: 64×64 (medium size)"""
        x = create_int_tensor((64, 64), torch.int32)
        y = create_int_tensor((64, 64), torch.int32)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)

    @pytest.mark.gcd
    def test_size_medium_256(self):
        """Test: 256×256 (medium size)"""
        x = create_int_tensor((256, 256), torch.int32)
        y = create_int_tensor((256, 256), torch.int32)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)

    @pytest.mark.gcd
    def test_size_large_1k(self):
        """Test: 1024×1024 (large size)"""
        x = create_int_tensor((1024, 1024), torch.int32)
        y = create_int_tensor((1024, 1024), torch.int32)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)

    @pytest.mark.gcd
    def test_size_large_4k(self):
        """Test: 4096×4096 (large size)"""
        x = create_int_tensor((4096, 4096), torch.int32)
        y = create_int_tensor((4096, 4096), torch.int32)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)


# ============================================================================
# 2. Input dimension coverage tests (competition requirement: cover all valid dimensions)
# ============================================================================


class TestGcdInputDimensions:
    """Test input dimension coverage: 1D, 2D, 3D, 4D"""

    @pytest.mark.gcd
    def test_dim_1d_small(self):
        """Test: 1D tensor (small size)"""
        x = create_int_tensor((10,), torch.int32)
        y = create_int_tensor((10,), torch.int32)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)

    @pytest.mark.gcd
    def test_dim_1d_large(self):
        """Test: 1D tensor (large size)"""
        x = create_int_tensor((10000,), torch.int32)
        y = create_int_tensor((10000,), torch.int32)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)

    @pytest.mark.gcd
    def test_dim_2d(self):
        """Test: 2D tensor"""
        x = create_int_tensor((100, 100), torch.int32)
        y = create_int_tensor((100, 100), torch.int32)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)

    @pytest.mark.gcd
    def test_dim_3d(self):
        """Test: 3D tensor"""
        x = create_int_tensor((64, 64, 64), torch.int32)
        y = create_int_tensor((64, 64, 64), torch.int32)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)

    @pytest.mark.gcd
    def test_dim_4d_batch(self):
        """Test: 4D batch tensor (batch × channel × height × width)"""
        x = create_int_tensor((16, 3, 128, 128), torch.int32)
        y = create_int_tensor((16, 3, 128, 128), torch.int32)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)


# ============================================================================
# 3. Data type coverage tests (competition requirement: support all integer types)
# ============================================================================


class TestGcdDataTypes:
    """Test data type coverage: int8, int16, int32, int64"""

    @pytest.mark.gcd
    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_all_int_dtypes(self, dtype):
        """Test: all integer data types"""
        x = create_int_tensor((256, 256), dtype)
        y = create_int_tensor((256, 256), dtype)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch, dtype=dtype)


# ============================================================================
# 4. Parameter pattern coverage tests (competition requirement: default values,
# boundary values, special values, mathematical properties)
# ============================================================================


class TestGcdParameterPatterns:
    """Test parameter patterns: default values, boundary values, special values, mathematical properties"""

    @pytest.mark.gcd
    def test_default_usage(self):
        """Test: default parameter usage"""
        x = create_int_tensor((100, 100), torch.int32)
        y = create_int_tensor((100, 100), torch.int32)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)

    @pytest.mark.gcd
    def test_edge_case_zero_values(self):
        """Test: boundary values - zero value handling"""
        # gcd(0, a) = a, gcd(a, 0) = a, gcd(0, 0) = 0
        x = torch.tensor([0, 5, 10, 0], device="cuda")
        y = torch.tensor([15, 0, 0, 0], device="cuda")
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        expected = torch.tensor([15, 5, 10, 0], device="cuda")
        assert_equal(z_gems, expected)
        assert_equal(z_torch, expected)

    @pytest.mark.gcd
    def test_special_negative_values(self):
        """Test: special values - negative values (gcd(a,b) = gcd(|a|,|b|))"""
        x = torch.tensor([-12, -18, 24], device="cuda")
        y = torch.tensor([8, -12, -36], device="cuda")
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        expected = torch.tensor([4, 6, 12], device="cuda")
        assert_equal(z_gems, expected)
        assert_equal(z_torch, expected)

    @pytest.mark.gcd
    def test_special_prime_numbers(self):
        """Test: special values - prime numbers (gcd = 1)"""
        x = torch.tensor([17, 19, 23], device="cuda")
        y = torch.tensor([13, 7, 11], device="cuda")
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        expected = torch.tensor([1, 1, 1], device="cuda")
        assert_equal(z_gems, expected)
        assert_equal(z_torch, expected)

    @pytest.mark.gcd
    def test_special_same_numbers(self):
        """Test: special values - same numbers (gcd(a, a) = a)"""
        x = torch.tensor([42, 100, 255], device="cuda")
        y = torch.tensor([42, 100, 255], device="cuda")
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        expected = torch.tensor([42, 100, 255], device="cuda")
        assert_equal(z_gems, expected)
        assert_equal(z_torch, expected)

    @pytest.mark.gcd
    def test_special_large_numbers(self):
        """Test: special values - large numbers"""
        x = torch.tensor([123456, 987654], device="cuda")
        y = torch.tensor([789012, 456789], device="cuda")
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch)


# ============================================================================
# 5. Functional completeness tests (competition requirement: all functional branches)
# ============================================================================


class TestGcdFunctionalCompleteness:
    """Test functional completeness: basic computation, zero value handling,
    negative value handling, mathematical properties"""

    @pytest.mark.gcd
    def test_basic_computation(self):
        """Test: basic element-wise computation"""
        x = torch.tensor([12, 18, 24, 35], device="cuda")
        y = torch.tensor([8, 12, 36, 49], device="cuda")
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        expected = torch.tensor([4, 6, 12, 7], device="cuda")
        assert_equal(z_gems, expected)
        assert_equal(z_torch, expected)

    @pytest.mark.gcd
    def test_commutative_property(self):
        """Test: commutative property gcd(a, b) = gcd(b, a)"""
        x = create_int_tensor((100, 100), torch.int32)
        y = create_int_tensor((100, 100), torch.int32)
        z_ab = gcd(x, y)
        z_ba = gcd(y, x)
        assert_equal(z_ab, z_ba)

    @pytest.mark.gcd
    def test_batch_processing(self):
        """Test: batch processing (multiple tensors)"""
        tensors_x = [
            create_int_tensor((64, 64), torch.int32),
            create_int_tensor((128, 128), torch.int32),
            create_int_tensor((256, 256), torch.int32),
        ]
        tensors_y = [
            create_int_tensor((64, 64), torch.int32),
            create_int_tensor((128, 128), torch.int32),
            create_int_tensor((256, 256), torch.int32),
        ]
        for x, y in zip(tensors_x, tensors_y):
            z_gems = gcd(x, y)
            z_torch = torch.gcd(x, y)
            assert_equal(z_gems, z_torch)

    @pytest.mark.gcd
    def test_consistency_with_pytorch(self):
        """Test: consistency with PyTorch implementation"""
        # Test multiple times using random tensors
        for _ in range(10):
            x = create_int_tensor((100, 100), torch.int32)
            y = create_int_tensor((100, 100), torch.int32)
            z_gems = gcd(x, y)
            z_torch = torch.gcd(x, y)
            assert_equal(z_gems, z_torch)


# ============================================================================
# 6. Comprehensive coverage tests (parameterized combinations)
# ============================================================================


class TestGcdComprehensive:
    """Comprehensive tests: multi-dimensional combination coverage"""

    @pytest.mark.gcd
    @pytest.mark.parametrize("shape", [8, 64 * 64, 256 * 256, 1024 * 1024])
    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_shape_dtype_combination(self, shape, dtype):
        """Test: combinations of shapes and data types"""
        x = create_int_tensor((shape,), dtype)
        y = create_int_tensor((shape,), dtype)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch, dtype=dtype)

    @pytest.mark.gcd
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((8, 8), torch.int8),
            ((64, 64), torch.int16),
            ((256, 256), torch.int32),
            ((1024, 1024), torch.int64),
        ],
    )
    def test_typical_use_cases(self, shape, dtype):
        """Test: typical use cases"""
        x = create_int_tensor(shape, dtype)
        y = create_int_tensor(shape, dtype)
        z_gems = gcd(x, y)
        z_torch = torch.gcd(x, y)
        assert_equal(z_gems, z_torch, dtype=dtype)
=======
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def make_gcd_tensor(shape, dtype, values):
    info = torch.iinfo(dtype)
    tensor = torch.randint(info.min, info.max, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    flat = tensor.reshape(-1)
    if flat.numel() > 0:
        boundary = torch.tensor(values, dtype=dtype, device=flag_gems.device)
        flat[: min(flat.numel(), boundary.numel())] = boundary[: flat.numel()]
    return tensor


@pytest.mark.gcd
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_gcd(shape, dtype):
    info = torch.iinfo(dtype)
    inp1 = make_gcd_tensor(shape, dtype, [0, -12, info.min, -27, 81])
    inp2 = make_gcd_tensor(shape, dtype, [6, 18, 0, -9, info.min])
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.gcd(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.gcd(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.gcd
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_gcd_out(shape, dtype):
    info = torch.iinfo(dtype)
    inp1 = make_gcd_tensor(shape, dtype, [0, -12, info.min, -27, 81])
    inp2 = make_gcd_tensor(shape, dtype, [6, 18, 0, -9, info.min])
    out = torch.empty_like(inp1)

    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)
    ref_out = torch.empty_like(ref_inp1)

    torch.gcd(ref_inp1, ref_inp2, out=ref_out)
    with flag_gems.use_gems():
        torch.gcd(inp1, inp2, out=out)

    utils.gems_assert_equal(out, ref_out)


@pytest.mark.gcd
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_gcd_special_values(dtype):
    info = torch.iinfo(dtype)
    inp1 = torch.tensor(
        [0, 0, -12, -27, info.min, info.min, info.min, info.max],
        dtype=dtype,
        device=flag_gems.device,
    )
    inp2 = torch.tensor(
        [0, 6, 18, -9, 0, info.min, 2, info.min],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.gcd(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.gcd(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.gcd
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_gcd_empty(dtype):
    inp1 = torch.empty((2, 0, 3), dtype=dtype, device=flag_gems.device)
    inp2 = torch.empty((2, 0, 3), dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.gcd(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.gcd(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.gcd
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_gcd_noncontiguous_broadcast(dtype):
    info = torch.iinfo(dtype)
    lhs = make_gcd_tensor((5, 7), dtype, [0, -12, info.min, -27, 81]).T
    rhs = make_gcd_tensor((1, 5), dtype, [6, 18, 0, -9, info.min])

    assert not lhs.is_contiguous()

    ref_lhs = utils.to_reference(lhs, False)
    ref_rhs = utils.to_reference(rhs, False)
    ref_out = torch.gcd(ref_lhs, ref_rhs)

    with flag_gems.use_gems():
        res_out = torch.gcd(lhs, rhs)

    utils.gems_assert_equal(res_out, ref_out)
>>>>>>> master
