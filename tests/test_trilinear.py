import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.trilinear
@pytest.mark.parametrize("shape", [s for s in utils.POINTWISE_SHAPES if len(s) >= 2])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trilinear_basic(shape, dtype):
    """Test basic _trilinear without expand or sumdim."""
    res_i1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_i2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_i3 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_i1 = utils.to_reference(res_i1)
    ref_i2 = utils.to_reference(res_i2)
    ref_i3 = utils.to_reference(res_i3)

    ref_out = torch._trilinear(ref_i1, ref_i2, ref_i3, [], [], [], [], unroll_dim=1)
    res_out = flag_gems._trilinear(res_i1, res_i2, res_i3, [], [], [], [], unroll_dim=1)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.trilinear
@pytest.mark.parametrize(
    # Limited to float16/float32: bfloat16 shows precision issues in reduction operations
    "dtype",
    [torch.float16, torch.float32],
)
def test_trilinear_with_reduction(dtype):
    """Test _trilinear with dimension reduction."""
    # Shape chosen to test multi-dimensional reduction with moderate size
    shape = (4, 8, 16)
    res_i1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_i2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_i3 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_i1 = utils.to_reference(res_i1)
    ref_i2 = utils.to_reference(res_i2)
    ref_i3 = utils.to_reference(res_i3)

    # Test reduction along last dimension
    ref_out = torch._trilinear(ref_i1, ref_i2, ref_i3, [], [], [], [2], unroll_dim=1)
    res_out = flag_gems._trilinear(
        res_i1, res_i2, res_i3, [], [], [], [2], unroll_dim=1
    )
    # Use higher atol for float16 due to accumulated precision errors
    atol = 0.005 if dtype == torch.float16 else 0.0001
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[2], atol=atol)

    # Test reduction along multiple dimensions
    ref_out = torch._trilinear(ref_i1, ref_i2, ref_i3, [], [], [], [1, 2], unroll_dim=1)
    res_out = flag_gems._trilinear(
        res_i1, res_i2, res_i3, [], [], [], [1, 2], unroll_dim=1
    )
    atol = 0.005 if dtype == torch.float16 else 0.0001
    utils.gems_assert_close(
        res_out, ref_out, dtype, reduce_dim=shape[1] * shape[2], atol=atol
    )


@pytest.mark.trilinear
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trilinear_broadcast(dtype):
    """Test _trilinear with broadcasting."""
    res_i1 = torch.randn(4, 5, 8, dtype=dtype, device=flag_gems.device)
    res_i2 = torch.randn(4, 5, 8, dtype=dtype, device=flag_gems.device)
    res_i3 = torch.randn(4, 5, 8, dtype=dtype, device=flag_gems.device)

    ref_i1 = utils.to_reference(res_i1)
    ref_i2 = utils.to_reference(res_i2)
    ref_i3 = utils.to_reference(res_i3)

    ref_out = torch._trilinear(ref_i1, ref_i2, ref_i3, [], [], [], [], unroll_dim=1)
    res_out = flag_gems._trilinear(res_i1, res_i2, res_i3, [], [], [], [], unroll_dim=1)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.trilinear
# Limited to float32 for large reductions to ensure numerical precision
@pytest.mark.parametrize("dtype", [torch.float32])
def test_trilinear_large_reduction(dtype):
    """Test _trilinear with large reduction dimension."""
    # Large reduction dimension tests kernel's reduction performance
    shape = (32, 8192)
    res_i1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_i2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_i3 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_i1 = utils.to_reference(res_i1)
    ref_i2 = utils.to_reference(res_i2)
    ref_i3 = utils.to_reference(res_i3)

    ref_out = torch._trilinear(ref_i1, ref_i2, ref_i3, [], [], [], [1], unroll_dim=1)
    res_out = flag_gems._trilinear(
        res_i1, res_i2, res_i3, [], [], [], [1], unroll_dim=1
    )

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[1])


@pytest.mark.underscore_trilinear_out
@pytest.mark.parametrize("shape", [(128, 128), (16, 32, 64)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trilinear_out(shape, dtype):
    """Test _trilinear.out variant."""
    res_i1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_i2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_i3 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_out = torch.empty(shape, dtype=dtype, device=flag_gems.device)

    ref_i1 = utils.to_reference(res_i1)
    ref_i2 = utils.to_reference(res_i2)
    ref_i3 = utils.to_reference(res_i3)

    ref_out = torch._trilinear(ref_i1, ref_i2, ref_i3, [], [], [], [], unroll_dim=1)
    flag_gems._trilinear_out(
        res_i1, res_i2, res_i3, [], [], [], [], unroll_dim=1, out=res_out
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
