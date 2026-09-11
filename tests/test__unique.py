import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.unique
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__unique_basic(shape, dtype):
    """Test _unique with default parameters (sorted=True, return_inverse=False)"""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out, ref_inverse = torch._unique(ref_inp)
    res_out, res_inverse = flag_gems._unique(inp)

    utils.gems_assert_equal(res_out, ref_out)
    # When return_inverse=False, inverse should be empty
    assert res_inverse.numel() == 0
    assert ref_inverse.numel() == 0


@pytest.mark.unique
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__unique_with_inverse(shape, dtype):
    """Test _unique with return_inverse=True"""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out, ref_inverse = torch._unique(ref_inp, return_inverse=True)
    res_out, res_inverse = flag_gems._unique(inp, return_inverse=True)

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_inverse, ref_inverse)


@pytest.mark.unique
@pytest.mark.parametrize("shape", [(100,), (1000,)])
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test__unique_integers(shape, dtype):
    """Test _unique with integer tensors"""
    inp = torch.randint(0, 50, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out, ref_inverse = torch._unique(ref_inp, return_inverse=True)
    res_out, res_inverse = flag_gems._unique(inp, return_inverse=True)

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_inverse, ref_inverse)


@pytest.mark.unique
def test__unique_all_same():
    """Test _unique when all elements are the same"""
    inp = torch.ones(100, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out, ref_inverse = torch._unique(ref_inp, return_inverse=True)
    res_out, res_inverse = flag_gems._unique(inp, return_inverse=True)

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_inverse, ref_inverse)
    assert res_out.numel() == 1


@pytest.mark.unique
def test__unique_already_unique():
    """Test _unique when all elements are already unique"""
    inp = torch.arange(100, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out, ref_inverse = torch._unique(ref_inp, return_inverse=True)
    res_out, res_inverse = flag_gems._unique(inp, return_inverse=True)

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_inverse, ref_inverse)


@pytest.mark.unique
def test__unique_with_duplicates():
    """Test _unique with specific duplicates pattern"""
    inp = torch.tensor([1, 2, 2, 3, 1, 4], dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out, ref_inverse = torch._unique(ref_inp, return_inverse=True)
    res_out, res_inverse = flag_gems._unique(inp, return_inverse=True)

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_inverse, ref_inverse)
