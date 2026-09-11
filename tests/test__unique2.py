import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.unique2
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test__unique2_basic(shape, dtype):
    """Test basic _unique2 with default parameters."""
    res_inp = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref_out, ref_inverse, ref_counts = torch._unique2(ref_inp)
    res_out, res_inverse, res_counts = flag_gems._unique2(res_inp)

    if dtype in utils.FLOAT_DTYPES:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)

    # inverse and counts should be empty by default
    assert (
        res_inverse is None and ref_inverse.numel() == 0
    ) or utils.gems_assert_equal(res_inverse, ref_inverse) is None
    assert (res_counts is None and ref_counts.numel() == 0) or utils.gems_assert_equal(
        res_counts, ref_counts
    ) is None


@pytest.mark.unique2
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test__unique2_return_inverse(shape, dtype):
    """Test _unique2 with return_inverse=True."""
    res_inp = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref_out, ref_inverse, ref_counts = torch._unique2(ref_inp, return_inverse=True)
    res_out, res_inverse, res_counts = flag_gems._unique2(res_inp, return_inverse=True)

    if dtype in utils.FLOAT_DTYPES:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)

    utils.gems_assert_equal(res_inverse, ref_inverse)
    assert (res_counts is None and ref_counts.numel() == 0) or utils.gems_assert_equal(
        res_counts, ref_counts
    ) is None


@pytest.mark.unique2
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test__unique2_return_counts(shape, dtype):
    """Test _unique2 with return_counts=True."""
    res_inp = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref_out, ref_inverse, ref_counts = torch._unique2(ref_inp, return_counts=True)
    res_out, res_inverse, res_counts = flag_gems._unique2(res_inp, return_counts=True)

    if dtype in utils.FLOAT_DTYPES:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)

    assert (
        res_inverse is None and ref_inverse.numel() == 0
    ) or utils.gems_assert_equal(res_inverse, ref_inverse) is None
    utils.gems_assert_equal(res_counts, ref_counts)


@pytest.mark.unique2
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test__unique2_return_inverse_counts(shape, dtype):
    """Test _unique2 with both return_inverse=True and return_counts=True."""
    res_inp = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref_out, ref_inverse, ref_counts = torch._unique2(
        ref_inp, return_inverse=True, return_counts=True
    )
    res_out, res_inverse, res_counts = flag_gems._unique2(
        res_inp, return_inverse=True, return_counts=True
    )

    if dtype in utils.FLOAT_DTYPES:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)

    utils.gems_assert_equal(res_inverse, ref_inverse)
    utils.gems_assert_equal(res_counts, ref_counts)


@pytest.mark.unique2
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test__unique2_large_reduce(dtype):
    """Test _unique2 with large tensors."""
    shape = (8192,)
    res_inp = torch.randint(0, 100, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref_out, ref_inverse, ref_counts = torch._unique2(
        ref_inp, return_inverse=True, return_counts=True
    )
    res_out, res_inverse, res_counts = flag_gems._unique2(
        res_inp, return_inverse=True, return_counts=True
    )

    if dtype in utils.FLOAT_DTYPES:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)

    utils.gems_assert_equal(res_inverse, ref_inverse)
    utils.gems_assert_equal(res_counts, ref_counts)


@pytest.mark.unique2
def test__unique2_edge_cases():
    """Test _unique2 with edge cases."""
    # All same values
    res_inp = torch.ones(100, dtype=torch.int64, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref_out, ref_inverse, ref_counts = torch._unique2(
        ref_inp, return_inverse=True, return_counts=True
    )
    res_out, res_inverse, res_counts = flag_gems._unique2(
        res_inp, return_inverse=True, return_counts=True
    )

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_inverse, ref_inverse)
    utils.gems_assert_equal(res_counts, ref_counts)

    # All unique values
    res_inp = torch.arange(100, dtype=torch.int64, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref_out, ref_inverse, ref_counts = torch._unique2(
        ref_inp, return_inverse=True, return_counts=True
    )
    res_out, res_inverse, res_counts = flag_gems._unique2(
        res_inp, return_inverse=True, return_counts=True
    )

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_inverse, ref_inverse)
    utils.gems_assert_equal(res_counts, ref_counts)


@pytest.mark.unique2
def test__unique2_float_special_values():
    """Test _unique2 with float special values."""
    # Test with inf, -inf, nan
    res_inp = torch.tensor(
        [1.0, 2.0, float("inf"), 2.0, 1.0, float("-inf"), float("inf")],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(res_inp)

    ref_out, ref_inverse, ref_counts = torch._unique2(
        ref_inp, return_inverse=True, return_counts=True
    )
    res_out, res_inverse, res_counts = flag_gems._unique2(
        res_inp, return_inverse=True, return_counts=True
    )

    utils.gems_assert_close(res_out, ref_out, torch.float32)
    utils.gems_assert_equal(res_inverse, ref_inverse)
    utils.gems_assert_equal(res_counts, ref_counts)
