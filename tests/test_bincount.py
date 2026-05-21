import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

BINCOUNT_SIZES = [16, 100, 1024, 10000] if not QUICK_MODE else [100, 1024]
BINCOUNT_MAXVALS = [10, 100, 1000] if not QUICK_MODE else [100]


@pytest.mark.bincount
@pytest.mark.parametrize("size", BINCOUNT_SIZES)
@pytest.mark.parametrize("max_val", BINCOUNT_MAXVALS)
def test_accuracy_bincount(size, max_val):
    """Test bincount without weights."""
    inp = torch.randint(0, max_val, (size,), dtype=torch.int64, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.bincount(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.bincount(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bincount
@pytest.mark.parametrize("size", BINCOUNT_SIZES)
@pytest.mark.parametrize("max_val", BINCOUNT_MAXVALS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_bincount_with_weights(size, max_val, dtype):
    """Test bincount with weights."""
    inp = torch.randint(0, max_val, (size,), dtype=torch.int64, device=flag_gems.device)
    weights = torch.randn(size, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_weights = utils.to_reference(weights)

    ref_out = torch.bincount(ref_inp, weights=ref_weights)
    with flag_gems.use_gems():
        res_out = torch.bincount(inp, weights=weights)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.bincount
@pytest.mark.parametrize("size", BINCOUNT_SIZES)
@pytest.mark.parametrize("max_val", BINCOUNT_MAXVALS)
@pytest.mark.parametrize("minlength", [0, 50, 2000])
def test_accuracy_bincount_with_minlength(size, max_val, minlength):
    """Test bincount with minlength parameter."""
    inp = torch.randint(0, max_val, (size,), dtype=torch.int64, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.bincount(ref_inp, minlength=minlength)
    with flag_gems.use_gems():
        res_out = torch.bincount(inp, minlength=minlength)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bincount
def test_accuracy_bincount_empty():
    """Test bincount with empty input."""
    inp = torch.tensor([], dtype=torch.int64, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.bincount(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.bincount(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bincount
def test_accuracy_bincount_single():
    """Test bincount with single element."""
    inp = torch.tensor([5], dtype=torch.int64, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.bincount(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.bincount(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bincount
def test_accuracy_bincount_all_zeros():
    """Test bincount with all zeros."""
    inp = torch.zeros(100, dtype=torch.int64, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.bincount(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.bincount(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bincount
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_bincount_weights_edge_cases(dtype):
    """Test bincount with edge case weights."""
    inp = torch.tensor([0, 1, 2, 1, 0], dtype=torch.int64, device=flag_gems.device)
    weights = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype, device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)
    ref_weights = utils.to_reference(weights)

    ref_out = torch.bincount(ref_inp, weights=ref_weights)
    with flag_gems.use_gems():
        res_out = torch.bincount(inp, weights=weights)

    utils.gems_assert_close(res_out, ref_out, dtype)
