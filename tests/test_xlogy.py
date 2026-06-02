import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.xlogy
@pytest.mark.parametrize("shape", [(8,), (16, 16), (32, 32, 32)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_basic(shape, dtype):
    """Test xlogy with various shapes and dtypes."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device).abs() + 0.1

    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)
    ref_out = torch.ops.aten.xlogy(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = flag_gems.xlogy(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.xlogy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_zero_x(dtype):
    """Test xlogy when x=0 should return 0 regardless of y."""
    x = torch.zeros(16, dtype=dtype, device=flag_gems.device)
    y = torch.randn(16, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)
    ref_out = torch.ops.aten.xlogy(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = flag_gems.xlogy(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.xlogy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_zero_y(dtype):
    """Test xlogy when y=0: should produce -inf (or 0 if x=0)."""
    x = torch.tensor([1.0, 0.0, 2.0, 0.0], dtype=dtype, device=flag_gems.device)
    y = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)
    ref_out = torch.ops.aten.xlogy(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = flag_gems.xlogy(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.xlogy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_negative_y(dtype):
    """Test xlogy with negative y values."""
    x = torch.randn(32, dtype=dtype, device=flag_gems.device)
    y = torch.randn(32, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)
    ref_out = torch.ops.aten.xlogy(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = flag_gems.xlogy(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.xlogy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_broadcast(dtype):
    """Test xlogy with broadcasting (scalar y)."""
    x = torch.randn(4, 6, dtype=dtype, device=flag_gems.device)
    y = torch.tensor(2.0, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)
    ref_out = torch.ops.aten.xlogy(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = flag_gems.xlogy(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.xlogy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_non_contiguous(dtype):
    """Test xlogy with non-contiguous input."""
    x = torch.randn(4, 8, dtype=dtype, device=flag_gems.device)[:, ::2]
    y = torch.randn(4, 4, dtype=dtype, device=flag_gems.device).abs() + 0.1

    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)
    ref_out = torch.ops.aten.xlogy(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = flag_gems.xlogy(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)
