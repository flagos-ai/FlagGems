import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.heaviside
@pytest.mark.parametrize("shape", [(8,), (16, 16), (32, 32, 32)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_heaviside_basic(shape, dtype):
    """Test heaviside with various shapes and dtypes."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    values = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_values = utils.to_reference(values)
    ref_out = torch.ops.aten.heaviside(ref_x, ref_values)

    with flag_gems.use_gems():
        res_out = flag_gems.heaviside(x, values)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.heaviside
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_heaviside_zero_input(dtype):
    """Test heaviside with zero inputs (should use values)."""
    x = torch.tensor([-1.0, -0.5, 0.0, 0.0, 0.5, 1.0], dtype=dtype, device=flag_gems.device)
    values = torch.tensor([0.0, 0.0, 0.5, 1.0, 0.0, 0.0], dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_values = utils.to_reference(values)
    ref_out = torch.ops.aten.heaviside(ref_x, ref_values)

    with flag_gems.use_gems():
        res_out = flag_gems.heaviside(x, values)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.heaviside
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_heaviside_scalar_values(dtype):
    """Test heaviside with scalar values (broadcast)."""
    x = torch.randn(4, 6, dtype=dtype, device=flag_gems.device)
    values = torch.tensor(0.5, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_values = utils.to_reference(values)
    ref_out = torch.ops.aten.heaviside(ref_x, ref_values)

    with flag_gems.use_gems():
        res_out = flag_gems.heaviside(x, values)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.heaviside
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_heaviside_non_contiguous(dtype):
    """Test heaviside with non-contiguous input."""
    x = torch.randn(4, 8, dtype=dtype, device=flag_gems.device)[:, ::2]
    values = torch.randn(4, 4, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_values = utils.to_reference(values)
    ref_out = torch.ops.aten.heaviside(ref_x, ref_values)

    with flag_gems.use_gems():
        res_out = flag_gems.heaviside(x, values)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.heaviside
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_heaviside_negative(dtype):
    """Test heaviside with all negative inputs (should return 0)."""
    x = -torch.rand(16, dtype=dtype, device=flag_gems.device) - 0.01
    values = torch.ones(16, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_values = utils.to_reference(values)
    ref_out = torch.ops.aten.heaviside(ref_x, ref_values)

    with flag_gems.use_gems():
        res_out = flag_gems.heaviside(x, values)

    utils.gems_assert_close(res_out, ref_out, dtype)
