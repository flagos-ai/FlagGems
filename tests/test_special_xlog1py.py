import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_xlog1py
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_xlog1py(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = inp2.clamp(min=-1.0 + 1e-6)
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.ops.aten.special_xlog1py(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.special_xlog1py(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_xlog1py
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_xlog1py_x_zero(dtype):
    """x == 0 and y is not NaN => result is 0."""
    x = torch.zeros((10,), dtype=dtype, device=flag_gems.device)
    y = torch.linspace(-0.5, 2.0, 10, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.ops.aten.special_xlog1py(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.special_xlog1py(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_xlog1py
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_xlog1py_y_nan(dtype):
    """y is NaN => result is NaN."""
    x = torch.randn((10,), dtype=dtype, device=flag_gems.device)
    y = torch.full((10,), float("nan"), dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.ops.aten.special_xlog1py(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.special_xlog1py(x, y)

    assert torch.isnan(res_out).all(), "Expected all-NaN output when y is NaN"
    assert torch.isnan(ref_out).all(), "Reference should also produce NaN"
