import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.fill_tensor
@pytest.mark.parametrize("value", [0, 1, 9])
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fill_tensor(value, shape, dtype):
    x = torch.ones(shape, device=flag_gems.device, dtype=dtype)
    ref_x = utils.to_reference(x, False)

    value_tensor = torch.tensor(value, device=flag_gems.device, dtype=dtype)
    ref_value_tensor = utils.to_reference(value_tensor, False)
    ref_out_tensor = torch.fill(ref_x, ref_value_tensor)
    with flag_gems.use_gems():
        res_out_tensor = torch.fill(x, value_tensor)

    utils.gems_assert_equal(res_out_tensor, ref_out_tensor)


@pytest.mark.fill_scalar
@pytest.mark.parametrize("value", [0, 1, 9])
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fill_scalar(value, shape, dtype):
    x = torch.ones(shape, device=flag_gems.device, dtype=dtype)
    ref_x = utils.to_reference(x, False)

    ref_out = torch.fill(ref_x, value)
    with flag_gems.use_gems():
        res_out = torch.fill(x, value)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.fill_tensor_out
@pytest.mark.parametrize("value", [0, 1, 9])
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fill_tensor_out(value, shape, dtype):
    x = torch.ones(shape, device=flag_gems.device, dtype=dtype)
    ref_x = utils.to_reference(x, False)

    value_tensor = torch.tensor(value, device=flag_gems.device, dtype=dtype)
    ref_value_tensor = utils.to_reference(value_tensor, False)
    out_tensor = torch.empty_like(x)
    ref_out_tensor = torch.empty_like(ref_x)

    ref_result_tensor = torch.ops.aten.fill.Tensor_out(
        ref_x, ref_value_tensor, out=ref_out_tensor
    )
    with flag_gems.use_gems():
        res_result_tensor = torch.ops.aten.fill.Tensor_out(
            x, value_tensor, out=out_tensor
        )

    utils.gems_assert_equal(res_result_tensor, ref_result_tensor)
    assert (
        res_result_tensor is out_tensor
    ), "fill.Tensor_out should return the out tensor"


@pytest.mark.fill_scalar_out
@pytest.mark.parametrize("value", [0, 1, 9])
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fill_scalar_out(value, shape, dtype):
    x = torch.ones(shape, device=flag_gems.device, dtype=dtype)
    ref_x = utils.to_reference(x, False)
    out = torch.empty_like(x)
    ref_out = torch.empty_like(ref_x)

    ref_result = torch.ops.aten.fill.Scalar_out(ref_x, value, out=ref_out)
    with flag_gems.use_gems():
        res_result = torch.ops.aten.fill.Scalar_out(x, value, out=out)

    utils.gems_assert_equal(res_result, ref_result)
    assert res_result is out, "fill.Scalar_out should return the out tensor"
