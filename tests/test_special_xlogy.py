import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_xlogy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_xlogy(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.special.xlogy(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.special.xlogy(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.special_xlogy_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_xlogy_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.ops.aten.xlogy_(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.xlogy_(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    utils.gems_assert_close(inp1, ref_inp1, dtype, equal_nan=True)


@pytest.mark.special_xlogy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_xlogy_tensor_scalar(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    scalar = 2.0
    ref_inp1 = utils.to_reference(inp1, True)

    ref_out = torch.special.xlogy(ref_inp1, scalar)
    with flag_gems.use_gems():
        res_out = torch.special.xlogy(inp1, scalar)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.special_xlogy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_xlogy_scalar_tensor(shape, dtype):
    scalar = 1.0
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, True)

    ref_out = torch.special.xlogy(scalar, ref_inp1)
    with flag_gems.use_gems():
        res_out = torch.special.xlogy(scalar, inp1)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
