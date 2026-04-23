import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.cosh
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cosh(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.cosh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.cosh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cosh_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cosh_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.cosh_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.cosh_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cosh_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cosh_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.cosh(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.empty_like(inp)
        torch.cosh(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cosh
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cosh_special_values(dtype):
    inp = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.cosh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.cosh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.cosh
@pytest.mark.parametrize("shape", [(0,), (4, 0), (2, 0, 3)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cosh_empty(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.cosh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.cosh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cosh
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cosh_even_property(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        pos = torch.cosh(inp)
        neg = torch.cosh(-inp)

    ref_pos = utils.to_reference(pos, True)
    utils.gems_assert_close(neg, ref_pos, dtype)


@pytest.mark.cosh
@pytest.mark.parametrize("shape", [(17, 33), (5, 7, 9)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cosh_noncontiguous(shape, dtype):
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = base.transpose(-1, -2)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.cosh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.cosh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
