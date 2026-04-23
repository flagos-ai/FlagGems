import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.log10
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.log10_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_out(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.log10(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.empty_like(inp)
        torch.log10(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_special_values(dtype):
    inp = torch.tensor(
        [0.0, -0.0, 1.0, 10.0, -1.0, float("inf"), float("-inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_empty(dtype):
    shapes = ((0,), (4, 0), (2, 0, 3))
    for shape in shapes:
        inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp, True)

        ref_out = torch.log10(ref_inp)
        with flag_gems.use_gems():
            res_out = torch.log10(inp)

        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_noncontiguous(dtype):
    inp = torch.rand((32, 64), dtype=dtype, device=flag_gems.device).transpose(0, 1)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_log10_int_promotes_to_float(dtype):
    inp = torch.randint(1, 100, (128, 64), dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    utils.gems_assert_close(res_out, ref_out, torch.float32)
