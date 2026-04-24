import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.asinh
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asinh(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.asinh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.asinh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.asinh_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asinh_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.asinh(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.empty_like(inp)
        torch.asinh(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.asinh_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asinh_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.asinh_()
    with flag_gems.use_gems():
        res_out = inp.asinh_()

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.asinh_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asinh_inplace_edge_cases(dtype):
    vals = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        10.0,
        -10.0,
        1e-4,
        -1e-4,
        1e4,
        -1e4,
        float("inf"),
        float("-inf"),
        float("nan"),
    ]
    inp = torch.tensor(vals, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.asinh_()
    with flag_gems.use_gems():
        res_out = inp.asinh_()

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.asinh_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asinh_inplace_non_contiguous(dtype):
    inp = torch.randn((33, 65), dtype=dtype, device=flag_gems.device).transpose(0, 1)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.asinh_()
    with flag_gems.use_gems():
        res_out = inp.asinh_()

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.asinh
@pytest.mark.parametrize(
    "shape",
    [(1, 1), (8, 8), (64, 64), (256, 256), (1024, 1024), (4096, 4096)],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asinh_various_sizes(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.asinh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.asinh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.asinh
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asinh_scalar(dtype):
    inp = torch.randn((), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.asinh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.asinh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.asinh
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asinh_edge_cases(dtype):
    vals = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        10.0,
        -10.0,
        1e-4,
        -1e-4,
        1e4,
        -1e4,
        float("inf"),
        float("-inf"),
        float("nan"),
    ]
    inp = torch.tensor(vals, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.asinh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.asinh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.asinh
def test_asinh_empty_tensor():
    inp = torch.empty(0, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.asinh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.asinh(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.asinh
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asinh_non_contiguous(dtype):
    inp = torch.randn((33, 65), dtype=dtype, device=flag_gems.device).transpose(0, 1)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.asinh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.asinh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.asinh_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asinh_out_non_contiguous(dtype):
    inp = torch.randn((33, 65), dtype=dtype, device=flag_gems.device).transpose(0, 1)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.asinh(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.empty_like(inp)
        torch.asinh(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)
