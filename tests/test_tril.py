import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SHAPE_DIAGONAL = list(zip(utils.POINTWISE_SHAPES, [-2, -2, -1, 0, 1, 3]))


def _make_tril_input(shape, dtype):
    if dtype in utils.FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    elif dtype in utils.ALL_INT_DTYPES:
        inp = torch.randint(-8, 9, shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device=flag_gems.device)
    return utils.unsqueeze_tensor(inp, 2)


@pytest.mark.tril
@pytest.mark.parametrize("shape, diagonal", SHAPE_DIAGONAL)
@pytest.mark.parametrize(
    "dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)
def test_tril(shape, diagonal, dtype):
    inp = _make_tril_input(shape, dtype)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.is_contiguous(), "tril output should be contiguous"


@pytest.mark.tril
@pytest.mark.parametrize("shape, diagonal", SHAPE_DIAGONAL)
@pytest.mark.parametrize(
    "dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)
def test_tril_noncontiguous(shape, diagonal, dtype):
    inp = _make_tril_input(shape, dtype)
    inp = inp.transpose(-2, -1)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.is_contiguous(), "tril output should always be contiguous"


@pytest.mark.tril_out
@pytest.mark.parametrize("shape, diagonal", SHAPE_DIAGONAL)
@pytest.mark.parametrize(
    "dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)
def test_tril_out(shape, diagonal, dtype):
    inp = _make_tril_input(shape, dtype)
    out = torch.empty_like(inp)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.empty_like(ref_inp)
    torch.tril(ref_inp, diagonal, out=ref_out)

    with flag_gems.use_gems():
        res = torch.tril(inp, diagonal, out=out)

    utils.gems_assert_equal(out, ref_out)
    assert res.data_ptr() == out.data_ptr()


@pytest.mark.tril_out
@pytest.mark.parametrize("shape, diagonal", SHAPE_DIAGONAL[:3])
@pytest.mark.parametrize(
    "dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)
def test_tril_out_empty_output(shape, diagonal, dtype):
    inp = _make_tril_input(shape, dtype)
    out = torch.empty(0, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.empty(0, dtype=dtype, device=ref_inp.device)
    torch.tril(ref_inp, diagonal, out=ref_out)

    with flag_gems.use_gems():
        res = torch.tril(inp, diagonal, out=out)

    utils.gems_assert_equal(out, ref_out)
    assert out.shape == inp.shape
    assert res.data_ptr() == out.data_ptr()


@pytest.mark.tril
@pytest.mark.parametrize("shape", [(0, 0), (0, 7), (5, 0), (2, 0, 7), (2, 5, 0)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_tril_empty(shape, dtype):
    inp = _make_tril_input(shape, dtype)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, -1)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, -1)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.shape == inp.shape


@pytest.mark.tril
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tril_special_values(dtype):
    inp = torch.tensor(
        [
            [float("nan"), float("inf"), 1.0],
            [float("-inf"), -2.0, 3.0],
            [4.0, -5.0, float("nan")],
        ],
        dtype=dtype,
        device=flag_gems.device,
    )

    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal=0)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal=0)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.tril
def test_tril_invalid_rank():
    inp = torch.tensor(1.0, device=flag_gems.device)

    with flag_gems.use_gems(), pytest.raises(
        RuntimeError, match="tril: input tensor must have at least 2 dimensions"
    ):
        torch.tril(inp)
