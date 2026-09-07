import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.fill_diagonal_
@pytest.mark.parametrize(
    "shape",
    [(1, 1), (2, 3), (5, 2), (32, 32), (4, 4, 4), (0, 0), (0, 0, 0)],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("wrap", [False, True])
def test_accuracy_fill_diagonal_(shape, dtype, wrap):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref = utils.to_reference(inp.clone())
    expected = ref.fill_diagonal_(-2.5, wrap=wrap)
    data_ptr = inp.data_ptr()
    stride = inp.stride()

    with flag_gems.use_gems():
        result = inp.fill_diagonal_(-2.5, wrap=wrap)

    utils.gems_assert_equal(result, expected)
    assert result is inp
    assert result.data_ptr() == data_ptr
    assert result.stride() == stride


@pytest.mark.fill_diagonal_
@pytest.mark.parametrize("shape", [(7, 3), (10, 3), (128, 4)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_fill_diagonal__wrap(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    expected = utils.to_reference(inp.clone()).fill_diagonal_(5.0, wrap=True)

    with flag_gems.use_gems():
        result = inp.fill_diagonal_(5.0, wrap=True)

    utils.gems_assert_equal(result, expected)


@pytest.mark.fill_diagonal_
@pytest.mark.parametrize("wrap", [False, True])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_fill_diagonal__noncontiguous(dtype, wrap):
    inp = torch.randn((11, 6), dtype=dtype, device=flag_gems.device)[:, ::2]
    expected = utils.to_reference(inp.clone()).fill_diagonal_(1.25, wrap=wrap)
    original_stride = inp.stride()

    with flag_gems.use_gems():
        result = inp.fill_diagonal_(1.25, wrap=wrap)

    utils.gems_assert_equal(result, expected)
    assert result.stride() == original_stride


@pytest.mark.fill_diagonal_
def test_fill_diagonal__wrap_rejects_out_of_bounds_strided_view():
    inp = torch.randn((3, 11), device=flag_gems.device).T
    with flag_gems.use_gems(), pytest.raises(RuntimeError, match="out of bounds"):
        inp.fill_diagonal_(1.25, wrap=True)


@pytest.mark.fill_diagonal_
@pytest.mark.parametrize(
    ("dtype", "fill_value"),
    [(torch.int32, -3), (torch.int64, 7), (torch.bool, True)],
)
def test_accuracy_fill_diagonal__non_float(dtype, fill_value):
    inp = torch.zeros((9, 4), dtype=dtype, device=flag_gems.device)
    expected = utils.to_reference(inp.clone()).fill_diagonal_(fill_value, wrap=True)

    with flag_gems.use_gems():
        result = inp.fill_diagonal_(fill_value, wrap=True)

    utils.gems_assert_equal(result, expected)


@pytest.mark.fill_diagonal_
@pytest.mark.parametrize("shape", [(), (4,), (0,)])
def test_fill_diagonal__rejects_tensors_with_fewer_than_two_dims(shape):
    inp = torch.empty(shape, device=flag_gems.device)
    with (
        flag_gems.use_gems(),
        pytest.raises(RuntimeError, match="dimensions must larger than 1"),
    ):
        inp.fill_diagonal_(1)


@pytest.mark.fill_diagonal_
def test_fill_diagonal__rejects_unequal_higher_dimensions():
    inp = torch.empty((3, 3, 2), device=flag_gems.device)
    with (
        flag_gems.use_gems(),
        pytest.raises(
            RuntimeError, match="all dimensions of input must be of equal length"
        ),
    ):
        inp.fill_diagonal_(1)
