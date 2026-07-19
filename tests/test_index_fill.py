import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

INDEX_FILL_CASES = [
    ((8,), 0, [0, 3, 7]),
    ((4, 5), 1, [0, 2, 4]),
    ((3, 4, 5), -2, [1, 3]),
    ((2, 3, 4, 5), 2, [0, 3]),
]

INDEX_FILL_DTYPES = utils.FLOAT_DTYPES + [torch.int32, torch.bool]


def _make_input(shape, dtype):
    if dtype in utils.FLOAT_DTYPES:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(-20, 20, shape, dtype=dtype, device=flag_gems.device)


def _scalar_value(dtype):
    if dtype is torch.bool:
        return True
    if dtype in utils.FLOAT_DTYPES:
        return 2.5
    return 7


def _assert_matches(actual, expected, dtype):
    if dtype in utils.FLOAT_DTYPES:
        utils.gems_assert_close(actual, expected, dtype=dtype)
    else:
        utils.gems_assert_equal(actual, expected)


@pytest.mark.index_fill
@pytest.mark.parametrize("shape, dim, index_values", INDEX_FILL_CASES)
@pytest.mark.parametrize("dtype", INDEX_FILL_DTYPES)
def test_index_fill_scalar(shape, dim, index_values, dtype):
    inp = _make_input(shape, dtype)
    index = torch.tensor(index_values, dtype=torch.int64, device=flag_gems.device)
    value = _scalar_value(dtype)

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_out = torch.index_fill(ref_inp, dim, ref_index, value)

    with flag_gems.use_gems():
        res_out = torch.index_fill(inp, dim, index, value)

    _assert_matches(res_out, ref_out, dtype)
    assert res_out.stride() == inp.stride()


@pytest.mark.index_fill
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_fill_tensor_value(dtype):
    inp = _make_input((3, 4, 5), dtype)
    index = torch.tensor([0, 2], dtype=torch.int64, device=flag_gems.device)
    value = torch.tensor(-1.25, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_value = utils.to_reference(value)
    ref_out = torch.ops.aten.index_fill.int_Tensor(ref_inp, 1, ref_index, ref_value)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.index_fill.int_Tensor(inp, 1, index, value)

    _assert_matches(res_out, ref_out, dtype)


@pytest.mark.index_fill_
@pytest.mark.parametrize("shape, dim, index_values", INDEX_FILL_CASES[:3])
@pytest.mark.parametrize("dtype", INDEX_FILL_DTYPES)
def test_index_fill_inplace(shape, dim, index_values, dtype):
    inp = _make_input(shape, dtype)
    index = torch.tensor(index_values, dtype=torch.int64, device=flag_gems.device)
    value = _scalar_value(dtype)

    ref_inp = utils.to_reference(inp.clone())
    ref_index = utils.to_reference(index)
    ref_inp.index_fill_(dim, ref_index, value)

    original_data_ptr = inp.data_ptr()
    with flag_gems.use_gems():
        res = inp.index_fill_(dim, index, value)

    _assert_matches(inp, ref_inp, dtype)
    assert res is inp
    assert inp.data_ptr() == original_data_ptr


@pytest.mark.index_fill_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_fill_out_resizes_and_matches(dtype):
    inp = _make_input((3, 4, 5), dtype)
    index = torch.tensor([1, 3], dtype=torch.int64, device=flag_gems.device)
    value = _scalar_value(dtype)

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_out = torch.empty(0, dtype=ref_inp.dtype, device=ref_inp.device)
    torch.ops.aten.index_fill.int_Scalar_out(ref_inp, 1, ref_index, value, out=ref_out)

    out = torch.empty(0, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res = torch.ops.aten.index_fill.int_Scalar_out(inp, 1, index, value, out=out)

    _assert_matches(out, ref_out, dtype)
    assert out.shape == inp.shape
    assert res.data_ptr() == out.data_ptr()


@pytest.mark.index_fill_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_fill_out_noncontiguous_out(dtype):
    inp = _make_input((3, 4), dtype)
    index = torch.tensor([0, 2], dtype=torch.int64, device=flag_gems.device)
    value = _scalar_value(dtype)

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_base = torch.empty((3, 8), dtype=ref_inp.dtype, device=ref_inp.device)
    ref_out = ref_base[:, ::2]
    torch.ops.aten.index_fill.int_Scalar_out(ref_inp, 1, ref_index, value, out=ref_out)

    out_base = torch.empty((3, 8), dtype=dtype, device=flag_gems.device)
    out = out_base[:, ::2]
    assert not out.is_contiguous()

    with flag_gems.use_gems():
        res = torch.ops.aten.index_fill.int_Scalar_out(inp, 1, index, value, out=out)

    _assert_matches(out, ref_out, dtype)
    assert res.data_ptr() == out.data_ptr()
    assert out.stride() == (8, 2)


@pytest.mark.index_fill
@pytest.mark.parametrize("dtype", INDEX_FILL_DTYPES)
def test_index_fill_empty_index(dtype):
    inp = _make_input((3, 4), dtype)
    index = torch.empty((0,), dtype=torch.int64, device=flag_gems.device)
    value = _scalar_value(dtype)

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_out = torch.index_fill(ref_inp, 1, ref_index, value)

    with flag_gems.use_gems():
        res_out = torch.index_fill(inp, 1, index, value)

    _assert_matches(res_out, ref_out, dtype)


@pytest.mark.index_fill
@pytest.mark.parametrize("dtype", INDEX_FILL_DTYPES)
def test_index_fill_negative_indices(dtype):
    inp = _make_input((3, 4), dtype)
    index = torch.tensor([-1, -4], dtype=torch.int64, device=flag_gems.device)
    value = _scalar_value(dtype)

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_out = torch.index_fill(ref_inp, 1, ref_index, value)

    with flag_gems.use_gems():
        res_out = torch.index_fill(inp, 1, index, value)

    _assert_matches(res_out, ref_out, dtype)


@pytest.mark.index_fill
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_fill_noncontiguous_input(dtype):
    base = _make_input((5, 4), dtype)
    inp = base.t()
    index = torch.tensor([0, 2, 4], dtype=torch.int64, device=flag_gems.device)
    value = _scalar_value(dtype)
    assert not inp.is_contiguous()

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_out = torch.index_fill(ref_inp, 1, ref_index, value)

    with flag_gems.use_gems():
        res_out = torch.index_fill(inp, 1, index, value)

    _assert_matches(res_out, ref_out, dtype)
    assert res_out.stride() == inp.stride()


@pytest.mark.index_fill
def test_index_fill_invalid_index_rank_raises():
    inp = torch.zeros((3, 4), device=flag_gems.device)
    index = torch.zeros((1, 1), dtype=torch.int64, device=flag_gems.device)

    with flag_gems.use_gems():
        with pytest.raises(RuntimeError, match="Index has to be a vector/scalar"):
            torch.index_fill(inp, 1, index, 1.0)


@pytest.mark.index_fill
@pytest.mark.parametrize("index_values", ([4], [-5]))
def test_index_fill_out_of_bounds_raises(index_values):
    inp = torch.zeros((3, 4), device=flag_gems.device)
    index = torch.tensor(index_values, dtype=torch.int64, device=flag_gems.device)

    with flag_gems.use_gems():
        with pytest.raises(IndexError, match="out of bounds"):
            torch.index_fill(inp, 1, index, 1.0)
