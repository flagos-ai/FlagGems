import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE


INDEX_FILL_SHAPES = (
    [(2, 32)] if QUICK_MODE else [(1, 2), (4, 8), (2, 3, 5)]
)
DIM_LIST = [1] if QUICK_MODE else [0, -1]
INDEX_CASES = ["normal", "negative", "scalar"]
INDEX_FILL_DTYPES = utils.FLOAT_DTYPES + utils.INT_DTYPES + utils.BOOL_TYPES
INDEX_FILL_OPS = [
    "index_fill_scalar",
    "index_fill_scalar_",
    "index_fill_scalar_out",
    "index_fill_tensor",
    "index_fill_tensor_",
    "index_fill_tensor_out",
]


def _make_input(shape, dtype):
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=flag_gems.device).bool()
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(-10, 10, shape, dtype=dtype, device=flag_gems.device)


def _scalar_value(dtype):
    if dtype == torch.bool:
        return True
    if dtype.is_floating_point:
        return -3.5
    return -3


def _make_index(dim_size, case):
    if case == "normal":
        values = [0, dim_size - 1] if dim_size > 1 else [0]
        return torch.tensor(values, dtype=torch.long, device=flag_gems.device)
    if case == "negative":
        return torch.tensor([-1], dtype=torch.long, device=flag_gems.device)
    if case == "scalar":
        return torch.tensor(0, dtype=torch.long, device=flag_gems.device)
    raise ValueError(f"Unknown index case: {case}")


def _to_ref_value(value):
    if isinstance(value, torch.Tensor):
        return utils.to_reference(value, False)
    return value


@pytest.mark.index_fill
@pytest.mark.parametrize("shape", INDEX_FILL_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", INDEX_FILL_DTYPES)
@pytest.mark.parametrize("index_case", INDEX_CASES)
def test_index_fill_scalar(shape, dim, dtype, index_case):
    inp = _make_input(shape, dtype)
    dim = dim % inp.ndim
    index = _make_index(inp.size(dim), index_case)
    value = _scalar_value(dtype)

    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    ref_out = ref_inp.index_fill(dim, ref_index, value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res_out = inp.index_fill(dim, index, value)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out is not inp


@pytest.mark.index_fill_
@pytest.mark.parametrize("shape", INDEX_FILL_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", INDEX_FILL_DTYPES)
@pytest.mark.parametrize("index_case", INDEX_CASES)
def test_index_fill_scalar_(shape, dim, dtype, index_case):
    inp = _make_input(shape, dtype)
    dim = dim % inp.ndim
    index = _make_index(inp.size(dim), index_case)
    value = _scalar_value(dtype)

    ref_inp = utils.to_reference(inp.clone(), False)
    ref_index = utils.to_reference(index, False)
    ref_inp.index_fill_(dim, ref_index, value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res_out = inp.index_fill_(dim, index, value)

    assert res_out is inp
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.index_fill
@pytest.mark.parametrize("dtype", INDEX_FILL_DTYPES)
@pytest.mark.parametrize("value_device", ["device", "cpu"])
def test_index_fill_tensor_value(dtype, value_device):
    inp = _make_input((3, 4), dtype)
    index = torch.tensor([1, -1], dtype=torch.long, device=flag_gems.device)
    value = torch.tensor(
        _scalar_value(dtype),
        dtype=dtype,
        device=flag_gems.device if value_device == "device" else "cpu",
    )

    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    ref_value = _to_ref_value(value)
    ref_out = ref_inp.index_fill(1, ref_index, ref_value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res_out = inp.index_fill(1, index, value)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.index_fill_
def test_index_fill_duplicate_index():
    inp = torch.arange(12, dtype=torch.float32, device=flag_gems.device).reshape(3, 4)
    index = torch.tensor([1, 1, -1], dtype=torch.long, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), False)
    ref_index = utils.to_reference(index, False)
    ref_inp.index_fill_(1, ref_index, -7.0)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        inp.index_fill_(1, index, -7.0)

    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.index_fill_
def test_index_fill_empty_index_noop():
    inp = torch.arange(12, dtype=torch.float32, device=flag_gems.device).reshape(3, 4)
    index = torch.empty(0, dtype=torch.long, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), False)
    ref_index = utils.to_reference(index, False)
    ref_inp.index_fill_(1, ref_index, -7.0)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        inp.index_fill_(1, index, -7.0)

    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.index_fill_
def test_index_fill_noncontiguous_view():
    base = torch.arange(12, dtype=torch.float32, device=flag_gems.device).reshape(3, 4)
    ref_base = utils.to_reference(base.clone(), False)
    res_base = base.clone()
    ref_view = ref_base.t()
    res_view = res_base.t()
    index = torch.tensor([0, -1], dtype=torch.long, device=flag_gems.device)
    ref_index = utils.to_reference(index, False)

    ref_view.index_fill_(1, ref_index, -8.0)
    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res = res_view.index_fill_(1, index, -8.0)

    assert res is res_view
    utils.gems_assert_equal(res_base, ref_base)


@pytest.mark.index_fill_out
@pytest.mark.parametrize("dtype", INDEX_FILL_DTYPES)
def test_index_fill_scalar_out(dtype):
    inp = _make_input((3, 4), dtype)
    index = torch.tensor([0, -1], dtype=torch.long, device=flag_gems.device)
    value = _scalar_value(dtype)
    out = torch.empty_like(inp)
    ref_out = torch.empty_like(utils.to_reference(inp, False))

    ref = torch.ops.aten.index_fill.int_Scalar_out(
        utils.to_reference(inp, False),
        1,
        utils.to_reference(index, False),
        value,
        out=ref_out,
    )
    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res = torch.ops.aten.index_fill.int_Scalar_out(inp, 1, index, value, out=out)

    assert res is out
    utils.gems_assert_equal(res, ref)


@pytest.mark.index_fill_out
@pytest.mark.parametrize("dtype", INDEX_FILL_DTYPES)
def test_index_fill_tensor_out(dtype):
    inp = _make_input((3, 4), dtype)
    index = torch.tensor([0, -1], dtype=torch.long, device=flag_gems.device)
    value = torch.tensor(_scalar_value(dtype), dtype=dtype, device=flag_gems.device)
    out = torch.empty_like(inp)
    ref_out = torch.empty_like(utils.to_reference(inp, False))

    ref = torch.ops.aten.index_fill.int_Tensor_out(
        utils.to_reference(inp, False),
        1,
        utils.to_reference(index, False),
        utils.to_reference(value, False),
        out=ref_out,
    )
    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res = torch.ops.aten.index_fill.int_Tensor_out(inp, 1, index, value, out=out)

    assert res is out
    utils.gems_assert_equal(res, ref)


@pytest.mark.index_fill_
def test_index_fill_invalid_index_dtype():
    inp = torch.randn((3, 4), device=flag_gems.device)
    index = torch.tensor([1], dtype=torch.int32, device=flag_gems.device)
    with flag_gems.use_gems(include=INDEX_FILL_OPS), pytest.raises(
        IndexError, match="Expected dtype int64"
    ):
        inp.index_fill_(1, index, -1.0)


@pytest.mark.index_fill_
def test_index_fill_invalid_index_ndim():
    inp = torch.randn((3, 4), device=flag_gems.device)
    index = torch.tensor([[1]], dtype=torch.long, device=flag_gems.device)
    with flag_gems.use_gems(include=INDEX_FILL_OPS), pytest.raises(
        IndexError, match="Index is supposed to be a vector"
    ):
        inp.index_fill_(1, index, -1.0)


@pytest.mark.index_fill_
def test_index_fill_out_of_range_index(monkeypatch):
    monkeypatch.setenv("FLAG_GEMS_INDEX_FILL_BOUNDS_CHECK", "sync")
    inp = torch.randn((3, 4), device=flag_gems.device)
    index = torch.tensor([4], dtype=torch.long, device=flag_gems.device)
    with flag_gems.use_gems(include=INDEX_FILL_OPS), pytest.raises(
        IndexError, match="index out of range"
    ):
        inp.index_fill_(1, index, -1.0)


@pytest.mark.index_fill_
def test_index_fill_invalid_tensor_value_ndim():
    inp = torch.randn((3, 4), device=flag_gems.device)
    index = torch.tensor([1], dtype=torch.long, device=flag_gems.device)
    value = torch.tensor([1.0], device=flag_gems.device)
    with flag_gems.use_gems(include=INDEX_FILL_OPS), pytest.raises(
        RuntimeError, match="0-dimensional value tensor"
    ):
        inp.index_fill_(1, index, value)


@pytest.mark.index_fill_
@pytest.mark.skipif(
    flag_gems.device == "cpu", reason="device mismatch requires device backend"
)
def test_index_fill_cpu_index_rejected():
    inp = torch.randn((3, 4), device=flag_gems.device)
    index = torch.tensor([1], dtype=torch.long, device="cpu")
    with flag_gems.use_gems(include=INDEX_FILL_OPS), pytest.raises(
        RuntimeError, match="same device"
    ):
        inp.index_fill_(1, index, -1.0)
