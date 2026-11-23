# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .accuracy_utils import (
    BOOL_TYPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    gems_assert_equal,
    to_reference,
)
from .conftest import QUICK_MODE

INDEX_FILL_SHAPES = [(2, 32)] if QUICK_MODE else [(1, 2), (4, 8), (2, 3, 5)]
DIM_LIST = [1] if QUICK_MODE else [0, -1]
INDEX_CASES = ["normal", "negative", "scalar"]
_INDEX_FILL_DTYPES = FLOAT_DTYPES + INT_DTYPES + BOOL_TYPES
INDEX_FILL_DTYPES = [
    pytest.param(
        dtype,
        marks=pytest.mark.skipif(
            flag_gems.device == "npu" and dtype == torch.int16,
            reason="torch_npu does not support int16 index_fill reference",
        ),
    )
    for dtype in _INDEX_FILL_DTYPES
]
INDEX_FILL_OPS = [
    "index_fill",
    "index_fill_",
]
INDEX_FILL_OOB_PATHS = ("python_contiguous", "python_strided")


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
        return to_reference(value, False)
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

    ref_inp = to_reference(inp, False)
    ref_index = to_reference(index, False)
    ref_out = ref_inp.index_fill(dim, ref_index, value)

    res_out = flag_gems.index_fill(inp, dim, index, value)

    gems_assert_equal(res_out, ref_out)
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

    ref_inp = to_reference(inp.clone(), False)
    ref_index = to_reference(index, False)
    ref_inp.index_fill_(dim, ref_index, value)

    res_out = flag_gems.index_fill_(inp, dim, index, value)

    assert res_out is inp
    gems_assert_equal(inp, ref_inp)


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

    ref_inp = to_reference(inp, False)
    ref_index = to_reference(index, False)
    ref_value = _to_ref_value(value)
    ref_out = ref_inp.index_fill(1, ref_index, ref_value)

    res_out = flag_gems.index_fill(inp, 1, index, value)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.index_fill_
def test_index_fill_duplicate_index():
    inp = torch.arange(12, dtype=torch.float32, device=flag_gems.device).reshape(3, 4)
    index = torch.tensor([1, 1, -1], dtype=torch.long, device=flag_gems.device)
    ref_inp = to_reference(inp.clone(), False)
    ref_index = to_reference(index, False)
    ref_inp.index_fill_(1, ref_index, -7.0)

    flag_gems.index_fill_(inp, 1, index, -7.0)

    gems_assert_equal(inp, ref_inp)


@pytest.mark.index_fill_
@pytest.mark.parametrize("shape", [(2, 16321), (4, 151936)])
def test_index_fill_large_row_fill(shape):
    # A large inner size used to produce a 2-D (m, n) launch with more than 255
    # n-blocks, which backends such as Enflame GCU reject ("out of resource:
    # grid.y"). The fill must stay within the launch limits and stay exact.
    inp = torch.arange(
        shape[0] * shape[1], dtype=torch.float32, device=flag_gems.device
    ).reshape(shape)
    index = torch.tensor([0, shape[0] - 1], dtype=torch.long, device=flag_gems.device)

    ref_inp = to_reference(inp.clone(), False)
    ref_index = to_reference(index, False)
    ref_inp.index_fill_(0, ref_index, -5.0)

    res_out = flag_gems.index_fill_(inp, 0, index, -5.0)

    assert res_out is inp
    gems_assert_equal(inp, ref_inp)


@pytest.mark.index_fill_
def test_index_fill_grid_stride(monkeypatch):
    # Cap the flat launch below the number of tile blocks so the kernel's
    # grid-stride loop runs several iterations; this is what keeps oversized
    # fills legal on backends with small grid limits.
    index_fill_ops = importlib.import_module("flag_gems.ops.index_fill")

    monkeypatch.setattr(index_fill_ops, "_MAX_GRID", 3)

    shape = (256, 16)
    inp = torch.arange(
        shape[0] * shape[1], dtype=torch.float32, device=flag_gems.device
    ).reshape(shape)
    index = torch.arange(shape[0], dtype=torch.long, device=flag_gems.device)

    ref_inp = to_reference(inp.clone(), False)
    ref_index = to_reference(index, False)
    ref_inp.index_fill_(0, ref_index, -6.0)
    flag_gems.index_fill_(inp, 0, index, -6.0)
    gems_assert_equal(inp, ref_inp)

    # Same fill through the generic strided kernel (non-contiguous input).
    base = torch.arange(
        shape[1] * shape[0], dtype=torch.float32, device=flag_gems.device
    ).reshape(shape[1], shape[0])
    view = base.t()
    ref_view = to_reference(view.clone(), False)
    ref_view.index_fill_(0, ref_index, -6.0)
    flag_gems.index_fill_(view, 0, index, -6.0)
    gems_assert_equal(view, ref_view)


@pytest.mark.index_fill
def test_index_fill_noncontiguous():
    # Functional fill on a transposed view: the output is a strided clone
    # filled via the generic strided kernel, and the input is left untouched.
    base = torch.arange(12, dtype=torch.float32, device=flag_gems.device).reshape(3, 4)
    view = base.t()
    index = torch.tensor([0, -1], dtype=torch.long, device=flag_gems.device)

    ref_view = to_reference(view.clone(), False)
    ref_index = to_reference(index, False)
    ref_out = ref_view.index_fill(1, ref_index, -8.0)

    res_out = flag_gems.index_fill(view, 1, index, -8.0)

    gems_assert_equal(res_out, ref_out)
    assert res_out is not view
    assert torch.equal(
        base,
        torch.arange(12, dtype=torch.float32, device=flag_gems.device).reshape(3, 4),
    )


INDEX_FILL_BACKWARD_CASES = [((1, 2), 0), ((4, 8), 0), ((2, 3, 5), 2)]


@pytest.mark.index_fill_backward
@pytest.mark.parametrize("shape, dim", INDEX_FILL_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_index_fill_backward(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    index = torch.randperm(shape[dim], device=flag_gems.device)[: shape[dim] // 2]
    value = torch.tensor(2.0, dtype=dtype, device=flag_gems.device)
    value.requires_grad_(True)

    ref_inp = utils.to_reference(inp, True)
    ref_index = utils.to_reference(index)
    ref_value = utils.to_reference(value, True)

    ref_out = torch.index_fill(ref_inp, dim, ref_index, ref_value)
    res_out = flag_gems.index_fill(inp, dim, index, value)

    out_grad = torch.randn_like(res_out)
    ref_out_grad = utils.to_reference(out_grad, True)

    res_dinp, res_dval = torch.autograd.grad(res_out, (inp, value), out_grad)
    ref_dinp, ref_dval = torch.autograd.grad(
        ref_out, (ref_inp, ref_value), ref_out_grad
    )
    utils.gems_assert_close(res_dinp, ref_dinp, dtype)
    utils.gems_assert_close(res_dval, ref_dval, dtype)

    res_dinp_direct, res_dval_direct = flag_gems.index_fill_backward(
        out_grad, dim, index, True
    )
    utils.gems_assert_close(res_dinp_direct, ref_dinp, dtype)
    utils.gems_assert_close(res_dval_direct, ref_dval, dtype)
