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

import pytest
import torch

import flag_gems

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

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res_out = inp.index_fill(dim, index, value)

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

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res_out = inp.index_fill_(dim, index, value)

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

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res_out = inp.index_fill(1, index, value)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.index_fill_
def test_index_fill_duplicate_index():
    inp = torch.arange(12, dtype=torch.float32, device=flag_gems.device).reshape(3, 4)
    index = torch.tensor([1, 1, -1], dtype=torch.long, device=flag_gems.device)
    ref_inp = to_reference(inp.clone(), False)
    ref_index = to_reference(index, False)
    ref_inp.index_fill_(1, ref_index, -7.0)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        inp.index_fill_(1, index, -7.0)

    gems_assert_equal(inp, ref_inp)


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

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res_out = view.index_fill(1, index, -8.0)

    gems_assert_equal(res_out, ref_out)
    assert res_out is not view
    assert torch.equal(
        base,
        torch.arange(12, dtype=torch.float32, device=flag_gems.device).reshape(3, 4),
    )
