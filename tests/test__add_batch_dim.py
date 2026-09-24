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
from _pytest.mark.structures import Mark, MarkDecorator
from torch._C._functorch import is_legacy_batchedtensor

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Register the underscore-prefixed pytest marker explicitly.
setattr(
    pytest.mark,
    "_add_batch_dim",
    MarkDecorator(Mark("_add_batch_dim", (), {}, _ispytest=True), _ispytest=True),
)

# _add_batch_dim hides a physical dimension in a legacy BatchedTensor view.
# Unwrapping with the same level must recover the original values and layout.
# Autograd rejects legacy BatchedTensor outputs, so there are no backward cases.
_ADD_BATCH_DIM_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [torch.bool, torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
)
# Scalars have no dimension to hide and are covered by a rejection test.
_ADD_BATCH_DIM_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) >= 1]
_ADD_BATCH_DIM_CASES = [
    (shape, batch_dim)
    for shape in _ADD_BATCH_DIM_SHAPES
    for batch_dim in sorted({0, len(shape) // 2, len(shape) - 1})
]


def _assert_batched_view(res_out, ref_out, inp, ref_inp, batch_dim, level):
    # A plain tensor can pass an unwrap round-trip for singleton batch dims.
    # Require the actual legacy batch wrapper as well as its visible metadata.
    assert is_legacy_batchedtensor(res_out)
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()

    # Unwrapping must recover the original physical input exactly.
    res_mat = torch.ops.aten._remove_batch_dim(
        res_out, level, ref_inp.size(batch_dim), batch_dim
    )
    tu.assert_result_equal(res_mat, ref_inp)
    assert torch._C._is_alias_of(res_mat, inp)


@pytest.mark._add_batch_dim
@pytest.mark.parametrize("shape, batch_dim", _ADD_BATCH_DIM_CASES)
@pytest.mark.parametrize("level", [0, 1, 3])
@pytest.mark.parametrize("dtype", _ADD_BATCH_DIM_DTYPES)
def test__add_batch_dim(shape, batch_dim, level, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._add_batch_dim(ref_inp, batch_dim, level)
    res_out = flag_gems._add_batch_dim(inp, batch_dim, level)

    _assert_batched_view(res_out, ref_out, inp, ref_inp, batch_dim, level)


@pytest.mark._add_batch_dim
@pytest.mark.parametrize("shape", _ADD_BATCH_DIM_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _ADD_BATCH_DIM_DTYPES)
def test__add_batch_dim_value_ranges(shape, dtype, value_range):
    batch_dim = len(shape) // 2
    level = 0
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._add_batch_dim(ref_inp, batch_dim, level)
    res_out = flag_gems._add_batch_dim(inp, batch_dim, level)

    _assert_batched_view(res_out, ref_out, inp, ref_inp, batch_dim, level)


@pytest.mark._add_batch_dim
@pytest.mark.parametrize("shape, batch_dim", [((8, 16, 32), 1), ((4, 8, 16, 32), 2)])
@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("dtype", _ADD_BATCH_DIM_DTYPES)
def test__add_batch_dim_non_contiguous(shape, batch_dim, level, dtype):
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten._add_batch_dim(ref_inp, batch_dim, level)
    res_out = flag_gems._add_batch_dim(inp, batch_dim, level)

    _assert_batched_view(res_out, ref_out, inp, ref_inp, batch_dim, level)


@pytest.mark._add_batch_dim
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(_ADD_BATCH_DIM_DTYPES))
)
def test__add_batch_dim_nan_inf(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)
    batch_dim, level = 0, 0

    ref_out = torch.ops.aten._add_batch_dim(ref_inp, batch_dim, level)
    res_out = flag_gems._add_batch_dim(inp, batch_dim, level)

    _assert_batched_view(res_out, ref_out, inp, ref_inp, batch_dim, level)


@pytest.mark._add_batch_dim
def test__add_batch_dim_rejects_0dim_input():
    inp = tu.make_input(torch.float32, (), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._add_batch_dim(inp, 0, 0)
    with pytest.raises(RuntimeError):
        flag_gems._add_batch_dim(inp, 0, 0)


@pytest.mark._add_batch_dim
def test__add_batch_dim_rejects_negative_level():
    inp = tu.make_input(torch.float32, (4, 5), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._add_batch_dim(inp, 1, -1)
    with pytest.raises(RuntimeError):
        flag_gems._add_batch_dim(inp, 1, -1)


@pytest.mark._add_batch_dim
def test__add_batch_dim_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten._add_batch_dim(3.14, 0, 0)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._add_batch_dim(3.14, 0, 0)
