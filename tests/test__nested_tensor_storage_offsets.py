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

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Register underscore-prefixed pytest markers explicitly.
setattr(
    pytest.mark,
    "_nested_tensor_storage_offsets",
    MarkDecorator(
        Mark("_nested_tensor_storage_offsets", (), {}, _ispytest=True),
        _ispytest=True,
    ),
)

# Read the CPU int64 metadata of a strided-layout nested tensor.
# Empty batches are valid; jagged layout and .out lack reference kernels.
_NUM_TENSORS = [0, 1, 8, 256]

_NUM_DIMS = [1, 2, 3, 5]

_DEFAULT_VALUE_RANGE = ["-1", "1"]

_COMPONENT_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool]
)


def _make_input(num_tensors, num_dims, dtype, seed=0, value_range=_DEFAULT_VALUE_RANGE):
    # Vary the leading extent per component; keep trailing dimensions fixed.
    gen = torch.Generator("cpu").manual_seed(seed)
    lengths = torch.randint(1, 9, (num_tensors,), generator=gen).tolist()
    components = [
        tu.make_input(dtype, (length,) + (4,) * (num_dims - 1), value_range)
        for length in lengths
    ]
    return torch.nested.nested_tensor(components, device=flag_gems.device)


def _make_strided_view_input(dtype, device=None):
    # Build gaps between components; construct each device separately because this view cannot be copied.
    if device is None:
        device = flag_gems.device
    # tu.make_input builds on the shared test device; the contiguous buffer
    # can always be moved to ``device``. Its values are never read.
    buf = tu.make_input(dtype, (64,), _DEFAULT_VALUE_RANGE).to(device)
    nested_size = torch.tensor([[2, 3], [4, 3], [1, 3], [3, 3]], dtype=torch.int64)
    nested_strides = torch.tensor([[6, 1], [6, 1], [6, 1], [6, 1]], dtype=torch.int64)
    storage_offsets = torch.tensor([0, 12, 30, 36], dtype=torch.int64)
    return torch.ops.aten._nested_view_from_buffer(
        buf, nested_size, nested_strides, storage_offsets
    )


def _assert_offsets(res_out, ref_out):
    assert res_out.device == torch.device("cpu")
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark._nested_tensor_storage_offsets
@pytest.mark.parametrize("num_tensors", _NUM_TENSORS)
@pytest.mark.parametrize("num_dims", _NUM_DIMS)
@pytest.mark.parametrize("dtype", _COMPONENT_DTYPES)
def test__nested_tensor_storage_offsets(num_tensors, num_dims, dtype):
    inp = _make_input(num_tensors, num_dims, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nested_tensor_storage_offsets(ref_inp)
    res_out = flag_gems._nested_tensor_storage_offsets(inp)

    _assert_offsets(res_out, ref_out)


@pytest.mark._nested_tensor_storage_offsets
@pytest.mark.parametrize("dtype", _COMPONENT_DTYPES)
def test__nested_tensor_storage_offsets_ragged_non_zero_dim(dtype):
    components = [
        tu.make_input(dtype, (2, length), _DEFAULT_VALUE_RANGE)
        for length in (3, 5, 1, 4)
    ]
    inp = torch.nested.nested_tensor(components, device=flag_gems.device)
    assert inp.is_nested
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nested_tensor_storage_offsets(ref_inp)
    res_out = flag_gems._nested_tensor_storage_offsets(inp)

    _assert_offsets(res_out, ref_out)


@pytest.mark._nested_tensor_storage_offsets
@pytest.mark.parametrize("dtype", _COMPONENT_DTYPES)
def test__nested_tensor_storage_offsets_uniform(dtype):
    num_tensors = 6
    components = [
        tu.make_input(dtype, (4, 4, 4), _DEFAULT_VALUE_RANGE)
        for _ in range(num_tensors)
    ]
    inp = torch.nested.nested_tensor(components, device=flag_gems.device)
    assert inp.is_nested
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nested_tensor_storage_offsets(ref_inp)
    res_out = flag_gems._nested_tensor_storage_offsets(inp)

    _assert_offsets(res_out, ref_out)


@pytest.mark._nested_tensor_storage_offsets
@pytest.mark.parametrize("dtype", _COMPONENT_DTYPES)
def test__nested_tensor_storage_offsets_with_empty_components(dtype):
    num_tensors = 4
    gen = torch.Generator("cpu").manual_seed(1)
    components = []
    for _ in range(num_tensors):
        length = int(torch.randint(0, 4, (1,), generator=gen).item())
        components.append(tu.make_input(dtype, (length, 4), _DEFAULT_VALUE_RANGE))
    inp = torch.nested.nested_tensor(components, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nested_tensor_storage_offsets(ref_inp)
    res_out = flag_gems._nested_tensor_storage_offsets(inp)

    _assert_offsets(res_out, ref_out)


@pytest.mark._nested_tensor_storage_offsets
@pytest.mark.parametrize("dtype", _COMPONENT_DTYPES)
def test__nested_tensor_storage_offsets_non_contiguous(dtype):
    inp = _make_strided_view_input(dtype)
    ref_device = torch.device("cpu") if utils.TO_CPU else flag_gems.device
    ref_inp = _make_strided_view_input(dtype, ref_device)
    assert inp.is_nested

    ref_out = torch.ops.aten._nested_tensor_storage_offsets(ref_inp)
    res_out = flag_gems._nested_tensor_storage_offsets(inp)

    _assert_offsets(res_out, ref_out)


@pytest.mark._nested_tensor_storage_offsets
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _COMPONENT_DTYPES)
def test__nested_tensor_storage_offsets_value_ranges(dtype, value_range):
    num_tensors, num_dims = 8, 3
    inp = _make_input(num_tensors, num_dims, dtype, value_range=value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nested_tensor_storage_offsets(ref_inp)
    res_out = flag_gems._nested_tensor_storage_offsets(inp)

    _assert_offsets(res_out, ref_out)


@pytest.mark._nested_tensor_storage_offsets
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_COMPONENT_DTYPES))
)
def test__nested_tensor_storage_offsets_nan_inf(dtype, scenario):
    num_tensors = 4
    gen = torch.Generator("cpu").manual_seed(0)
    lengths = torch.randint(1, 9, (num_tensors,), generator=gen).tolist()
    special = tu.make_special_input(dtype, scenario)
    components = []
    for length in lengths:
        numel = length * 4
        repeats = (numel + special.numel() - 1) // special.numel()
        values = special.repeat(repeats)[:numel].reshape(length, 4)
        components.append(values)
    inp = torch.nested.nested_tensor(components, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nested_tensor_storage_offsets(ref_inp)
    res_out = flag_gems._nested_tensor_storage_offsets(inp)

    _assert_offsets(res_out, ref_out)


@pytest.mark._nested_tensor_storage_offsets
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__nested_tensor_storage_offsets_non_nested_raises(dtype):
    inp = tu.make_input(dtype, (4, 4), _DEFAULT_VALUE_RANGE)
    ref_inp = tu.to_reference(inp)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError, ValueError)):
        torch.ops.aten._nested_tensor_storage_offsets(ref_inp)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError, ValueError)):
        flag_gems._nested_tensor_storage_offsets(inp)
