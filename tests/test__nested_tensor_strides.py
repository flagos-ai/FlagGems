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
    "_nested_tensor_strides",
    MarkDecorator(
        Mark("_nested_tensor_strides", (), {}, _ispytest=True), _ispytest=True
    ),
)

# Read the CPU int64 metadata of a strided-layout nested tensor.
# Empty batches and .out are excluded because the current reference cannot run them.
_NUM_TENSORS = [1, 8, 64]

_VALUE_RANGE = ["-1", "1"]

_TRAILING_EXTENT = 4

_MAX_TRAILING_EXTENT = 4

_COMPONENT_DTYPES = (
    [torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
    + utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [torch.bool]
)

_NEGATIVE_EXC = (
    NotImplementedError,
    RuntimeError,
    TypeError,
    ValueError,
    AttributeError,
    IndexError,
)

_COMPONENT_RANKS = sorted(
    {len(shape) for shape in tu.selected_shapes() if len(shape) >= 1}
)

# (shape, trailing dimensions); cap trailing extents while retaining component rank.
_SHAPE_LEVEL_CASES = [
    (shape, tuple(min(int(dim), _MAX_TRAILING_EXTENT) for dim in shape[1:]))
    for shape in tu.selected_shapes()
    if len(shape) >= 1
]

_VALUE_RANGE_LAYOUTS = tu.selected_cases([(8, 2), (32, 3), (64, 4)], quick=[(8, 2)])


def _make_nested(
    num_tensors,
    num_dims,
    dtype,
    value_range=_VALUE_RANGE,
    seed=0,
    trailing=_TRAILING_EXTENT,
    device=None,
):
    # Vary the leading extent per component; keep trailing dimensions fixed.
    if device is None:
        device = flag_gems.device
    gen = torch.Generator("cpu").manual_seed(seed)
    lengths = torch.randint(1, 9, (num_tensors,), generator=gen).tolist()
    components = [
        tu.make_input(dtype, (length,) + (trailing,) * (num_dims - 1), value_range)
        for length in lengths
    ]
    inp = torch.nested.nested_tensor(components, device=device)
    return inp


def _assert_strides(res_out, ref_out):
    assert res_out.device == torch.device("cpu")
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark._nested_tensor_strides
@pytest.mark.parametrize("num_tensors", _NUM_TENSORS)
@pytest.mark.parametrize("num_dims", _COMPONENT_RANKS)
@pytest.mark.parametrize("dtype", _COMPONENT_DTYPES)
def test__nested_tensor_strides(num_tensors, num_dims, dtype):
    inp = _make_nested(num_tensors, num_dims, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nested_tensor_strides(ref_inp)
    res_out = flag_gems._nested_tensor_strides(inp)

    _assert_strides(res_out, ref_out)


@pytest.mark._nested_tensor_strides
@pytest.mark.parametrize("case", _SHAPE_LEVEL_CASES)
@pytest.mark.parametrize("dtype", _COMPONENT_DTYPES)
def test__nested_tensor_strides_shape_levels(case, dtype):
    _, trailing = case
    num_tensors = 3
    gen = torch.Generator("cpu").manual_seed(0)
    lengths = torch.randint(1, 5, (num_tensors,), generator=gen).tolist()
    components = [
        tu.make_input(dtype, (length,) + trailing, _VALUE_RANGE) for length in lengths
    ]
    inp = torch.nested.nested_tensor(components, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nested_tensor_strides(ref_inp)
    res_out = flag_gems._nested_tensor_strides(inp)

    _assert_strides(res_out, ref_out)


@pytest.mark._nested_tensor_strides
@pytest.mark.parametrize("case", _VALUE_RANGE_LAYOUTS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _COMPONENT_DTYPES)
def test__nested_tensor_strides_value_ranges(case, value_range, dtype):
    num_tensors, num_dims = case
    inp = _make_nested(num_tensors, num_dims, dtype, value_range=value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nested_tensor_strides(ref_inp)
    res_out = flag_gems._nested_tensor_strides(inp)

    _assert_strides(res_out, ref_out)


@pytest.mark._nested_tensor_strides
@pytest.mark.parametrize("dtype", _COMPONENT_DTYPES)
def test__nested_tensor_strides_uniform(dtype):
    num_tensors = 6
    components = [
        tu.make_input(dtype, (4, 4, 4), _VALUE_RANGE) for _ in range(num_tensors)
    ]
    inp = torch.nested.nested_tensor(components, device=flag_gems.device)
    assert inp.is_nested
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nested_tensor_strides(ref_inp)
    res_out = flag_gems._nested_tensor_strides(inp)

    _assert_strides(res_out, ref_out)


@pytest.mark._nested_tensor_strides
@pytest.mark.parametrize("dtype", _COMPONENT_DTYPES)
def test__nested_tensor_strides_with_empty_components(dtype):
    num_tensors = 4
    gen = torch.Generator("cpu").manual_seed(1)
    lengths = [
        int(torch.randint(0, 4, (1,), generator=gen).item()) for _ in range(num_tensors)
    ]
    components = [tu.make_input(dtype, (length, 4), _VALUE_RANGE) for length in lengths]
    inp = torch.nested.nested_tensor(components, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nested_tensor_strides(ref_inp)
    res_out = flag_gems._nested_tensor_strides(inp)

    # Every component is contiguous: the innermost stride is 1 regardless of the
    # dim-0 extent (including extent 0), and the outer stride is the trailing
    # extent 4.
    _assert_strides(res_out, ref_out)


@pytest.mark._nested_tensor_strides
@pytest.mark.parametrize("dtype", _COMPONENT_DTYPES)
def test__nested_tensor_strides_transposed(dtype):
    num_tensors, num_dims = 5, 2
    inp = _make_nested(num_tensors, num_dims, dtype, seed=2)
    inp = inp.transpose(1, 2)
    # A non-contiguous nested tensor cannot be moved with Tensor.to()
    # ("NestedTensor must be contiguous to get buffer"), so build the reference
    # view directly on CPU when the TO_CPU reference is in use.
    ref_device = torch.device("cpu") if utils.TO_CPU else flag_gems.device
    ref_inp = _make_nested(num_tensors, num_dims, dtype, seed=2, device=ref_device)
    ref_inp = ref_inp.transpose(1, 2)
    assert inp.is_nested

    ref_out = torch.ops.aten._nested_tensor_strides(ref_inp)
    res_out = flag_gems._nested_tensor_strides(inp)

    # The transposed view is non-contiguous: the innermost stride is the leading
    # component extent (4) rather than 1.
    _assert_strides(res_out, ref_out)


@pytest.mark._nested_tensor_strides
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_COMPONENT_DTYPES))
)
def test__nested_tensor_strides_nan_inf_values(dtype, scenario):
    num_tensors = 4
    gen = torch.Generator("cpu").manual_seed(3)
    lengths = torch.randint(1, 5, (num_tensors,), generator=gen).tolist()
    special = tu.make_special_input(dtype, scenario)
    components = []
    for length in lengths:
        numel = length * 4
        repeats = (numel + special.numel() - 1) // special.numel()
        values = special.repeat(repeats)[:numel].reshape(length, 4)
        components.append(values)
    inp = torch.nested.nested_tensor(components, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nested_tensor_strides(ref_inp)
    res_out = flag_gems._nested_tensor_strides(inp)

    _assert_strides(res_out, ref_out)


@pytest.mark._nested_tensor_strides
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__nested_tensor_strides_dense_raises(dtype):
    inp = tu.make_input(dtype, (4, 4), _VALUE_RANGE)
    with pytest.raises(NotImplementedError):
        torch.ops.aten._nested_tensor_strides(tu.to_reference(inp))
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._nested_tensor_strides(inp)


@pytest.mark._nested_tensor_strides
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__nested_tensor_strides_jagged_raises(dtype):
    ref_device = torch.device("cpu") if utils.TO_CPU else flag_gems.device
    offsets = torch.tensor([0, 2, 5, 9], dtype=torch.int64)
    values = tu.make_input(dtype, (9, 2), _VALUE_RANGE)
    inp = torch.nested.nested_tensor_from_jagged(values, offsets.to(flag_gems.device))
    ref_inp = torch.nested.nested_tensor_from_jagged(
        values.to(ref_device), offsets.to(ref_device)
    )
    assert inp.layout == torch.jagged
    with pytest.raises(NotImplementedError):
        torch.ops.aten._nested_tensor_strides(ref_inp)
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._nested_tensor_strides(inp)


@pytest.mark._nested_tensor_strides
def test__nested_tensor_strides_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten._nested_tensor_strides(3.14)
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._nested_tensor_strides(3.14)
