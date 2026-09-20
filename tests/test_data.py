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

from . import accuracy_utils as utils
from . import test_utils as tu

# Return a storage alias detached from autograd.
_DATA_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool, torch.complex32, torch.complex64]
)

_LAYOUT_FNS = [
    ("stride2", lambda t: t[..., ::2]),
    ("offset1", lambda t: t[..., 1:]),
    ("transpose", lambda t: t.transpose(-1, -2)),
]

_LAYOUT_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
_MUTATION_SHAPES = [(16, 32), (4, 8, 16)]
_AUTOGRAD_SHAPES = [(16, 64), (7, 13, 29)]


def _assert_alias_semantics(res_out, ref_out, inp):
    assert res_out.device == inp.device
    assert res_out.data_ptr() == inp.data_ptr()
    assert res_out.stride() == inp.stride()
    assert res_out.storage_offset() == inp.storage_offset()
    assert not res_out.requires_grad
    assert res_out.is_leaf
    assert res_out.grad_fn is None
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.data
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _DATA_DTYPES)
def test_data(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.data(ref_inp)
    res_out = flag_gems.data(inp)

    _assert_alias_semantics(res_out, ref_out, inp)


@pytest.mark.data
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DATA_DTYPES)
def test_data_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.data(ref_inp)
    res_out = flag_gems.data(inp)

    _assert_alias_semantics(res_out, ref_out, inp)


@pytest.mark.data
@pytest.mark.parametrize("layout", _LAYOUT_FNS, ids=[name for name, _ in _LAYOUT_FNS])
@pytest.mark.parametrize("shape", _LAYOUT_SHAPES)
@pytest.mark.parametrize("dtype", _DATA_DTYPES)
def test_data_non_contiguous(layout, shape, dtype):
    _, extract = layout
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = extract(base)
    ref_inp = extract(ref_base)
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten.data(ref_inp)
    res_out = flag_gems.data(inp)

    _assert_alias_semantics(res_out, ref_out, inp)


@pytest.mark.data
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES))
def test_data_special_values(dtype):
    values = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1.5, -2.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = tu.to_reference(values)

    ref_out = torch.ops.aten.data(ref_inp)
    res_out = flag_gems.data(values)

    assert res_out.data_ptr() == values.data_ptr()
    # nan must compare equal to nan (the op must not sanitize it).
    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.data
@pytest.mark.parametrize("shape", _MUTATION_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_data_mutation(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    res_out = flag_gems.data(inp)
    ref_out = torch.ops.aten.data(ref_inp)

    res_out.add_(1.0)
    ref_out.add_(1.0)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.data
@pytest.mark.parametrize("shape", _AUTOGRAD_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_data_autograd_detach(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    ref_inp = tu.to_reference(inp)
    if not ref_inp.requires_grad:
        ref_inp.requires_grad_(True)

    ref_out = torch.ops.aten.data(ref_inp)
    res_out = flag_gems.data(inp)

    _assert_alias_semantics(res_out, ref_out, inp)


@pytest.mark.data
def test_data_rejects_non_tensor():
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.data(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.data(3.14)

    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.data("not-a-tensor")
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.data("not-a-tensor")


@pytest.mark.data
def test_data_rejects_extra_arguments():
    inp = tu.make_input(torch.float32, (4, 4), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    with pytest.raises((TypeError, RuntimeError)):
        torch.ops.aten.data(ref_inp, ref_inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.data(inp, inp)


@pytest.mark.data
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(_DATA_DTYPES))
)
def test_data_special_scenarios(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    reference = tu.to_reference(inp)
    expected = torch.ops.aten.data(reference)
    actual = flag_gems.data(inp)
    tu.assert_result_equal(actual, expected)


@pytest.mark.data
@pytest.mark.parametrize("initial_mutations", [0, 2])
def test_data_independent_version_counter(initial_mutations):
    inp = tu.make_input(torch.float32, (3, 10), ["-1", "1"])[..., 1::2]
    ref_inp = tu.to_reference(inp)
    for _ in range(initial_mutations):
        inp.add_(1)
        ref_inp.add_(1)
    input_version = inp._version
    res_out = flag_gems.data(inp)
    ref_out = torch.ops.aten.data(ref_inp)
    assert res_out._version == ref_out._version

    res_out.add_(1)
    ref_out.add_(1)
    assert inp._version == input_version
    assert res_out._version == ref_out._version
    output_version = res_out._version

    inp.add_(1)
    ref_inp.add_(1)
    assert res_out._version == output_version
    _assert_alias_semantics(res_out, ref_out, inp)
    tu.assert_result_equal(inp, ref_inp)
