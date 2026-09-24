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
    "_fw_primal",
    MarkDecorator(Mark("_fw_primal", (), {}, _ispytest=True), _ispytest=True),
)

# Return the primal view, preserving storage and layout.
_FW_PRIMAL_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + [torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
    + utils.ALL_INT_DTYPES
    + utils.BOOL_TYPES
    + utils.COMPLEX_DTYPES
)

_FW_PRIMAL_LEVELS = [0, 1, 3]
_FW_PRIMAL_LEVEL_SHAPES = [(), (256,), (7, 13, 29)]
_FW_PRIMAL_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
_FW_PRIMAL_MUTATION_SHAPES = [(16, 32), (4, 8, 16)]
_FW_PRIMAL_EMPTY_SHAPES = [(0,), (2, 0, 3)]
_FW_PRIMAL_BACKWARD_SHAPES = [(), (256,), (7, 13, 29)]
_FW_PRIMAL_BACKWARD_DTYPES = [
    d for d in _FW_PRIMAL_DTYPES if d.is_floating_point or d.is_complex
]

_FW_PRIMAL_SPECIAL_VALUES = [
    0.0,
    -0.0,
    float("inf"),
    float("-inf"),
    1.5,
    -1.5,
    float("nan"),
]


def _assert_view_semantics(res_out, ref_out, inp):
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out._is_view() == ref_out._is_view()
    assert res_out.data_ptr() == inp.data_ptr()


@pytest.mark._fw_primal
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _FW_PRIMAL_DTYPES)
def test__fw_primal(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._fw_primal(ref_inp, 0)
    res_out = flag_gems._fw_primal(inp, 0)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark._fw_primal
@pytest.mark.parametrize("shape", _FW_PRIMAL_LEVEL_SHAPES)
@pytest.mark.parametrize("level", _FW_PRIMAL_LEVELS)
@pytest.mark.parametrize("dtype", _FW_PRIMAL_DTYPES)
def test__fw_primal_level(shape, level, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._fw_primal(ref_inp, level)
    res_out = flag_gems._fw_primal(inp, level)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark._fw_primal
@pytest.mark.parametrize("shape", _FW_PRIMAL_NONCONTIG_SHAPES)
@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("dtype", _FW_PRIMAL_DTYPES)
def test__fw_primal_non_contiguous(shape, level, dtype):
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten._fw_primal(ref_inp, level)
    res_out = flag_gems._fw_primal(inp, level)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark._fw_primal
@pytest.mark.parametrize("shape", _FW_PRIMAL_MUTATION_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)
def test__fw_primal_mutation(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._fw_primal(ref_inp, 0)
    res_out = flag_gems._fw_primal(inp, 0)

    if dtype == torch.bool:
        res_out.fill_(True)
        ref_out.fill_(True)
    elif dtype.is_floating_point:
        res_out.fill_(2.5)
        ref_out.fill_(2.5)
    else:
        res_out.fill_(7)
        ref_out.fill_(7)

    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark._fw_primal
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES))
def test__fw_primal_special_values(dtype):
    values = torch.tensor(
        _FW_PRIMAL_SPECIAL_VALUES, dtype=dtype, device=flag_gems.device
    )
    ref_inp = tu.to_reference(values)

    ref_out = torch.ops.aten._fw_primal(ref_inp, 0)
    res_out = flag_gems._fw_primal(values, 0)

    _assert_view_semantics(res_out, ref_out, values)
    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
    assert torch.signbit(res_out[0]).item() == torch.signbit(values[0]).item()
    assert torch.signbit(res_out[1]).item() == torch.signbit(values[1]).item()


@pytest.mark._fw_primal
@pytest.mark.parametrize("shape", _FW_PRIMAL_EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", _FW_PRIMAL_DTYPES)
def test__fw_primal_empty(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._fw_primal(ref_inp, 0)
    res_out = flag_gems._fw_primal(inp, 0)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark._fw_primal
@pytest.mark.parametrize("shape", _FW_PRIMAL_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", tu.selected_cases(_FW_PRIMAL_BACKWARD_DTYPES))
def test__fw_primal_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_grad = tu.to_reference(grad)

    ref_out = torch.ops.aten._fw_primal(ref_inp, 0)
    res_out = flag_gems._fw_primal(inp, 0)
    tu.assert_result_equal(res_out, ref_out)

    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
    tu.assert_result_equal(res_in_grad, ref_in_grad)


@pytest.mark._fw_primal
def test__fw_primal_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten._fw_primal(3.14, 0)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._fw_primal(3.14, 0)


@pytest.mark._fw_primal
def test__fw_primal_rejects_non_int_level():
    inp = tu.make_input(torch.float32, (8,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten._fw_primal(ref_inp, 1.5)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._fw_primal(inp, 1.5)


@pytest.mark._fw_primal
def test__fw_primal_rejects_missing_level():
    inp = tu.make_input(torch.float32, (8,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten._fw_primal(ref_inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._fw_primal(inp)


@pytest.mark._fw_primal
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(_FW_PRIMAL_DTYPES))
)
def test__fw_primal_special_scenarios(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    reference = tu.to_reference(inp)
    expected = torch.ops.aten._fw_primal(reference, 0)
    actual = flag_gems._fw_primal(inp, 0)
    tu.assert_result_equal(actual, expected)


@pytest.mark._fw_primal
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("shape", [(), (7,), (3, 5)])
def test__fw_primal_dual(shape, dtype):
    primal = tu.make_input(dtype, shape, ["-1", "1"])
    tangent = torch.ones_like(primal)
    ref_primal = tu.to_reference(primal)
    ref_tangent = tu.to_reference(tangent)

    with torch.autograd.forward_ad.dual_level() as level:
        inp = torch.autograd.forward_ad.make_dual(primal, tangent)
        ref_inp = torch.autograd.forward_ad.make_dual(ref_primal, ref_tangent)
        ref_out = torch.ops.aten._fw_primal(ref_inp, level)
        res_out = flag_gems._fw_primal(inp, level)

        tu.assert_result_equal(res_out, ref_out)
        _assert_view_semantics(res_out, ref_out, inp)
        assert torch.autograd.forward_ad.unpack_dual(res_out).tangent is None
