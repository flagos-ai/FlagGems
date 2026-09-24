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

import math

import pytest
import torch

import flag_gems

from . import test_utils as tu

# Place flattened input values on a diagonal and zero the rest.
_DIAGFLAT_DTYPES = tu.REQUIRED_DTYPES + [torch.int16, torch.float64, torch.bool]

_GRAD_DTYPES = [d for d in _DIAGFLAT_DTYPES if d.is_floating_point]

_DIAGFLAT_OFFSETS = [-2, -1, 0, 1, 2]

# Output size is quadratic in input numel; use small representatives for ranks 2-5.
_DIAGFLAT_SHAPES = [
    (),
    (1,),
    (256,),
    (2, 3),
    (4, 5, 6),
    (2, 3, 4, 5),
    (2, 2, 2, 2, 3),
    (0,),
]

_DIAGFLAT_RANGE_SHAPES = tu.selected_cases(
    _DIAGFLAT_SHAPES[:-1], quick=[(2, 19, 7), (2, 3), (4, 5, 6)]
)

_DIAGFLAT_NONCONTIG_SHAPES = [(4, 8), (6, 3), (2, 3, 4)]

_DIAGFLAT_STRIDED_SHAPES = [(16, 32), (4, 8, 16)]

_DIAGFLAT_BACKWARD_SHAPES = [(8,), (2, 3), (4, 5, 6)]


def _assert_output(res_out, ref_out):
    assert res_out.is_contiguous()
    assert not res_out._is_view()
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize(
    "shape", tu.selected_cases(_DIAGFLAT_SHAPES, quick=[(2, 19, 7)])
)
@pytest.mark.parametrize("offset", _DIAGFLAT_OFFSETS)
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat(shape, offset, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = flag_gems.diagflat(inp, offset)

    _assert_output(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", _DIAGFLAT_RANGE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.diagflat(ref_inp, 0)
    res_out = flag_gems.diagflat(inp, 0)

    _assert_output(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", [(2,), (16,)])
@pytest.mark.parametrize("offset", [-7, -3, 3, 7])
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat_large_offset(shape, offset, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = flag_gems.diagflat(inp, offset)

    _assert_output(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", _DIAGFLAT_NONCONTIG_SHAPES)
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat_non_contiguous(shape, offset, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    inp = inp.transpose(-1, -2)
    ref_inp = ref_inp.transpose(-1, -2)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = flag_gems.diagflat(inp, offset)

    _assert_output(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", _DIAGFLAT_STRIDED_SHAPES)
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat_strided(shape, offset, dtype):
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = flag_gems.diagflat(inp, offset)

    _assert_output(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(_DIAGFLAT_DTYPES))
)
def test_diagflat_nan_inf(dtype, scenario):
    values = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(values)

    ref_out = torch.ops.aten.diagflat(ref_inp, 1)
    res_out = flag_gems.diagflat(values, 1)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("offset", [-4, -1, 0, 1, 4])
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat_empty_input(offset, dtype):
    inp = tu.make_input(dtype, (0,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = flag_gems.diagflat(inp, offset)

    _assert_output(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", _DIAGFLAT_BACKWARD_SHAPES)
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("dtype", tu.selected_cases(_GRAD_DTYPES))
def test_diagflat_backward(shape, offset, dtype):
    # Backward gathers the offset diagonal and reshapes it to the input.
    n = math.prod(shape)
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, (n + abs(offset), n + abs(offset)), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_grad = tu.to_reference(grad)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    res_out = flag_gems.diagflat(inp, offset)
    tu.assert_result_equal(res_out, ref_out)

    assert res_out.requires_grad
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
    tu.assert_result_equal(res_in_grad, ref_in_grad)


@pytest.mark.diagflat
def test_diagflat_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten.diagflat(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.diagflat(3.14)


@pytest.mark.diagflat
def test_diagflat_rejects_non_int_offset():
    inp = tu.make_input(torch.float32, (4,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.diagflat(ref_inp, 1.5)
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.diagflat(inp, 1.5)
