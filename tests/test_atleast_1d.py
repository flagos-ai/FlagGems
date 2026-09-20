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

# Scalars become (1,) views; tensors with ndim >= 1 are returned unchanged.
_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
_SUPPORTED_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + [torch.int8, torch.uint8]
    + utils.ALL_INT_DTYPES
    + _FP8_DTYPES
    + [torch.bool]
)
_NAN_INF_DTYPES = utils.ALL_FLOAT_DTYPES + _FP8_DTYPES
# Keep the scalar boundary in quick mode as well.
_ATLEAST_1D_SHAPES = [()] + [shape for shape in tu.selected_shapes() if shape]


@pytest.mark.atleast_1d
@pytest.mark.parametrize("shape", _ATLEAST_1D_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_atleast_1d_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.atleast_1d(ref_inp)
    res_out = flag_gems.atleast_1d(inp)

    # atleast_1d is a view op: the result must alias the input storage.
    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.atleast_1d
@pytest.mark.parametrize("dtype", tu.selected_cases(_NAN_INF_DTYPES))
def test_atleast_1d_nan_inf(dtype):
    # atleast_1d is a view: nan/inf/-inf/+-0.0 must pass through bit-for-bit.
    inp = torch.tensor(
        [
            float("inf"),
            float("-inf"),
            float("nan"),
            0.0,
            -0.0,
            1.5,
            -2.5,
            1e30,
            -1e30,
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.atleast_1d(ref_inp)
    res_out = flag_gems.atleast_1d(inp)

    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.atleast_1d_sequence
@pytest.mark.parametrize("shape", _ATLEAST_1D_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_atleast_1d_sequence(shape, value_range, dtype):
    # Mix a 0-dim scalar with the current shape so the sequence overload
    # exercises both the scalar -> (1,) view path and the identity path.
    inp = [
        tu.make_input(dtype, (), value_range),
        tu.make_input(dtype, shape, value_range),
        tu.make_input(dtype, shape, value_range),
    ]
    ref_inp = [tu.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.atleast_1d.Sequence(ref_inp)
    res_out = flag_gems.atleast_1d(inp)

    assert len(res_out) == len(ref_out)
    for res, ref, src in zip(res_out, ref_out, inp):
        # atleast_1d is a view op: each result must alias its input.
        assert res.data_ptr() == src.data_ptr()
        tu.assert_result_equal(res, ref)


@pytest.mark.atleast_1d_sequence
def test_atleast_1d_sequence_empty():
    # A Tensor[] input may legitimately be empty: the reference returns an
    # empty list and the candidate must return an empty list too.
    ref_out = torch.ops.aten.atleast_1d.Sequence([])
    res_out = flag_gems.atleast_1d([])
    assert len(res_out) == len(ref_out)


@pytest.mark.atleast_1d_backward
@pytest.mark.parametrize("shape", [(), (3,), (16, 64), (7, 13, 29)])
@pytest.mark.parametrize(
    "dtype",
    tu.selected_cases(
        [d for d in _SUPPORTED_DTYPES if d.is_floating_point or d.is_complex]
    ),
)
def test_atleast_1d_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.atleast_1d(ref_inp)
    # Explicit upstream values detect gradients that always return ones.
    grad = tu.make_input(dtype, ref_out.shape, ["-1", "1"])
    ref_grad = tu.to_reference(grad)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    res_out = flag_gems.atleast_1d(inp)
    tu.assert_result_equal(res_out, ref_out)

    assert res_out.requires_grad
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
    tu.assert_result_equal(res_in_grad, ref_in_grad)


@pytest.mark.atleast_1d_negative
def test_atleast_1d_rejects_non_tensor():
    # The aten op only accepts a Tensor (a list of Tensors goes through the
    # .Sequence overload); Python scalars hit a schema mismatch and raise.
    with pytest.raises(RuntimeError):
        torch.ops.aten.atleast_1d(3.14)
    with pytest.raises(RuntimeError):
        torch.ops.aten.atleast_1d.Sequence(
            [torch.zeros(2, device=flag_gems.device), 3.14]
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.atleast_1d(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.atleast_1d([torch.zeros(2, device=flag_gems.device), 3.14])


@pytest.mark.atleast_1d
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(_SUPPORTED_DTYPES))
)
def test_atleast_1d_special_scenarios(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.atleast_1d(ref_inp)
    res_out = flag_gems.atleast_1d(inp)

    tu.assert_result_equal(res_out, ref_out)
