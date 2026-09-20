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

# adjoint swaps the last two dimensions and toggles the conjugate bit for
# complex inputs. The result shares storage with the input.
_ADJOINT_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16, torch.complex32])
    + [torch.complex64, torch.bool]
)
# Scalar and 1-D behavior is covered separately below.
_ADJOINT_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) >= 2]


def _transposed_shape(shape):
    # Shape of adjoint(x): the last two dimensions are swapped.
    if len(shape) >= 2:
        return shape[:-2] + (shape[-1], shape[-2])
    return shape


def _assert_view_semantics(res_out, ref_out, inp):
    # adjoint returns an aliasing view (Tensor(a)): shape, strides, storage
    # offset, conjugation state and the shared storage must match aten exactly.
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out._is_view() == ref_out._is_view()
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.data_ptr() == inp.data_ptr()


@pytest.mark.adjoint
@pytest.mark.parametrize("shape", _ADJOINT_SHAPES)
@pytest.mark.parametrize("dtype", _ADJOINT_DTYPES)
def test_adjoint(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.adjoint(ref_inp)
    res_out = flag_gems.adjoint(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark.adjoint
@pytest.mark.parametrize("shape", _ADJOINT_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _ADJOINT_DTYPES)
def test_adjoint_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.adjoint(ref_inp)
    res_out = flag_gems.adjoint(inp)

    _assert_view_semantics(res_out, ref_out, inp)
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.adjoint
@pytest.mark.parametrize("shape", [(8, 16, 32), (4, 8, 16, 32)])
@pytest.mark.parametrize("dtype", _ADJOINT_DTYPES)
def test_adjoint_non_contiguous(shape, dtype):
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten.adjoint(ref_inp)
    res_out = flag_gems.adjoint(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark.adjoint
@pytest.mark.parametrize("shape", [(16, 32), (4, 8, 16)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.COMPLEX_DTYPES)
def test_adjoint_toggle(shape, dtype):
    # An already-conjugated input must produce an unconjugated view.
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)

    inp = torch.ops.aten.adjoint(base)
    ref_inp = torch.ops.aten.adjoint(ref_base)
    assert inp.is_conj() == ref_inp.is_conj()

    ref_out = torch.ops.aten.adjoint(ref_inp)
    res_out = flag_gems.adjoint(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, base)
    assert not res_out.is_conj()


@pytest.mark.adjoint
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES))
def test_adjoint_special_values(dtype):
    values = torch.tensor(
        [
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
            ]
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = tu.to_reference(values)

    ref_out = torch.ops.aten.adjoint(ref_inp)
    res_out = flag_gems.adjoint(values)

    _assert_view_semantics(res_out, ref_out, values)
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.adjoint
@pytest.mark.parametrize("shape", [(16, 32), (4, 8, 16)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_adjoint_mutation(shape, dtype):
    # Writing through the returned view must update the input.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    res_out = flag_gems.adjoint(inp)
    ref_out = torch.ops.aten.adjoint(ref_inp)

    res_out.fill_(2.5)
    ref_out.fill_(2.5)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.adjoint
@pytest.mark.parametrize("shape", [(16, 64), (7, 13, 29)])
@pytest.mark.parametrize(
    "dtype",
    tu.selected_cases(
        [d for d in _ADJOINT_DTYPES if d.is_floating_point or d.is_complex]
    ),
)
def test_adjoint_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, _transposed_shape(shape), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_grad = tu.to_reference(grad)

    ref_out = torch.ops.aten.adjoint(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    res_out = flag_gems.adjoint(inp)
    tu.assert_result_equal(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp)

    assert res_out.requires_grad
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
    tu.assert_result_close(res_in_grad, ref_in_grad)


@pytest.mark.adjoint
@pytest.mark.parametrize("dtype", _ADJOINT_DTYPES)
def test_adjoint_0d(dtype):
    # ATen accepts scalars as a lazy conj(), with a deprecation warning.
    inp = tu.make_input(dtype, (), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.adjoint(ref_inp)
    res_out = flag_gems.adjoint(inp)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.is_conj() == ref_out.is_conj()


@pytest.mark.adjoint
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_adjoint_1d_raises(dtype):
    inp = tu.make_input(dtype, (5,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.adjoint(ref_inp)
    with pytest.raises(RuntimeError):
        flag_gems.adjoint(inp)


@pytest.mark.adjoint
def test_adjoint_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten.adjoint(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.adjoint(3.14)


@pytest.mark.adjoint
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(_ADJOINT_DTYPES))
)
def test_adjoint_special_scenarios(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    inp = inp.reshape(1, -1)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.adjoint(ref_inp)
    res_out = flag_gems.adjoint(inp)

    tu.assert_result_equal(res_out, ref_out)
