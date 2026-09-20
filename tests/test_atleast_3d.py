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

# Scalars, vectors and matrices become (1, 1, 1), (1, N, 1) and (M, N, 1)
# views respectively. Tensors with ndim >= 3 are returned unchanged.
_FP8_DTYPES = [
    getattr(torch, name)
    for name in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz")
    if getattr(torch, name, None) is not None
]
ATLEAST_3D_DTYPES = (
    [torch.int8, torch.uint8]
    + _FP8_DTYPES
    + utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [torch.bool]
)


@pytest.mark.atleast_3d
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", ATLEAST_3D_DTYPES)
def test_atleast_3d(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.atleast_3d(ref_inp)
    res_out = flag_gems.atleast_3d(inp)

    tu.assert_result_equal(res_out, ref_out)
    # A view/identity op must alias its input (Tensor(a)).
    assert res_out.data_ptr() == inp.data_ptr()


@pytest.mark.atleast_3d
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", ATLEAST_3D_DTYPES)
def test_atleast_3d_value_ranges(shape, dtype, value_range):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.atleast_3d(ref_inp)
    res_out = flag_gems.atleast_3d(inp)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.data_ptr() == inp.data_ptr()


@pytest.mark.atleast_3d_sequence
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", ATLEAST_3D_DTYPES)
def test_atleast_3d_sequence(shape, dtype):
    # The Tensor[] overload must apply the same view per element: scalar ->
    # (1,1,1), 1-dim -> (1,N,1), 2-dim -> (M,N,1) and >= 3-dim identity.
    inp = [
        tu.make_input(dtype, (), ["-1", "1"]),
        tu.make_input(dtype, (3,), ["-1", "1"]),
        tu.make_input(dtype, (4, 5), ["-1", "1"]),
        tu.make_input(dtype, shape, ["-1", "1"]),
    ]
    ref_inp = [tu.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.atleast_3d.Sequence(ref_inp)
    res_out = flag_gems.atleast_3d(inp)

    assert len(res_out) == len(ref_out) == 4
    for res, ref, src in zip(res_out, ref_out, inp):
        tu.assert_result_equal(res, ref)
        # Each result is a view of its own input.
        assert res.data_ptr() == src.data_ptr()


@pytest.mark.atleast_3d_sequence
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", ATLEAST_3D_DTYPES)
def test_atleast_3d_sequence_value_ranges(dtype, value_range):
    # Range sweep for the Tensor[] overload over the three shape-changing
    # paths (0-dim / 1-dim / 2-dim).
    inp = [
        tu.make_input(dtype, (), value_range),
        tu.make_input(dtype, (3,), value_range),
        tu.make_input(dtype, (4, 5), value_range),
    ]
    ref_inp = [tu.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.atleast_3d.Sequence(ref_inp)
    res_out = flag_gems.atleast_3d(inp)

    assert len(res_out) == len(ref_out) == 3
    for res, ref, src in zip(res_out, ref_out, inp):
        tu.assert_result_equal(res, ref)
        assert res.data_ptr() == src.data_ptr()


@pytest.mark.atleast_3d_sequence
def test_atleast_3d_sequence_empty():
    # An empty Tensor[] is legitimate: the reference returns an empty list and
    # the candidate must do the same (atleast_3d.Sequence([]) does not raise).
    ref_out = torch.ops.aten.atleast_3d.Sequence([])
    res_out = flag_gems.atleast_3d([])
    assert len(res_out) == len(ref_out)


@pytest.mark.atleast_3d
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES))
def test_atleast_3d_nan_inf(dtype):
    # Values pass through a view untouched: nan/inf/-inf and signed zeros must
    # be preserved (the float comparison path uses equal_nan=True). 1e30
    # overflows to inf in fp16 and remains finite in bf16.
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

    ref_out = torch.ops.aten.atleast_3d(ref_inp)
    res_out = flag_gems.atleast_3d(inp)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.data_ptr() == inp.data_ptr()


@pytest.mark.atleast_3d
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_atleast_3d_complex(dtype):
    inp = tu.make_input(dtype, (2, 5), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.atleast_3d(ref_inp)
    res_out = flag_gems.atleast_3d(inp)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.data_ptr() == inp.data_ptr()


@pytest.mark.atleast_3d_backward
@pytest.mark.parametrize("shape", [(), (3,), (4, 5), (16, 64), (7, 13, 29)])
@pytest.mark.parametrize(
    "dtype",
    tu.selected_cases(
        [d for d in ATLEAST_3D_DTYPES if d.is_floating_point or d.is_complex]
    ),
)
def test_atleast_3d_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.atleast_3d(ref_inp)
    # Explicit upstream values detect gradients that always return ones.
    grad = tu.make_input(dtype, ref_out.shape, ["-1", "1"])
    ref_grad = tu.to_reference(grad)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    res_out = flag_gems.atleast_3d(inp)
    tu.assert_result_equal(res_out, ref_out)

    assert res_out.requires_grad
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
    tu.assert_result_equal(res_in_grad, ref_in_grad)


@pytest.mark.atleast_3d_negative
def test_atleast_3d_rejects_non_tensor():
    # The aten op requires a Tensor (the Tensor overload) / a TensorList whose
    # elements are Tensors (the Sequence overload); a Python float must raise on
    # both the reference and the candidate path rather than being accepted.
    with pytest.raises(RuntimeError):
        torch.ops.aten.atleast_3d(3.14)
    with pytest.raises(RuntimeError):
        torch.ops.aten.atleast_3d.Sequence(
            [torch.zeros(2, device=flag_gems.device), 3.14]
        )

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.atleast_3d(3.14)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.atleast_3d([torch.zeros(2, device=flag_gems.device), 3.14])


@pytest.mark.atleast_3d
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(ATLEAST_3D_DTYPES))
)
def test_atleast_3d_special_scenarios(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.atleast_3d(ref_inp)
    res_out = flag_gems.atleast_3d(inp)

    tu.assert_result_equal(res_out, ref_out)
