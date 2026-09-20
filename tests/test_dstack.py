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

# Promote inputs to at least 3-D, then concatenate along dimension 2.
DSTACK_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool]
    + utils.COMPLEX_DTYPES
)

_MAIN_RANGE = ["-1", "1"]

_DSTACK_EXTRA_SHAPE_SETS = tu.selected_cases(
    [
        [(3,), (3,)],
        [(3, 33), (3, 33)],
        [(16, 16, 333), (16, 16, 333), (16, 16, 333)],
        [(8, 8, 16, 16), (8, 8, 16, 16)],
        [(13, 3, 64, 5, 2), (13, 3, 96, 5, 2), (13, 3, 32, 5, 2)],
    ],
    quick=[[(3,), (3,)], [(8, 16, 32), (8, 16, 48)]],
)

_DSTACK_SHAPE_SETS = _DSTACK_EXTRA_SHAPE_SETS + [
    [shape, shape] for shape in tu.selected_shapes()
]

_DSTACK_RANGE_SHAPE_SETS = tu.selected_cases(
    [
        [(), ()],
        [(3,), (3,)],
        [(4, 5), (4, 5)],
        [(4, 5, 6), (4, 5, 6)],
        [(4, 5, 6), (4, 5, 7)],
    ],
    quick=[[(), ()], [(3,), (3,)], [(4, 5), (4, 5)], [(4, 5, 6), (4, 5, 7)]],
)

_DSTACK_OUT_SHAPE_SETS = tu.selected_cases(
    [
        [(3,), (3,)],
        [(4, 5), (4, 5)],
        [(8, 16, 32), (8, 16, 48)],
        [(8, 8, 16, 16), (8, 8, 16, 16)],
    ],
    quick=[[(3,), (3,)], [(8, 16, 32), (8, 16, 48)]],
)

_DSTACK_EMPTY_SHAPE_SETS = [
    [(0,), (0,)],
    [(2, 0), (2, 0)],
    [(0, 3, 4), (0, 3, 4)],
]

_DSTACK_BACKWARD_SHAPE_SETS = [
    [(3,), (3,)],
    [(4, 5), (4, 5)],
    [(4, 5, 6), (4, 5, 7)],
]

_DTYPE_RANGE_PAIRS = [
    (dtype, value_range)
    for dtype in DSTACK_DTYPES
    for value_range in tu.selected_ranges()
]


def _assert_dstack_output(res_out, ref_out):
    assert res_out.is_contiguous()
    assert not res_out._is_view()
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.dstack
@pytest.mark.parametrize("shape_set", _DSTACK_SHAPE_SETS)
@pytest.mark.parametrize("dtype", DSTACK_DTYPES)
def test_dstack(shape_set, dtype):
    inp = [tu.make_input(dtype, s, _MAIN_RANGE) for s in shape_set]
    ref_inp = [tu.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    res_out = flag_gems.dstack(inp)

    _assert_dstack_output(res_out, ref_out)


@pytest.mark.dstack
@pytest.mark.parametrize("shape_set", _DSTACK_RANGE_SHAPE_SETS)
@pytest.mark.parametrize("dtype, value_range", _DTYPE_RANGE_PAIRS)
def test_dstack_value_ranges(shape_set, dtype, value_range):
    inp = [tu.make_input(dtype, s, value_range) for s in shape_set]
    ref_inp = [tu.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    res_out = flag_gems.dstack(inp)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.dstack_out
@pytest.mark.parametrize("shape_set", _DSTACK_OUT_SHAPE_SETS)
@pytest.mark.parametrize("dtype", DSTACK_DTYPES)
def test_dstack_out(shape_set, dtype):
    inp = [tu.make_input(dtype, s, _MAIN_RANGE) for s in shape_set]
    ref_inp = [tu.to_reference(t) for t in inp]

    ref_shape = torch.ops.aten.dstack(ref_inp).shape
    ref_out = torch.empty(ref_shape, dtype=dtype, device=ref_inp[0].device)
    ref_ret = torch.ops.aten.dstack.out(ref_inp, out=ref_out)

    out = torch.empty(ref_shape, dtype=dtype, device=inp[0].device)
    res_ret = flag_gems.dstack(inp, out=out)

    # The .out variant must return the out tensor itself (alias semantics).
    assert res_ret.data_ptr() == out.data_ptr()
    tu.assert_result_equal(res_ret, ref_ret)
    tu.assert_result_equal(out, ref_out)


@pytest.mark.dstack
@pytest.mark.parametrize("shape_set", _DSTACK_EMPTY_SHAPE_SETS)
@pytest.mark.parametrize("dtype", DSTACK_DTYPES)
def test_dstack_empty_inputs(shape_set, dtype):
    inp = [tu.make_input(dtype, s, _MAIN_RANGE) for s in shape_set]
    ref_inp = [tu.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    res_out = flag_gems.dstack(inp)

    _assert_dstack_output(res_out, ref_out)


@pytest.mark.dstack
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(DSTACK_DTYPES))
)
def test_dstack_nan_inf(dtype, scenario):
    values = tu.make_special_input(dtype, scenario)
    inp = [values, values]
    ref_inp = [tu.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    res_out = flag_gems.dstack(inp)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.dstack
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_dstack_complex(dtype):
    inp = [
        tu.make_input(dtype, (4, 5, 6), _MAIN_RANGE),
        tu.make_input(dtype, (4, 5, 7), _MAIN_RANGE),
    ]
    ref_inp = [tu.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    res_out = flag_gems.dstack(inp)

    _assert_dstack_output(res_out, ref_out)


@pytest.mark.dstack_backward
@pytest.mark.parametrize("shape_set", _DSTACK_BACKWARD_SHAPE_SETS)
@pytest.mark.parametrize(
    "dtype",
    tu.selected_cases(
        [d for d in DSTACK_DTYPES if d.is_floating_point or d.is_complex]
    ),
)
def test_dstack_backward(shape_set, dtype):
    inp = [tu.make_input(dtype, s, _MAIN_RANGE).requires_grad_() for s in shape_set]
    ref_inp = [tu.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    grad = tu.make_input(dtype, ref_out.shape, _MAIN_RANGE)
    ref_grad = tu.to_reference(grad)
    ref_in_grads = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)

    res_out = flag_gems.dstack(inp)
    tu.assert_result_equal(res_out, ref_out)

    assert res_out.requires_grad
    res_in_grads = torch.autograd.grad(res_out, inp, grad_outputs=grad)
    for res_g, ref_g in zip(res_in_grads, ref_in_grads):
        tu.assert_result_equal(res_g, ref_g)


@pytest.mark.dstack_negative
def test_dstack_empty_list():
    with pytest.raises(RuntimeError):
        torch.ops.aten.dstack([])
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.dstack([])


@pytest.mark.dstack_negative
@pytest.mark.parametrize(
    "shape_set",
    [
        [(2, 3), (4, 3)],  # dim 0 mismatch: (2,3,1) vs (4,3,1)
        [(3,), (2, 3)],  # 1-D (1,3,1) vs 2-D (2,3,1): dim 0 mismatch
        [(4, 5, 6), (4, 7, 6)],  # dim 1 mismatch
    ],
)
def test_dstack_mismatched_shapes(shape_set):
    inp = [tu.make_input(torch.float32, s, _MAIN_RANGE) for s in shape_set]
    ref_inp = [tu.to_reference(t) for t in inp]

    with pytest.raises(RuntimeError):
        torch.ops.aten.dstack(ref_inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.dstack(inp)


@pytest.mark.dstack_negative
def test_dstack_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten.dstack([torch.zeros(2, device=flag_gems.device), 3.14])
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.dstack([torch.zeros(2, device=flag_gems.device), 3.14])
