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

from . import test_utils as tu

# aten::movedim permutes the axis order and returns a view that aliases the
# input storage: no element is copied and no value is rounded, so values are
# compared exactly and the stride / offset / alias state is checked separately.
# Both overloads (scalar axes and axis lists) are exercised by calling the
# native packet, which dispatches on the argument type.
_DTYPE_CAPABILITY = {
    torch.float64: "support_fp64",
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name, False))


_DTYPES = [
    dtype
    for dtype in tu.REQUIRED_DTYPES + [torch.float64, torch.bool, torch.complex64]
    if _dtype_supported(dtype)
]
_BACKWARD_DTYPES = [
    dtype for dtype in _DTYPES if dtype.is_floating_point or dtype.is_complex
]


def _assert_view_of(res, inp, ref, ref_inp):
    """Stride / offset / alias state the shared value assertions do not cover."""
    assert res.stride() == ref.stride()
    assert res.storage_offset() == ref.storage_offset()
    assert res._is_view() == ref._is_view()
    assert res.device == inp.device
    # Out-of-place view: the input keeps the metadata its reference has.
    assert inp.stride() == ref_inp.stride()
    assert inp.storage_offset() == ref_inp.storage_offset()
    assert res is not inp
    assert inp.shape == ref_inp.shape
    assert res.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()


def _grid_moves(rank):
    """One move per call form; rank 0 and 1 admit only the identity move."""
    if rank < 2:
        return [(0, 0), ([0], [0])]
    last = rank - 1
    return [(0, last), ([0, last], [last, 0])]


@pytest.mark.movedim
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("move_index", [0, 1])
def test_movedim(shape, value_range, dtype, move_index):
    source, destination = _grid_moves(len(shape))[move_index]
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.movedim(ref_inp, source, destination)
    res_out = flag_gems.movedim(inp, source, destination)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_of(res_out, inp, ref_out, ref_inp)


# Call forms: scalar axes, single- and multi-axis lists, negative axes, the
# empty-list identity and the rank boundaries (rank 0/1 only admit identity).
# These rows are small, so both levels collect all of them.
_FORM_ROWS = [
    (torch.float32, ["-1", "1"], (2, 19, 7), 1, 2),
    (torch.float32, ["-1", "1"], (2, 19, 7), -1, 0),
    (torch.float32, ["-1", "1"], (2, 19, 7), 0, 0),
    (torch.float32, ["-1", "1"], (2, 19, 7), [0, 2], [2, 0]),
    (torch.float32, ["-1", "1"], (2, 19, 7), [0, 1, 2], [2, 1, 0]),
    (torch.float32, ["-1", "1"], (2, 19, 7), [0, 1], [-1, -2]),
    (torch.float32, ["-1", "1"], (2, 19, 7), [], []),
    (torch.int32, ["min", "0"], (1024, 1024), 0, 1),
    (torch.float16, ["0", "max"], (20, 320, 15), [0, 2], [1, 0]),
    (torch.uint8, ["0", "max"], (16, 128, 64, 60), 3, 0),
    (torch.bfloat16, ["-1", "1"], (16, 7, 57, 32, 29), [1, 4], [4, 1]),
    (torch.float32, ["-1", "1"], (256,), 0, 0),
    (torch.float32, ["-1", "1"], (1,), [0], [0]),
    (torch.float32, ["-1", "1"], (), 0, 0),
    (torch.float32, ["-1", "1"], (), [0], [0]),
    (torch.float8_e4m3fn, ["-1", "1"], (2, 19, 7), [1], [0]),
]


@pytest.mark.movedim
@pytest.mark.parametrize("dtype,value_range,shape,source,destination", _FORM_ROWS)
def test_movedim_dim_forms(dtype, value_range, shape, source, destination):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.movedim(ref_inp, source, destination)
    res_out = flag_gems.movedim(inp, source, destination)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_of(res_out, inp, ref_out, ref_inp)


# The result has to compose with the input's own layout, so every row feeds a
# non-contiguous and/or offset input; the small rows stay in quick as well.
_LAYOUT_ROWS = tu.selected_cases(
    [
        ("sliced", (6, 8, 5), 0, -1, torch.float32),
        ("transposed", (5, 4, 3), 0, 2, torch.float32),
        ("sliced", (16, 128, 64, 60), 1, -1, torch.bfloat16),
        ("transposed", (16, 7, 57, 32, 29), 0, 2, torch.float16),
        ("empty", (0, 3, 4), 0, 2, torch.float32),
    ],
    quick=[
        ("sliced", (6, 8, 5), 0, -1, torch.float32),
        ("transposed", (5, 4, 3), 0, 2, torch.float32),
        ("empty", (0, 3, 4), 0, 2, torch.float32),
    ],
)


def _layout_input(kind, shape, dtype, value_range):
    """Build one non-contiguous and/or offset input per layout kind."""
    if kind == "empty":
        return tu.make_input(dtype, shape, value_range)
    if kind == "transposed":
        base_shape = (shape[1], shape[0]) + tuple(shape[2:])
        return tu.make_input(dtype, base_shape, value_range).transpose(0, 1)
    wide_shape = (shape[0], shape[1] * 2) + tuple(shape[2:])
    return tu.make_input(dtype, wide_shape, value_range)[:, 1::2]


@pytest.mark.movedim
@pytest.mark.parametrize("kind,shape,source,destination,dtype", _LAYOUT_ROWS)
def test_movedim_preserves_input_layout(kind, shape, source, destination, dtype):
    inp = _layout_input(kind, shape, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.movedim(ref_inp, source, destination)
    res_out = flag_gems.movedim(inp, source, destination)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_of(res_out, inp, ref_out, ref_inp)


# The lazy conjugate bit is input state, not a stored value: a view of a
# conjugated tensor has to stay conjugated.
_CONJ_ROWS = [
    (torch.complex64, (2, 19, 7), 0, 1),
    (torch.complex64, (20, 320, 15), 1, -1),
    (torch.complex64, (2, 19, 7), [0, 2], [2, 0]),
]


@pytest.mark.movedim
@pytest.mark.parametrize("dtype,shape,source,destination", _CONJ_ROWS)
def test_movedim_preserves_conjugate_bit(dtype, shape, source, destination):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).conj()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.movedim(ref_inp, source, destination)
    res_out = flag_gems.movedim(inp, source, destination)

    assert res_out.is_conj() == ref_out.is_conj()
    tu.assert_result_equal(res_out, ref_out)
    _assert_view_of(res_out, inp, ref_out, ref_inp)


# No element is copied, so a write through the result has to reach the input.
# The two small rows stay in quick; the 5-D row is default-only.
_MUTATION_ROWS = tu.selected_cases(
    [
        ((6, 8, 5), 0, -1),
        ((5, 4, 3), [0, 2], [2, 0]),
        ((16, 7, 57, 32, 29), 1, -1),
    ],
    quick=[
        ((6, 8, 5), 0, -1),
        ((5, 4, 3), [0, 2], [2, 0]),
    ],
)


@pytest.mark.movedim
@pytest.mark.parametrize("shape,source,destination", _MUTATION_ROWS)
def test_movedim_view_writes_through(shape, source, destination):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.movedim(ref_inp, source, destination)
    res_out = flag_gems.movedim(inp, source, destination)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_of(res_out, inp, ref_out, ref_inp)

    res_out.fill_(2.5)
    ref_out.fill_(2.5)
    tu.assert_result_equal(inp, ref_inp)


# Gradients are taken with respect to the original leaf, so the upstream
# gradient really travels back through the move; a pure relayout adds nothing,
# so the gradient is compared exactly.
_BACKWARD_ROWS = tu.selected_cases(
    [
        (dtype, shape, source, destination)
        for shape, source, destination in (
            ((4, 5, 6), 0, -1),
            ((20, 320, 15), 2, 0),
            ((2, 19, 7), [0, 2], [2, 0]),
            ((16, 128, 64, 60), 1, -1),
        )
        for dtype in _BACKWARD_DTYPES
    ],
    quick=[],
)


@pytest.mark.movedim
@pytest.mark.parametrize("dtype,shape,source,destination", _BACKWARD_ROWS)
def test_movedim_backward(dtype, shape, source, destination):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.movedim(ref_inp, source, destination)
    upstream = tu.make_input(dtype, tuple(ref_out.shape), ["-1", "1"])
    ref_upstream = tu.to_reference(upstream)
    res_out = flag_gems.movedim(inp, source, destination)

    # The forward relayout is checked before the gradient path is exercised.
    tu.assert_result_equal(res_out, ref_out)
    _assert_view_of(res_out, inp, ref_out, ref_inp)

    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]
    ref_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)[0]

    tu.assert_result_equal(res_grad, ref_grad)


# The shared payloads are 1-D; a lossless reshape gives axis 0 a real move
# while keeping every payload value, so NaN/Inf classification is unchanged.
_SPECIAL_DTYPES = [dtype for dtype in _DTYPES if dtype.is_floating_point]


@pytest.mark.movedim
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_SPECIAL_DTYPES), quick=[]),
)
def test_movedim_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario).reshape(1, -1)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.movedim(ref_inp, [0, 1], [1, 0])
    res_out = flag_gems.movedim(inp, [0, 1], [1, 0])

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_of(res_out, inp, ref_out, ref_inp)


# Native validation rejects an out-of-range axis with IndexError and a
# malformed or non-integer axis with RuntimeError; either class is accepted for
# the same rejection and every row is collected in both levels.
_INVALID_DIM_ROWS = [
    ((2, 3, 4), 0, 3, (IndexError, RuntimeError)),
    ((2, 3, 4), -4, 0, (IndexError, RuntimeError)),
    ((2, 3, 4), [0, 0], [1, 2], (IndexError, RuntimeError)),
    ((2, 3, 4), [0, 1], [1, 1], (IndexError, RuntimeError)),
    ((2, 3, 4), [0, 1], [1], (RuntimeError, TypeError)),
    ((), 0, 1, (IndexError, RuntimeError)),
    ((1,), [0, 0], [0, 0], (IndexError, RuntimeError)),
    ((2, 3, 4), [0, None], [1, 2], (RuntimeError, TypeError)),
    ((2, 3, 4), [0, 1.5], [1, 2], (RuntimeError, TypeError)),
    ((2, 3, 4), 0, "a", (RuntimeError, TypeError)),
]


@pytest.mark.movedim
@pytest.mark.parametrize("shape,source,destination,expected", _INVALID_DIM_ROWS)
def test_movedim_rejects_invalid_axes(shape, source, destination, expected):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    with pytest.raises(expected):
        flag_gems.movedim(inp, source, destination)


@pytest.mark.movedim
def test_movedim_rejects_non_tensor_input():
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.movedim(3, 0, 0)
