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

# Optional dtypes join the grid only when the backend reports the capability;
# every dtype outside this map stays unconditional, so a supported case is never
# dropped and no test probes support or skips at run time.
_DTYPE_CAPABILITY = {
    torch.bfloat16: "support_bf16",
    torch.float64: "support_fp64",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name, False))


SUPPORTED_DTYPES = [
    dtype
    for dtype in tu.REQUIRED_DTYPES + [torch.float64, torch.bool, torch.complex64]
    if _dtype_supported(dtype)
]

# Rank 0 and rank 1 only accept the (0, 0) axis pair.
_AXIS_BY_RANK = {0: (0, 0), 1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 3), 5: (2, 4)}
SHAPE_ROWS = [(shape, *_AXIS_BY_RANK[len(shape)]) for shape in tu.selected_shapes()]


def _assert_view_of(res_out, ref_out, inp, ref_inp):
    assert res_out is not inp
    assert inp.shape == ref_inp.shape
    assert inp.stride() == ref_inp.stride()
    assert inp.storage_offset() == ref_inp.storage_offset()
    # The native result states the expected layout, and moveaxis is Tensor(a) ->
    # Tensor(a): the candidate must be a view of its own input with that layout
    # and untouched input metadata.
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.dtype == ref_out.dtype
    assert res_out._is_view() == ref_out._is_view()
    assert res_out.data_ptr() == inp.data_ptr()
    assert res_out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
    assert res_out.device == inp.device


def _strided_view(tensor, state):
    if state == "transposed":
        return tensor.transpose(-1, -2)
    if state == "column_step":
        return tensor[:, ::2]
    if state == "offset_window":
        return tensor[2:6, 1:5]
    if state == "both_steps":
        return tensor[::3, ::4]
    raise ValueError(f"unknown layout state: {state}")


def _assert_operand_unchanged(inp, snapshot):
    assert (inp.shape, inp.stride(), inp.storage_offset(), inp.data_ptr()) == snapshot


@pytest.mark.moveaxis
@pytest.mark.parametrize("shape,source,destination", SHAPE_ROWS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_moveaxis_int_axis(shape, source, destination, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    snapshot = (inp.shape, inp.stride(), inp.storage_offset(), inp.data_ptr())

    ref_out = torch.ops.aten.moveaxis(ref_inp, source, destination)
    res_out = flag_gems.moveaxis(inp, source, destination)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_of(res_out, ref_out, inp, ref_inp)
    _assert_operand_unchanged(inp, snapshot)


def _intlist_rows(shapes):
    # The empty pair is a documented no-op valid for every rank, including 0-D.
    rows = [(shape, [], []) for shape in shapes]
    for shape in shapes:
        rank = len(shape)
        if rank >= 2:
            rows.append((shape, [0, 1], [1, 0]))
        if rank >= 3:
            rows.append((shape, [0, 2], [2, 0]))
            rows.append((shape, [-1, -2], [0, 1]))
        if rank >= 4:
            rows.append((shape, [0, 1, 2], [3, 2, 1]))
    return rows


INTLIST_ROWS = _intlist_rows(tu.selected_shapes())


@pytest.mark.moveaxis
@pytest.mark.parametrize("shape,source,destination", INTLIST_ROWS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_moveaxis_intlist_axis(shape, source, destination, dtype):
    # A metadata move never reads stored values, so the int[] overload runs with
    # one representative range while test_moveaxis_int_axis carries the grid.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.moveaxis(ref_inp, source, destination)
    res_out = flag_gems.moveaxis(inp, source, destination)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_of(res_out, ref_out, inp, ref_inp)


IDENTITY_ROWS = [
    (shape, axis, axis)
    for shape in tu.selected_shapes()
    for axis in ((0,) if len(shape) < 2 else (0, -1))
]


@pytest.mark.moveaxis
@pytest.mark.parametrize("shape,source,destination", IDENTITY_ROWS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_moveaxis_identity_keeps_layout(shape, source, destination, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.moveaxis(ref_inp, source, destination)
    res_out = flag_gems.moveaxis(inp, source, destination)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_of(res_out, ref_out, inp, ref_inp)
    # Native also hands back a fresh view object for an identity mapping, so the
    # unchanged layout over shared storage is asserted, not object identity.
    assert res_out is not inp
    assert res_out.shape == inp.shape
    assert res_out.stride() == inp.stride()
    assert res_out.storage_offset() == inp.storage_offset()


NONCONTIG_ROWS = [
    ((64, 96), "transposed", 0, 1),
    ((64, 96), "column_step", 1, 0),
    ((64, 96), "offset_window", 0, 1),
    ((64, 96), "both_steps", -1, 0),
    ((8, 24, 12), "transposed", 0, 2),
    ((8, 24, 12), "column_step", 2, 0),
]

NONCONTIG_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int32,
    torch.int64,
]


@pytest.mark.moveaxis
@pytest.mark.parametrize("shape,state,source,destination", NONCONTIG_ROWS)
@pytest.mark.parametrize("dtype", NONCONTIG_DTYPES)
def test_moveaxis_noncontiguous_input(shape, state, source, destination, dtype):
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = _strided_view(base, state)
    ref_inp = _strided_view(ref_base, state)

    ref_out = torch.ops.aten.moveaxis(ref_inp, source, destination)
    res_out = flag_gems.moveaxis(inp, source, destination)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_of(res_out, ref_out, inp, ref_inp)


# A stride-0 expanded operand is ordinary layout semantics for a view operator
# and stays in the quick subset, so all three cheap rows run in every mode.
EXPANDED_ROWS = [
    ((1, 5), (4, 5), 0, 1),
    ((1, 5), (4, 5), 1, 0),
    ((1, 4, 5), (3, 4, 5), 0, 2),
]


@pytest.mark.moveaxis
@pytest.mark.parametrize("base_shape,expanded_shape,source,destination", EXPANDED_ROWS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32])
def test_moveaxis_expanded_input(
    base_shape, expanded_shape, source, destination, dtype
):
    base = tu.make_input(dtype, base_shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base.expand(expanded_shape)
    ref_inp = ref_base.expand(expanded_shape)

    ref_out = torch.ops.aten.moveaxis(ref_inp, source, destination)
    res_out = flag_gems.moveaxis(inp, source, destination)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_of(res_out, ref_out, inp, ref_inp)


@pytest.mark.moveaxis
@pytest.mark.parametrize("source,destination", [(0, 1), (1, 0)])
def test_moveaxis_conjugate_input(source, destination):
    # The lazy conjugate bit is view state that must survive the axis move
    # instead of being materialized.
    base = tu.make_input(torch.complex64, (6, 4), ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base.conj()
    ref_inp = ref_base.conj()

    ref_out = torch.ops.aten.moveaxis(ref_inp, source, destination)
    res_out = flag_gems.moveaxis(inp, source, destination)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.is_conj() == ref_out.is_conj()
    _assert_view_of(res_out, ref_out, inp, ref_inp)


@pytest.mark.moveaxis
@pytest.mark.parametrize("source,destination", [(0, 1), ([0, 1], [1, 0])])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_moveaxis_result_mutation_aliases_input(source, destination, dtype):
    inp = tu.make_input(dtype, (4, 5), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    res_out = flag_gems.moveaxis(inp, source, destination)
    ref_out = torch.ops.aten.moveaxis(ref_inp, source, destination)
    tu.assert_result_equal(res_out, ref_out)

    # result[1, 0] is input[0, 1] for both axis forms used here; the identical
    # write on the input comparison below shows the result aliases its input.
    res_out[1, 0] = 3
    ref_out[1, 0] = 3

    tu.assert_result_equal(inp, ref_inp)


EMPTY_ROWS = [
    ((0,), 0, 0),
    ((0, 3), 0, 1),
    ((3, 0), 0, 1),
    ((0, 3, 4), 0, 2),
    ((0, 3, 4), -1, 0),
]


@pytest.mark.moveaxis
@pytest.mark.parametrize("shape,source,destination", EMPTY_ROWS)
def test_moveaxis_empty_tensor(shape, source, destination):
    # Zero-filled rather than uninitialized: an empty view holds no elements,
    # and the comparison must never read undefined memory.
    inp = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.moveaxis(ref_inp, source, destination)
    res_out = flag_gems.moveaxis(inp, source, destination)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_of(res_out, ref_out, inp, ref_inp)


BACKWARD_ROWS = tu.selected_cases(
    [
        ((4, 5), 0, 1),
        ((4, 5), 1, 0),
        ((6, 4, 5), 0, 2),
        ((6, 4, 5), -1, 0),
    ],
    quick=[],
)

BACKWARD_DTYPES = [
    dtype for dtype in SUPPORTED_DTYPES if dtype.is_floating_point or dtype.is_complex
]


@pytest.mark.moveaxis
@pytest.mark.parametrize("shape,source,destination", BACKWARD_ROWS)
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test_moveaxis_backward(shape, source, destination, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    inp.requires_grad_(True)
    ref_inp.requires_grad_(True)

    res_out = flag_gems.moveaxis(inp, source, destination)
    ref_out = torch.ops.aten.moveaxis(ref_inp, source, destination)
    upstream = tu.make_input(dtype, tuple(ref_out.shape), ["-1", "1"])
    ref_upstream = tu.to_reference(upstream)

    tu.assert_result_equal(res_out, ref_out)

    # The gradient with respect to the original leaf is a pure relayout of the
    # upstream gradient, so it is compared exactly.
    (res_grad,) = torch.autograd.grad(res_out, inp, grad_outputs=upstream)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)

    tu.assert_result_equal(res_grad, ref_grad)
    assert res_grad.shape == inp.shape


@pytest.mark.moveaxis
@pytest.mark.parametrize(
    "source,destination", tu.selected_cases([(0, 1), ([0, 1], [1, 0])], quick=[])
)
def test_moveaxis_backward_accumulates(source, destination):
    inp = tu.make_input(torch.float32, (3, 4), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    inp.requires_grad_(True)
    ref_inp.requires_grad_(True)

    res_out = flag_gems.moveaxis(inp, source, destination)
    ref_out = torch.ops.aten.moveaxis(ref_inp, source, destination)
    tu.assert_result_equal(res_out, ref_out)

    res_loss = res_out.sum()
    ref_loss = ref_out.sum()

    (res_grad,) = torch.autograd.grad(res_loss, inp)
    (ref_grad,) = torch.autograd.grad(ref_loss, ref_inp)

    tu.assert_result_equal(res_grad, ref_grad)


SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(SUPPORTED_DTYPES), quick=[])


@pytest.mark.moveaxis
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test_moveaxis_special_values(dtype, scenario):
    # moveaxis computes nothing, so nan/inf payloads must survive unchanged;
    # matching NaNs are the native semantics for this operator.
    inp = tu.make_special_input(dtype, scenario).reshape(1, -1)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.moveaxis(ref_inp, 0, 1)
    res_out = flag_gems.moveaxis(inp, 0, 1)

    tu.assert_result_equal(res_out, ref_out)
    _assert_view_of(res_out, ref_out, inp, ref_inp)


NEG_BASE = (4, 5)


@pytest.mark.moveaxis
@pytest.mark.parametrize("bad_dim", [2, -3])
def test_moveaxis_int_dim_out_of_range(bad_dim):
    inp = tu.make_input(torch.float32, NEG_BASE, ["-1", "1"])
    with pytest.raises((IndexError, RuntimeError)):
        flag_gems.moveaxis(inp, bad_dim, 0)


@pytest.mark.moveaxis
def test_moveaxis_int_destination_out_of_range():
    inp = tu.make_input(torch.float32, NEG_BASE, ["-1", "1"])
    with pytest.raises((IndexError, RuntimeError)):
        flag_gems.moveaxis(inp, 0, 5)


@pytest.mark.moveaxis
def test_moveaxis_intlist_length_mismatch():
    inp = tu.make_input(torch.float32, NEG_BASE, ["-1", "1"])
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        flag_gems.moveaxis(inp, [0, 1], [1])


@pytest.mark.moveaxis
@pytest.mark.parametrize("source,destination", [([0, 0], [1, 1]), ([0, 1], [1, 1])])
def test_moveaxis_intlist_repeated_axis(source, destination):
    inp = tu.make_input(torch.float32, NEG_BASE, ["-1", "1"])
    with pytest.raises((RuntimeError, ValueError, IndexError)):
        flag_gems.moveaxis(inp, source, destination)


@pytest.mark.moveaxis
@pytest.mark.parametrize("bad_axis", [0.5, -1.5])
def test_moveaxis_int_non_integer_axis(bad_axis):
    inp = tu.make_input(torch.float32, NEG_BASE, ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.moveaxis(inp, bad_axis, 1)


@pytest.mark.moveaxis
def test_moveaxis_intlist_non_integer_axis():
    inp = tu.make_input(torch.float32, NEG_BASE, ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.moveaxis(inp, [0.0], [1])


@pytest.mark.moveaxis
def test_moveaxis_non_tensor_input():
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.moveaxis([[1, 2], [3, 4]], 0, 1)
