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

# aten::select_copy only moves elements, so every result is compared exactly.
# The capability flags are static (read once at import), so no dtype support
# probe runs while collecting or running these tests.
_SELECT_DTYPES = [torch.int8, torch.uint8, torch.float32, torch.float16, torch.int32]
if utils.fp8_is_supported:
    _SELECT_DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]
if utils.bf16_is_supported:
    _SELECT_DTYPES.append(torch.bfloat16)
if utils.int64_is_supported:
    _SELECT_DTYPES.append(torch.int64)
_SELECT_DTYPES += [torch.bool, torch.complex64]
if utils.fp64_is_supported:
    _SELECT_DTYPES.append(torch.float64)

# Value grid: the spec's seven shapes minus the 0-dim entry, because the native
# operator rejects a 0-dim input ("select() cannot be applied to a 0-dim
# tensor"); that rejection is asserted in the negative cases below.
_SELECT_CASES = tu.selected_cases(
    [
        ((1,), 0, 0),
        ((256,), 0, 128),
        ((1024, 1024), 1, 1023),
        ((20, 320, 15), 0, 19),
        ((16, 128, 64, 60), 2, 63),
        ((16, 7, 57, 32, 29), -1, 28),
    ],
    quick=[((2, 19, 7), 1, 18)],
)

# dim/index sweep: both arguments are normalized ints (negative values are
# valid), so every axis of a rank-3 input is exercised with its first and last
# index. Two dtypes are enough: the copied values are dtype-independent and the
# full dtype grid runs in test_select_copy.
_DIM_INDEX_CASES = tu.selected_cases(
    [
        ((17, 33, 9), 0, 0),
        ((17, 33, 9), 0, 16),
        ((17, 33, 9), 1, 0),
        ((17, 33, 9), 1, 32),
        ((17, 33, 9), 2, 0),
        ((17, 33, 9), 2, 8),
        ((17, 33, 9), -1, -1),
        ((17, 33, 9), -2, -33),
        ((17, 33, 9), -3, -17),
    ],
    quick=[],
)
_DIM_INDEX_DTYPES = tu.selected_cases([torch.float32, torch.int32], quick=[])

# bool is a valid int argument for this schema (True == 1) and the native
# operator accepts it for both dim and index.
_BOOL_PARAM_CASES = tu.selected_cases([(True, 2), (0, True)], quick=[])

# The native `int_out` overload is callable on this backend with a correctly
# shaped buffer (probed), so the out variant is called directly in both the
# reference and candidate paths instead of being simulated with default+copy_.
_OUT_CASES = tu.selected_cases(
    [
        ((1,), 0, 0),
        ((1024, 1024), 1, 500),
        ((16, 128, 64, 60), 2, 61),
        ((8, 12, 10), -2, 5),
        ((3, 0, 5), 0, 1),
    ],
    quick=[((2, 19, 7), 0, 1)],
)
_STRIDED_OUT_CASES = tu.selected_cases([((4, 8, 6), 0, 2)], quick=[])

# Non-contiguous inputs with a nonzero storage offset; select_copy must read
# through the real strides of the selected axis.
_LAYOUT_CASES = tu.selected_cases(["transpose", "strided_slice"], quick=[])
_LAYOUT_DTYPES = tu.selected_cases([torch.float16, torch.float32, torch.int8], quick=[])
_MUTATION_DTYPES = tu.selected_cases(
    [torch.float16, torch.float32, torch.int32, torch.uint8], quick=[]
)
_EMPTY_DTYPES = tu.selected_cases([torch.float16, torch.int8, torch.float32], quick=[])

# select_copy is differentiable: its backward scatters each upstream entry once
# into zeros, so gradients are compared exactly like the forward result.
_BACKWARD_CASES = tu.selected_cases(
    [((16, 32), 0, 5), ((4, 8, 16), 1, 3), ((4, 8, 16), -2, -5)], quick=[]
)
_BACKWARD_DTYPES = tu.selected_cases(
    [
        dtype
        for dtype in (torch.float16, torch.float32, torch.bfloat16)
        if dtype != torch.bfloat16 or utils.bf16_is_supported
    ],
    quick=[],
)
_BACKWARD_VIEW_LAYOUTS = tu.selected_cases(["transpose", "expand"], quick=[])
_BACKWARD_VIEW_DTYPES = tu.selected_cases([torch.float16, torch.float32], quick=[])

# nan/inf scenarios are default-only. tu.special_value_cases keeps only the
# scenarios each dtype can represent (float8_e4m3fn has no infinity) and its
# payload places the special values in the first three slots.
_SPECIAL_CASES = tu.selected_cases(
    [
        (dtype, scenario, index)
        for dtype, scenario in tu.special_value_cases(_SELECT_DTYPES)
        for index in (0, 1, 2)
    ],
    quick=[],
)


def _out_shape(shape, dim):
    """Shape of the selection, normalizing a negative dim first."""
    dim = dim + len(shape) if dim < 0 else dim
    return shape[:dim] + shape[dim + 1 :]


def _strided_input(dtype, layout):
    """Return (parent, view): a non-contiguous view with a nonzero storage
    offset together with the tensor whose storage it aliases, so a write into
    elements the view does not cover is detectable."""
    parent = tu.make_input(dtype, (8, 12, 10), ["-1", "1"])
    if layout == "transpose":
        return parent, parent.transpose(1, 2)[1:, :, 2:]
    return parent, parent[::2, 1:11:3, ::2]


def _sentinel_tensor(shape, dtype, device):
    """Out buffer pre-filled with a value the copied data cannot contain, so a
    buffer that is only returned instead of written fails the comparison."""
    if dtype == torch.bool:
        value = True
    elif dtype.is_complex:
        value = 1.25 + 0j
    elif dtype.is_floating_point:
        value = 1.25
    else:
        value = 3
    return torch.full(shape, value, dtype=dtype, device=device)


def _assert_dense_copy(res_out, ref_out, inp):
    """select_copy returns a materialized dense copy, not the `select` view."""
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == 0
    if inp.numel() > 0:
        assert res_out.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()
    # An empty result may share its input's (null) storage pointer, so isolation
    # is not asserted from pointers alone for that case.


@pytest.mark.select_copy
@pytest.mark.parametrize("shape,dim,index", _SELECT_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SELECT_DTYPES)
def test_select_copy(shape, dim, index, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.select_copy.int(ref_inp, dim, index)
    res_out = flag_gems.select_copy(inp, dim, index)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    _assert_dense_copy(res_out, ref_out, inp)


@pytest.mark.select_copy
@pytest.mark.parametrize("shape,dim,index", _DIM_INDEX_CASES)
@pytest.mark.parametrize("dtype", _DIM_INDEX_DTYPES)
def test_select_copy_dim_index_boundaries(shape, dim, index, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.select_copy.int(ref_inp, dim, index)
    res_out = flag_gems.select_copy(inp, dim, index)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.select_copy
@pytest.mark.parametrize("dim,index", _BOOL_PARAM_CASES)
def test_select_copy_accepts_bool_parameters(dim, index):
    inp = tu.make_input(torch.float32, (3, 4, 5), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.select_copy.int(ref_inp, dim, index)
    res_out = flag_gems.select_copy(inp, dim, index)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.select_copy
@pytest.mark.parametrize("shape,dim,index", _OUT_CASES)
@pytest.mark.parametrize("dtype", _SELECT_DTYPES)
def test_select_copy_out(shape, dim, index, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    out_shape = _out_shape(shape, dim)

    buf = _sentinel_tensor(out_shape, dtype, flag_gems.device)
    ref_buf = _sentinel_tensor(out_shape, dtype, ref_inp.device)

    ref_out = torch.ops.aten.select_copy.int_out(ref_inp, dim, index, out=ref_buf)
    res_out = flag_gems.select_copy(inp, dim, index, out=buf)

    assert res_out is buf
    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.select_copy
@pytest.mark.parametrize("shape,dim,index", _STRIDED_OUT_CASES)
def test_select_copy_out_into_strided_view(shape, dim, index):
    # `int_out` writes into a non-contiguous, offset view of a larger tensor:
    # only the view region may be written, the parent's other elements must keep
    # their values, and the buffer must stay backed by the parent's storage
    # instead of being swapped for a fresh allocation.
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    parent = torch.full((17, 6), -7.0, dtype=torch.float32, device=flag_gems.device)
    ref_parent = tu.to_reference(parent)
    buf = parent[1:16:2]
    ref_buf = ref_parent[1:16:2]

    ref_out = torch.ops.aten.select_copy.int_out(ref_inp, dim, index, out=ref_buf)
    res_out = flag_gems.select_copy(inp, dim, index, out=buf)

    assert res_out is buf
    assert res_out.untyped_storage().data_ptr() == parent.untyped_storage().data_ptr()
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(parent, ref_parent)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.select_copy
@pytest.mark.parametrize("layout", _LAYOUT_CASES)
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
@pytest.mark.parametrize("dim,index", [(2, 0), (2, -1)])
def test_select_copy_strided_input(layout, dtype, dim, index):
    parent, inp = _strided_input(dtype, layout)
    ref_parent = tu.to_reference(parent)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.select_copy.int(ref_inp, dim, index)
    res_out = flag_gems.select_copy(inp, dim, index)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(parent, ref_parent)
    _assert_dense_copy(res_out, ref_out, inp)


@pytest.mark.select_copy
@pytest.mark.parametrize("dtype", _EMPTY_DTYPES)
def test_select_copy_empty_result(dtype):
    # (3, 0, 5) with dim 0 index 1 -> (0, 5): selecting a present, nonzero axis
    # of a zero-sized tensor yields an empty result instead of raising.
    inp = tu.make_input(dtype, (3, 0, 5), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.select_copy.int(ref_inp, 0, 1)
    res_out = flag_gems.select_copy(inp, 0, 1)

    assert res_out.shape == (0, 5)
    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.select_copy
@pytest.mark.parametrize("dtype", _MUTATION_DTYPES)
def test_select_copy_result_does_not_alias_input(dtype):
    inp = tu.make_input(dtype, (8, 16, 4), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.select_copy.int(ref_inp, 1, 5)
    res_out = flag_gems.select_copy(inp, 1, 5)
    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)

    fill = 1.5 if dtype.is_floating_point else 3
    res_out.fill_(fill)
    ref_out.fill_(fill)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.select_copy
@pytest.mark.parametrize("shape,dim,index", _BACKWARD_CASES)
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
def test_select_copy_backward(shape, dim, index, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    ref_inp = tu.to_reference(inp)
    upstream = tu.make_input(dtype, _out_shape(shape, dim), ["-1", "1"])
    ref_upstream = tu.to_reference(upstream)

    ref_out = torch.ops.aten.select_copy.int(ref_inp, dim, index)
    ref_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)[0]

    res_out = flag_gems.select_copy(inp, dim, index)
    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]
    assert res_grad.device == inp.device

    # backward scatters every upstream entry exactly once into zeros, so the
    # gradient introduces no rounding and is compared exactly.
    tu.assert_result_equal(res_grad, ref_grad)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.select_copy
@pytest.mark.parametrize("layout", _BACKWARD_VIEW_LAYOUTS)
@pytest.mark.parametrize("dtype", _BACKWARD_VIEW_DTYPES)
def test_select_copy_backward_view_input(layout, dtype):
    base_tensor = tu.make_input(dtype, (8, 16), ["-1", "1"])
    view = (
        base_tensor.transpose(0, 1)
        if layout == "transpose"
        else base_tensor[0].expand(8, 16)
    )
    inp = view.requires_grad_()
    ref_base = tu.to_reference(base_tensor)
    ref_inp = tu.to_reference(inp)
    upstream = tu.make_input(dtype, ref_inp.shape[1:], ["-1", "1"])
    ref_upstream = tu.to_reference(upstream)

    ref_out = torch.ops.aten.select_copy.int(ref_inp, 0, 5)
    ref_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)[0]

    res_out = flag_gems.select_copy(inp, 0, 5)
    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]
    assert res_grad.device == inp.device

    tu.assert_result_equal(res_grad, ref_grad)
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(base_tensor, ref_base)


@pytest.mark.select_copy
@pytest.mark.parametrize("dtype,scenario,index", _SPECIAL_CASES)
def test_select_copy_special_values(dtype, scenario, index):
    inp = tu.make_special_input(dtype, scenario).reshape(1, -1)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.select_copy.int(ref_inp, 1, index)
    res_out = flag_gems.select_copy(inp, 1, index)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


# Negative cases call the candidate only; the native probe evidence belongs to
# generation, not to a second assertion block here.
@pytest.mark.select_copy
def test_select_copy_rejects_0_dim_input():
    inp = tu.make_input(torch.float32, (), ["-1", "1"])
    with pytest.raises(IndexError):
        flag_gems.select_copy(inp, 0, 0)


@pytest.mark.select_copy
@pytest.mark.parametrize("dim", [3, -4])
def test_select_copy_rejects_out_of_range_dim(dim):
    # Rank 3 accepts dim in [-3, 2]; 3 and -4 are the closest invalid values.
    inp = tu.make_input(torch.float32, (3, 4, 5), ["-1", "1"])
    with pytest.raises(IndexError):
        flag_gems.select_copy(inp, dim, 0)


@pytest.mark.select_copy
@pytest.mark.parametrize("index", [4, -5, 5, -6])
def test_select_copy_rejects_out_of_range_index(index):
    # dim 1 has extent 4, so valid indices are [-4, 3]; 4 / -5 are the closest
    # invalid values and 5 / -6 reach beyond them.
    inp = tu.make_input(torch.float32, (3, 4, 5), ["-1", "1"])
    with pytest.raises(IndexError):
        flag_gems.select_copy(inp, 1, index)


@pytest.mark.select_copy
@pytest.mark.parametrize("shape,dim,index", [((3, 0, 5), 1, 0), ((0,), 0, 0)])
def test_select_copy_rejects_index_on_empty_axis(shape, dim, index):
    # A zero-extent axis has no valid index even though the input tensor itself
    # is legal: (3, 0, 5) dim 1 and (0,) dim 0 both raise natively.
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    with pytest.raises(IndexError):
        flag_gems.select_copy(inp, dim, index)


@pytest.mark.select_copy
@pytest.mark.parametrize("dim,index", [(1.5, 0), (0, 1.5)], ids=["dim", "index"])
def test_select_copy_rejects_non_integer_arguments(dim, index):
    # The schema declares int arguments; the native dispatcher rejects a float
    # with a RuntimeError instead of truncating it.
    inp = tu.make_input(torch.float32, (3, 4), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.select_copy(inp, dim, index)


@pytest.mark.select_copy
def test_select_copy_rejects_non_tensor_input():
    with pytest.raises(RuntimeError):
        flag_gems.select_copy(1.5, 0, 0)
