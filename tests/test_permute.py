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

"""Correctness tests for ``aten::permute`` (a zero-copy relayout view)."""

import pytest
import torch

import flag_gems

from . import test_utils as tu

# aten::permute is a metadata-only view, so every dtype the shared framework can
# build is accepted. Probed with the real overload beyond the 9 spec dtypes:
# bool, int16, float64, complex64 and complex128.
SUPPORTED_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.int16,
    torch.float64,
    torch.complex64,
    torch.complex128,
]

# tu.special_value_cases keeps e4m3fn's nan-only case and gives e5m2 nan, inf
# and mixed, matching what each dtype can represent.
SPECIAL_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]

# torch.autograd.grad on the native operator succeeds for all of these (FP8 and
# complex included); the gradient of a permute is a pure relayout with no
# arithmetic, so it is compared exactly.
BACKWARD_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.complex64,
    torch.complex128,
]


@pytest.mark.permute
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_permute(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    # Reverse permutation: [] for 0-D, [0] for 1-D, both valid natively.
    dims = list(range(len(shape) - 1, -1, -1))

    ref_out = torch.ops.aten.permute(ref_inp, dims)
    res_out = flag_gems.permute(inp, dims)

    tu.assert_result_equal(res_out, ref_out)


# dims is required and has no schema default, so it cannot be omitted in a
# positive case; every accepted dims form is covered instead.
DIMS_ROWS = [
    ((2, 3, 4), [0, 1, 2]),
    ((2, 3, 4), [2, 1, 0]),
    ((2, 3, 4), [1, 2, 0]),
    ((2, 3, 4), [-1, -3, -2]),
    ((2, 3, 4), (0, -1, 1)),
    ((2, 3, 4, 5), [3, 0, 2, 1]),
]


@pytest.mark.permute
@pytest.mark.parametrize(
    "shape,dims", tu.selected_cases(DIMS_ROWS, quick=[((2, 3, 4), [2, 1, 0])])
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_permute_dims(shape, dims, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.permute(ref_inp, dims)
    res_out = flag_gems.permute(inp, dims)

    tu.assert_result_equal(res_out, ref_out)


# Source layouts: dense, offset window, column-strided slice, zero-stride expand
# view, plus the 0-D and 1-D forms.
VIEW_ROWS = [
    ((2, 3, 4), [2, 1, 0], "dense"),
    ((3, 4, 5), [1, 0, 2], "window"),
    ((4, 5), [1, 0], "colstep"),
    ((5, 4), [1, 0], "expand"),
    ((), [], "dense"),
    ((1,), [0], "dense"),
]


def _make_view_source(shape, kind):
    """Build a permute input whose layout is not necessarily contiguous."""
    if kind == "dense":
        return tu.make_input(torch.float32, shape, ["-1", "1"])
    if kind == "window":
        base = tu.make_input(
            torch.float32, (shape[0] + 3, shape[1] + 5, shape[2] + 7), ["-1", "1"]
        )
        return base[1 : 1 + shape[0], 2 : 2 + shape[1], 3 : 3 + shape[2]]
    if kind == "colstep":
        base = tu.make_input(torch.float32, (shape[0] * 2, shape[1]), ["-1", "1"])
        return base[::2]
    if kind == "expand":
        base = tu.make_input(torch.float32, (1, shape[1]), ["-1", "1"])
        return base.expand(shape)
    raise ValueError(f"unknown source layout {kind!r}")


@pytest.mark.permute
@pytest.mark.parametrize("case", tu.selected_cases(VIEW_ROWS, quick=VIEW_ROWS[:1]))
def test_permute_view_metadata(case):
    shape, dims, kind = case
    inp = _make_view_source(shape, kind)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.permute(ref_inp, dims)
    res_out = flag_gems.permute(inp, dims)

    tu.assert_result_equal(res_out, ref_out)
    # Metadata the shared value assertion does not cover: permute must stay a
    # view aliasing the input storage with permuted strides and the same offset.
    assert tuple(res_out.stride()) == tuple(inp.stride()[d] for d in dims)
    assert res_out.storage_offset() == inp.storage_offset()
    assert res_out._is_view()
    assert res_out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()


MUTATION_ROWS = [
    ((2, 3, 4), [2, 1, 0]),
    ((4, 5), [1, 0]),
]


@pytest.mark.permute
@pytest.mark.parametrize(
    "shape,dims", tu.selected_cases(MUTATION_ROWS, quick=MUTATION_ROWS[:1])
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32])
def test_permute_writes_through_the_view(shape, dims, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    res_out = flag_gems.permute(inp, dims)
    ref_out = torch.ops.aten.permute(ref_inp, dims)
    res_out.fill_(3)
    ref_out.fill_(3)

    tu.assert_result_equal(res_out, ref_out)
    # The write must land in the base tensor: a materialized copy would leave
    # inp untouched.
    tu.assert_result_equal(inp, ref_inp)


LAZY_ROWS = ["plain", "conj", "neg", "conj_neg"]


@pytest.mark.permute
@pytest.mark.parametrize("case", tu.selected_cases(LAZY_ROWS, quick=LAZY_ROWS[:1]))
def test_permute_preserves_lazy_bits(case):
    inp = tu.make_input(torch.complex64, (4, 6), ["-1", "1"])
    if case in ("conj", "conj_neg"):
        inp = inp.conj()
    if case in ("neg", "conj_neg"):
        inp = torch._neg_view(inp)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.permute(ref_inp, [1, 0])
    res_out = flag_gems.permute(inp, [1, 0])

    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.is_neg() == ref_out.is_neg()
    tu.assert_result_equal(res_out, ref_out)


BACKWARD_ROWS = [
    ((20, 320, 15), [2, 1, 0]),
    ((16, 7, 57, 32, 29), [4, 2, 0, 3, 1]),
]


@pytest.mark.permute
@pytest.mark.parametrize("case", tu.selected_cases(BACKWARD_ROWS, quick=[]))
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test_permute_backward(case, dtype):
    shape, dims = case
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_inp = tu.to_reference(inp.detach()).requires_grad_(True)
    upstream = tu.make_input(dtype, tuple(shape[d] for d in dims), ["-1", "1"])
    ref_upstream = tu.to_reference(upstream)

    (res_grad,) = torch.autograd.grad(
        flag_gems.permute(inp, dims), inp, grad_outputs=upstream
    )
    (ref_grad,) = torch.autograd.grad(
        torch.ops.aten.permute(ref_inp, dims), ref_inp, grad_outputs=ref_upstream
    )

    tu.assert_result_equal(res_grad, ref_grad)


@pytest.mark.permute
@pytest.mark.parametrize(
    "case", tu.selected_cases(tu.special_value_cases(SPECIAL_DTYPES), quick=[])
)
def test_permute_special_values(case):
    dtype, scenario = case
    # Reshape the shared 1-D payload so the permute performs a real dim swap.
    inp = tu.make_special_input(dtype, scenario).reshape(1, -1)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.permute(ref_inp, [1, 0])
    res_out = flag_gems.permute(inp, [1, 0])

    tu.assert_result_equal(res_out, ref_out)


# There is no "unsupported dtype" negative: the native overload accepted every
# dtype in SUPPORTED_DTYPES. Native dims errors are RuntimeException / IndexError;
# ValueError is accepted alongside because a candidate may validate dims itself
# (the checkout's own permute does). Only the candidate's exception is asserted.
INVALID_DIMS_ROWS = [
    pytest.param("duplicate", (3, 4), [0, 0], id="duplicate"),
    pytest.param("duplicate_3d", (2, 3, 4), [1, 1, 2], id="duplicate_3d"),
    pytest.param("out_of_range", (3, 4), [0, 2], id="out_of_range"),
    pytest.param("negative_out_of_range", (3, 4), [-3, 0], id="negative_out_of_range"),
    pytest.param("out_of_range_3d", (2, 3, 4), [0, 1, 3], id="out_of_range_3d"),
    pytest.param("wrong_arity", (3, 4), [0], id="wrong_arity"),
    pytest.param("empty_dims_non_scalar", (3, 4), [], id="empty_dims_non_scalar"),
]


@pytest.mark.permute
@pytest.mark.parametrize("name,shape,dims", INVALID_DIMS_ROWS)
def test_permute_invalid_dims_raise(name, shape, dims):
    del name
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    with pytest.raises((RuntimeError, IndexError, ValueError)):
        flag_gems.permute(inp, dims)


# Keep the omitted-argument form apart: it is a missing required argument, not
# an invalid dims value.
@pytest.mark.permute
def test_permute_missing_dims_raises():
    inp = tu.make_input(torch.float32, (3, 4), ["-1", "1"])
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.permute(inp)
