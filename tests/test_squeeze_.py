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

_DTYPE_FLAGS = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.float64: utils.fp64_is_supported,
    torch.complex128: utils.fp64_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
}

# squeeze_ is an in-place metadata operator: it never moves element data, it
# rebinds the same Tensor object to the shape without unit dimensions and bumps
# the version counter. Dtype is not part of its contract, so every dtype the
# native kernel accepts is covered. A row is (shape, extra), where `extra` is
# the positional argument tuple after the tensor: () selects the default
# overload, (int,) the dim overload and ([int, ...],) the dims overload of the
# same public entry point. There is no broadcast dimension: the operator has a
# single tensor operand and never combines values.
_DTYPES = list(tu.REQUIRED_DTYPES) + [torch.bool, torch.complex64]
if utils.fp64_is_supported:
    _DTYPES.append(torch.float64)
_DTYPES = [dtype for dtype in _DTYPES if _DTYPE_FLAGS.get(dtype, True)]

# Special-value scenarios only exist for floating dtypes; the shared generator
# already restricts e4m3fn to nan-only and e5m2 to nan/inf/mixed.
_SPECIAL_DTYPES = [dtype for dtype in _DTYPES if dtype.is_floating_point]

# Layout semantics are dtype independent, so the view tests use two cheap
# dtypes; only complex can carry a lazy conjugate bit.
_LAYOUT_DTYPES = [torch.float32, torch.int64]
_LAYOUT_DTYPES = [dtype for dtype in _LAYOUT_DTYPES if _DTYPE_FLAGS.get(dtype, True)]

_BACKWARD_DTYPES = [
    dtype for dtype in _DTYPES if dtype.is_floating_point or dtype.is_complex
]

_SQUEEZE_ROWS = [
    # default overload: every unit dimension is removed in one call
    ((1,), ()),
    ((1, 256, 1), ()),
    ((1, 1024, 1024), ()),
    ((1024, 1, 1024, 1), ()),
    ((1, 20, 320, 1, 15), ()),
    ((1, 16, 128, 1, 64), ()),
    ((1, 16, 7, 57, 32), ()),
    ((1, 1, 1, 1, 1), ()),
    ((), ()),
    ((1, 0), ()),
    # dim overload: only the addressed dimension is considered
    ((1, 20, 320, 1, 15), (0,)),
    ((1, 20, 320, 1, 15), (3,)),
    ((1, 20, 320, 1, 15), (-2,)),
    ((1, 20, 320, 1, 15), (1,)),
    ((2, 19, 7), (0,)),
    ((1, 2, 19, 7), (0,)),
    # dims overload
    ((1, 1, 0), ([0, 1],)),
    ((1, 20, 320, 1, 15), ([0, 3],)),
    ((1, 20, 320, 1, 15), ([0],)),
    ((1, 20, 320, 1, 15), ([],)),
    ((1, 20, 320, 1, 15), ([4, 0, 3],)),
    ((1, 20, 320, 1, 15), ((-5, -2),)),
    ((2, 19, 7), ([0, 1],)),
]

# Quick keeps every argument form and every small semantic boundary: default,
# dim (positive/negative/no-op), dims (list/empty/unsorted/no-op), 0-dim and
# empty tensors.
_QUICK_SQUEEZE_ROWS = [
    ((1,), ()),
    ((1, 2, 19, 1, 7), (0,)),
    ((1, 2, 19, 1, 7), ([-2],)),
    ((1, 2, 19, 1, 7), ([0, 3],)),
    ((1, 2, 19, 1, 7), ([3, 0],)),
    ((1, 2, 19, 1, 7), ((3, 0),)),
    ((1, 2, 19, 1, 7), ([],)),
    ((2, 1, 19, 7), (-3,)),
    ((2, 19, 7), ()),
    ((2, 19, 7), (0,)),
    ((2, 19, 7), ([0, 1],)),
    ((1, 1, 1), ()),
    ((), ()),
    ((1, 1, 0), ()),
    ((1, 1, 0), ([0, 1],)),
]

# The op refuses a grad-requiring leaf, so these shapes are differentiated
# through their non-leaf clone (see test_squeeze__backward).
_BACKWARD_ROWS = [
    ((1, 5, 1), ()),
    ((1, 4, 3), (0,)),
    ((1, 4, 1), ([0, 2],)),
    ((2, 3), (1,)),
]


def _snapshot(inp):
    return (
        inp.untyped_storage().data_ptr(),
        inp.data_ptr(),
        inp.storage_offset(),
        inp._version,
    )


def _assert_squeeze_inplace(res, inp, ref_out, ref_inp, before, ref_version_delta):
    # The return value is the input object itself, only metadata changes, and no
    # new storage is allocated. The candidate must also bump the version counter
    # exactly like the native in-place op.
    storage_ptr, data_ptr, offset, version = before
    assert res is inp
    assert inp.shape == ref_inp.shape
    assert inp.stride() == ref_inp.stride()
    assert inp.storage_offset() == offset
    assert inp.untyped_storage().data_ptr() == storage_ptr
    assert inp.data_ptr() == data_ptr
    assert inp._version - version == ref_version_delta
    tu.assert_result_equal(res, ref_out)


@pytest.mark.squeeze_
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_squeeze__value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_version = ref_inp._version
    ref_out = torch.ops.aten.squeeze_(ref_inp)
    ref_version_delta = ref_inp._version - ref_version

    before = _snapshot(inp)
    res_out = flag_gems.squeeze_(inp)

    _assert_squeeze_inplace(res_out, inp, ref_out, ref_inp, before, ref_version_delta)


@pytest.mark.squeeze_
@pytest.mark.parametrize(
    "shape, extra", tu.selected_cases(_SQUEEZE_ROWS, quick=_QUICK_SQUEEZE_ROWS)
)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_squeeze__argument_forms(shape, extra, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_version = ref_inp._version
    ref_out = torch.ops.aten.squeeze_(ref_inp, *extra)
    ref_version_delta = ref_inp._version - ref_version

    before = _snapshot(inp)
    res_out = flag_gems.squeeze_(inp, *extra)

    _assert_squeeze_inplace(res_out, inp, ref_out, ref_inp, before, ref_version_delta)


@pytest.mark.squeeze_
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test_squeeze__non_contiguous_input(dtype):
    # A transposed view keeps its original strides; squeeze must reuse them
    # instead of compacting the result into a contiguous tensor.
    base = tu.make_input(dtype, (3, 1, 5), ["-1", "1"])
    inp = base.transpose(0, 2)
    ref_inp = tu.to_reference(inp)
    ref_version = ref_inp._version
    ref_out = torch.ops.aten.squeeze_(ref_inp)
    ref_version_delta = ref_inp._version - ref_version

    before = _snapshot(inp)
    res_out = flag_gems.squeeze_(inp)

    _assert_squeeze_inplace(res_out, inp, ref_out, ref_inp, before, ref_version_delta)
    assert res_out.shape == (5, 3)
    assert not res_out.is_contiguous()


@pytest.mark.squeeze_
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test_squeeze__expanded_stride_zero_input(dtype):
    # An expanded dimension has stride 0 and overlaps its single stored
    # element; the squeezed view must keep that stride.
    row = tu.make_input(dtype, (1,), ["-1", "1"])
    inp = row.expand(5, 1)
    ref_inp = tu.to_reference(inp)
    ref_version = ref_inp._version
    ref_out = torch.ops.aten.squeeze_(ref_inp)
    ref_version_delta = ref_inp._version - ref_version

    before = _snapshot(inp)
    res_out = flag_gems.squeeze_(inp)

    _assert_squeeze_inplace(res_out, inp, ref_out, ref_inp, before, ref_version_delta)
    assert res_out.shape == (5,)
    assert res_out.stride() == (0,)


@pytest.mark.squeeze_
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test_squeeze__storage_offset_view(dtype):
    # Slicing leaves a nonzero storage offset that squeeze must preserve.
    base = tu.make_input(dtype, (4, 12), ["-1", "1"])
    inp = base[:, 3:9].view(4, 1, 6)
    ref_inp = tu.to_reference(inp)
    ref_version = ref_inp._version
    ref_out = torch.ops.aten.squeeze_(ref_inp)
    ref_version_delta = ref_inp._version - ref_version

    before = _snapshot(inp)
    res_out = flag_gems.squeeze_(inp)

    _assert_squeeze_inplace(res_out, inp, ref_out, ref_inp, before, ref_version_delta)
    assert res_out.shape == (4, 6)
    assert res_out.storage_offset() == 3


@pytest.mark.squeeze_
def test_squeeze__lazy_conjugate_input():
    # Only complex tensors carry a lazy conjugate bit, and it must survive the
    # in-place reshape of the same storage.
    inp = tu.make_input(torch.complex64, (1, 3, 1), ["-1", "1"]).conj()
    ref_inp = tu.to_reference(inp)
    ref_version = ref_inp._version
    ref_out = torch.ops.aten.squeeze_(ref_inp)
    ref_version_delta = ref_inp._version - ref_version

    before = _snapshot(inp)
    res_out = flag_gems.squeeze_(inp)

    _assert_squeeze_inplace(res_out, inp, ref_out, ref_inp, before, ref_version_delta)
    assert res_out.shape == (3,)
    assert res_out.is_conj()


@pytest.mark.squeeze_
@pytest.mark.parametrize(
    "dtype, scenario",
    tu.selected_cases(tu.special_value_cases(_SPECIAL_DTYPES), quick=[]),
)
def test_squeeze__special_values(dtype, scenario):
    # The shared payload is 1-D; add unit dimensions so the case squeezes real
    # dimensions instead of only exercising the no-op path.
    inp = tu.make_special_input(dtype, scenario).reshape(1, 5, 1)
    ref_inp = tu.to_reference(inp)

    ref_version = ref_inp._version
    ref_out = torch.ops.aten.squeeze_(ref_inp)
    ref_version_delta = ref_inp._version - ref_version

    before = _snapshot(inp)
    res_out = flag_gems.squeeze_(inp)

    _assert_squeeze_inplace(res_out, inp, ref_out, ref_inp, before, ref_version_delta)
    assert res_out.shape == (5,)


# Backward is a default-only dimension: excluded from quick.
@pytest.mark.squeeze_
@pytest.mark.parametrize("shape, extra", tu.selected_cases(_BACKWARD_ROWS, quick=[]))
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
def test_squeeze__backward(shape, extra, dtype):
    # The native op rejects a grad-requiring leaf, so the operator is applied in
    # place to a non-leaf clone and differentiated through the original leaf.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    leaf = inp.detach().clone().requires_grad_(True)
    ref_leaf = ref_inp.detach().clone().requires_grad_(True)

    res_out = flag_gems.squeeze_(leaf.clone(), *extra)
    ref_out = torch.ops.aten.squeeze_(ref_leaf.clone(), *extra)

    # The forward result is checked before the gradient is taken.
    tu.assert_result_equal(res_out, ref_out)

    # Squeeze only relabels dimensions, so a non-uniform upstream is relayed
    # exactly for every supported floating and complex dtype.
    upstream = tu.make_input(dtype, tuple(ref_out.shape), ["-1", "1"])
    ref_upstream = tu.to_reference(upstream)

    (res_grad,) = torch.autograd.grad(res_out, leaf, grad_outputs=upstream)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_leaf, grad_outputs=ref_upstream)

    assert res_grad.shape == ref_grad.shape
    tu.assert_result_equal(res_grad, ref_grad)


# Negative cases assert only the candidate's rejection; the native call is a
# separate capability probe and is not repeated here.
@pytest.mark.squeeze_
@pytest.mark.parametrize("dim", [5, -6])
def test_squeeze__rejects_out_of_range_dim(dim):
    inp = tu.make_input(torch.float32, (1, 2, 19, 7), ["-1", "1"])
    with pytest.raises((IndexError, RuntimeError)):
        flag_gems.squeeze_(inp, dim)


@pytest.mark.squeeze_
@pytest.mark.parametrize("dims", [[9], [0, 9]])
def test_squeeze__rejects_out_of_range_dims(dims):
    inp = tu.make_input(torch.float32, (1, 2, 19, 7), ["-1", "1"])
    with pytest.raises((IndexError, RuntimeError)):
        flag_gems.squeeze_(inp, dims)


@pytest.mark.squeeze_
@pytest.mark.parametrize("dims", [[0, 0], [0, 0, 1]])
def test_squeeze__rejects_duplicate_dims(dims):
    inp = tu.make_input(torch.float32, (1, 2, 19, 7), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.squeeze_(inp, dims)


@pytest.mark.squeeze_
def test_squeeze__rejects_dimname():
    # Named tensors are NYI in this build, so the dimname overload has no valid
    # positive workload; the string form must still be rejected.
    inp = tu.make_input(torch.float32, (1, 2, 19, 7), ["-1", "1"])
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.squeeze_(inp, "a")


@pytest.mark.squeeze_
def test_squeeze__rejects_non_tensor():
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.squeeze_(3.14)


@pytest.mark.squeeze_
def test_squeeze__rejects_inplace_on_grad_leaf():
    # Autograd forbids in-place mutation of a leaf; rebinding metadata must not
    # bypass that check.
    inp = tu.make_input(torch.float32, (1, 2, 19, 7), ["-1", "1"]).requires_grad_(True)
    with pytest.raises(RuntimeError):
        flag_gems.squeeze_(inp)
