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

# swapdims_ rewrites the sizes and strides of two dims in place and returns its
# own argument without moving data: comparisons are exact
# (tu.assert_result_equal) and operand broadcasting does not apply.
DTYPES = tu.REQUIRED_DTYPES + [torch.bool, torch.complex64]
DTYPES = [dtype for dtype in DTYPES if _DTYPE_FLAGS.get(dtype, True)]
SPECIAL_DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]
if utils.fp64_is_supported:
    DTYPES.append(torch.float64)
    SPECIAL_DTYPES.append(torch.float64)
SPECIAL_DTYPES = [dtype for dtype in SPECIAL_DTYPES if _DTYPE_FLAGS.get(dtype, True)]

BACKWARD_DTYPES = [
    dtype for dtype in DTYPES if dtype.is_floating_point or dtype.is_complex
]


def _dims_for(shape):
    # Swap the outermost dims; a rank-0/1 tensor only has dim 0, so the swap is
    # a no-op there.
    return 0, len(shape) - 1


def _swapped_shape(shape, dim0, dim1):
    order = list(range(len(shape)))
    order[dim0], order[dim1] = order[dim1], order[dim0]
    return tuple(shape[i] for i in order)


@pytest.mark.swapdims_
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", DTYPES)
def test_swapdims_(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    dim0, dim1 = _dims_for(shape)

    ptr_before = inp.untyped_storage().data_ptr()
    offset_before = inp.storage_offset()

    ref_version, version = ref_inp._version, inp._version
    ref_out = torch.ops.aten.swapdims_(ref_inp, dim0, dim1)
    res_out = flag_gems.swapdims_(inp, dim0, dim1)
    assert inp._version - version == ref_inp._version - ref_version

    # Same object, same storage, same offset: only sizes and strides change.
    assert res_out is inp
    assert inp.untyped_storage().data_ptr() == ptr_before
    assert res_out.storage_offset() == offset_before
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    tu.assert_result_equal(res_out, ref_out)


# Dim normalization is dtype independent, so one dtype keeps this matrix small.
_EDGE_CASES = [
    ((0, 3), (0, 1)),
    ((3, 0), (1, 0)),
    ((0, 3, 4), (0, 2)),
    ((2, 0, 4, 3), (0, 3)),
]

FULL_DIM_CASES = [
    ((20, 320, 15), (0, 2)),
    ((20, 320, 15), (2, 0)),
    ((20, 320, 15), (1, 0)),
    ((20, 320, 15), (-1, 0)),
    ((20, 320, 15), (-3, -1)),
    ((20, 320, 15), (2, 2)),
    ((20, 320, 15), (1, -2)),
    ((1024, 1024), (0, 1)),
    ((256,), (0, 0)),
    ((256,), (0, -1)),
    ((), (0, 0)),
    ((), (0, -1)),
] + _EDGE_CASES

# Quick keeps positive / negative / equal axis pairs and the empty, 1-D and
# scalar boundaries.
QUICK_DIM_CASES = [
    ((2, 19, 7), (0, 2)),
    ((2, 19, 7), (2, 0)),
    ((2, 19, 7), (1, 1)),
    ((2, 19, 7), (-1, 0)),
    ((2, 19, 7), (-3, -1)),
    ((0, 3), (0, 1)),
    ((3, 0), (1, 0)),
    ((256,), (0, -1)),
    ((), (0, 0)),
]

DIM_CASES = tu.selected_cases(FULL_DIM_CASES, quick=QUICK_DIM_CASES)


@pytest.mark.swapdims_
@pytest.mark.parametrize("shape,dims", DIM_CASES)
def test_swapdims__dim_forms(shape, dims):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.swapdims_(ref_inp, dims[0], dims[1])
    res_out = flag_gems.swapdims_(inp, dims[0], dims[1])

    assert res_out is inp
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    tu.assert_result_equal(res_out, ref_out)


def _layout_case(kind, dtype, value_range):
    """Build (base, view) plus the same view of an independent reference base.

    The reference is taken from the cloned base instead of cloning the view so
    that stride-0 (expanded), sliced and offset layouts keep their metadata on
    both sides.
    """
    base_shape = (12, 16, 20) if kind == "sliced" else (6, 8, 10)
    if kind == "expanded":
        base_shape = (1, 5, 1)
    base = tu.make_input(dtype, base_shape, value_range)
    ref_base = tu.to_reference(base)

    if kind == "plain":
        return base, base, ref_base, ref_base
    if kind == "transposed":
        return base, base.transpose(0, 1), ref_base, ref_base.transpose(0, 1)
    if kind == "sliced":
        return (base, base[2:10:2, ::3, ::4], ref_base, ref_base[2:10:2, ::3, ::4])
    if kind == "offset":
        return base, base[1:4], ref_base, ref_base[1:4]
    if kind == "expanded":
        return base, base.expand(4, 5, 3), ref_base, ref_base.expand(4, 5, 3)
    raise ValueError(f"unknown layout kind {kind}")


# The small layout rows are cheap and stay in both modes.
LAYOUT_KINDS = ["plain", "transposed", "sliced", "offset", "expanded"]


@pytest.mark.swapdims_
@pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
@pytest.mark.parametrize("kind", LAYOUT_KINDS)
def test_swapdims__layout(kind, dtype):
    base, inp, ref_base, ref_inp = _layout_case(kind, dtype, ["-1", "1"])
    dim0, dim1 = 0, 2

    ptr_before = inp.untyped_storage().data_ptr()
    offset_before = inp.storage_offset()

    ref_version, version = ref_inp._version, inp._version
    ref_out = torch.ops.aten.swapdims_(ref_inp, dim0, dim1)
    res_out = flag_gems.swapdims_(inp, dim0, dim1)
    assert inp._version - version == ref_inp._version - ref_version

    assert res_out is inp
    assert inp.untyped_storage().data_ptr() == ptr_before
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == offset_before == ref_out.storage_offset()
    tu.assert_result_equal(res_out, ref_out)
    # A metadata swap must not touch the shared allocation, so the owning
    # tensor still reads back its pre-call contents.
    tu.assert_result_equal(base, ref_base)


CONJ_KINDS = ["plain", "transposed", "sliced"]


@pytest.mark.swapdims_
@pytest.mark.parametrize("kind", CONJ_KINDS)
def test_swapdims__preserves_conjugate_view(kind):
    base = tu.make_input(torch.complex64, (6, 8, 10), ["-1", "1"])
    ref_base = tu.to_reference(base)
    if kind == "plain":
        inp, ref_inp = base, ref_base
    elif kind == "transposed":
        inp, ref_inp = base.transpose(0, 1), ref_base.transpose(0, 1)
    else:
        inp, ref_inp = base[1:5, ::2, ::3], ref_base[1:5, ::2, ::3]
    inp, ref_inp = inp.conj(), ref_inp.conj()
    dim0, dim1 = 0, inp.dim() - 1

    ptr_before = inp.untyped_storage().data_ptr()

    ref_version, version = ref_inp._version, inp._version
    ref_out = torch.ops.aten.swapdims_(ref_inp, dim0, dim1)
    res_out = flag_gems.swapdims_(inp, dim0, dim1)
    assert inp._version - version == ref_inp._version - ref_version

    assert res_out is inp
    assert inp.untyped_storage().data_ptr() == ptr_before
    assert res_out.is_conj()
    assert res_out.stride() == ref_out.stride()
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.swapdims_
def test_swapdims__result_shares_storage():
    # The returned handle is the caller's own tensor, so a write through it must
    # be visible in the original object.
    inp = tu.make_input(torch.float32, (20, 320, 15), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.swapdims_(ref_inp, 0, 2)
    res_out = flag_gems.swapdims_(inp, 0, 2)

    assert res_out is inp
    tu.assert_result_equal(res_out, ref_out)
    res_out.fill_(3.5)
    ref_out.fill_(3.5)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(SPECIAL_DTYPES), quick=[])


@pytest.mark.swapdims_
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test_swapdims__special_values(dtype, scenario):
    # The shared generator contributes only the nan case for e4m3fn, which
    # cannot represent infinity; nan/inf/mixed are covered for the other dtypes.
    inp = tu.make_special_input(dtype, scenario).reshape(1, -1)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.swapdims_(ref_inp, 0, 1)
    res_out = flag_gems.swapdims_(inp, 0, 1)

    assert res_out is inp
    assert res_out.stride() == ref_out.stride()
    tu.assert_result_equal(res_out, ref_out)


BACKWARD_CASES = tu.selected_cases(
    [((20, 320, 15), (0, 2)), ((16, 128, 64, 60), (0, 3))], quick=[]
)


@pytest.mark.swapdims_
@pytest.mark.parametrize("shape,dims", BACKWARD_CASES)
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test_swapdims__backward(shape, dims, dtype):
    dim0, dim1 = dims
    # An in-place call on a leaf that requires grad is rejected natively, so the
    # operator runs on a non-leaf copy while the gradient still reaches the
    # original leaf instead of the returned view itself.
    base = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_base = tu.to_reference(base.detach()).requires_grad_(True)
    inp = base.clone()
    ref_inp = ref_base.clone()
    upstream = tu.make_input(dtype, _swapped_shape(shape, dim0, dim1), ["-1", "1"])
    ref_upstream = tu.to_reference(upstream.detach())

    ref_version, version = ref_inp._version, inp._version
    ref_out = torch.ops.aten.swapdims_(ref_inp, dim0, dim1)
    res_out = flag_gems.swapdims_(inp, dim0, dim1)
    assert inp._version - version == ref_inp._version - ref_version
    assert res_out.grad_fn is not None
    tu.assert_result_equal(res_out, ref_out)

    res_grad = torch.autograd.grad(res_out, base, grad_outputs=upstream)[0]
    ref_grad = torch.autograd.grad(ref_out, ref_base, grad_outputs=ref_upstream)[0]
    assert res_grad.shape == ref_grad.shape
    tu.assert_result_equal(res_grad, ref_grad)


@pytest.mark.swapdims_
@pytest.mark.parametrize("dims", [(3, 0), (0, 3), (-4, 0), (0, -4)])
def test_swapdims__dim_out_of_range(dims):
    inp = tu.make_input(torch.float32, (2, 3, 4), ["-1", "1"])
    # Native reports IndexError; a RuntimeError rejection is accepted as well.
    with pytest.raises((IndexError, RuntimeError)):
        flag_gems.swapdims_(inp, dims[0], dims[1])


@pytest.mark.swapdims_
@pytest.mark.parametrize("bad_dim", [0.5, "0"])
def test_swapdims__invalid_dim_type(bad_dim):
    inp = tu.make_input(torch.float32, (2, 3, 4), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.swapdims_(inp, bad_dim, 1)


@pytest.mark.swapdims_
def test_swapdims__rejects_leaf_requiring_grad():
    inp = tu.make_input(torch.float32, (2, 3, 4), ["-1", "1"]).requires_grad_(True)
    with pytest.raises(RuntimeError):
        flag_gems.swapdims_(inp, 0, 1)
