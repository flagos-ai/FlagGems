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

# aten::_lazy_clone(Tensor self) -> Tensor relocates the input's storage while
# keeping its dense geometry (shape, strides, storage offset, dtype, device) and
# materializing a lazy conjugate/negative bit into the values. It is a
# CompositeExplicitAutograd storage op, not a view; it has no parameters, no
# scalar operand and no broadcast operands, and its only structural boundary is
# the empty tensor.

# Static capability flags of the active backend, read at import time: collection
# and --list-cases allocate no tensor and call no operator. A dtype outside this
# map is a baseline type the backend always handles.
_DTYPE_CAPABILITY = {
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name))


_DTYPES = [
    dtype
    for dtype in tu.REQUIRED_DTYPES
    + [torch.float64, torch.bool, torch.complex64, torch.complex128]
    if _dtype_supported(dtype)
]


def _assert_clone_storage(res_out, inp, ref_out):
    """Storage semantics the shared value assertions do not cover."""
    assert res_out.device == inp.device
    assert tuple(res_out.stride()) == tuple(ref_out.stride())
    assert res_out.storage_offset() == ref_out.storage_offset()
    # The relocated storage keeps the input's dense geometry ...
    assert tuple(res_out.stride()) == tuple(inp.stride())
    assert res_out.storage_offset() == inp.storage_offset()
    # ... but the result is a fresh tensor, never a view of the input, and a
    # lazy conjugate/negative bit is materialized instead of being copied.
    assert not res_out._is_view()
    assert res_out._base is None
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.is_neg() == ref_out.is_neg()
    if inp.numel() != 0:
        # Empty inputs share the single empty storage on this backend.
        assert res_out.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()


@pytest.mark.lazy_clone
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test__lazy_clone(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._lazy_clone(ref_inp)
    res_out = flag_gems._lazy_clone(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_clone_storage(res_out, inp, ref_out)


# Layout/alias coverage: a fresh contiguous input does not exercise the copied
# stride/offset geometry, and the conj/neg rows exercise lazy-flag
# materialization. These rows are small and run at both levels.
_LAYOUT_ROWS = [
    ("contiguous", (256, 256), torch.float32),
    ("transposed", (256, 256), torch.float32),
    ("sliced", (256, 256), torch.float32),
    ("stepped", (256, 256), torch.float32),
    ("expanded", (256, 256), torch.int32),
    ("offset_row", (256, 256), torch.bfloat16),
    ("conj", (256, 256), torch.complex64),
    ("neg", (256, 256), torch.float32),
]
_LAYOUT_CASES = [row for row in _LAYOUT_ROWS if _dtype_supported(row[2])]


def _layout_input(layout, storage_shape, dtype):
    numel = 1
    for dim in storage_shape:
        numel *= dim
    values = torch.arange(numel + 4, dtype=torch.float32, device=flag_gems.device)
    values = values.to(torch.complex64) if dtype.is_complex else values.to(dtype)
    base = values[:numel].view(storage_shape)

    if layout == "contiguous":
        return base
    if layout == "transposed":
        return base.transpose(0, 1)
    if layout == "sliced":
        # Contiguous geometry with a nonzero storage offset.
        return values[4:].view(storage_shape)
    if layout == "stepped":
        return base[::2, ::2]
    if layout == "expanded":
        return base[:1].expand(storage_shape)
    if layout == "offset_row":
        return base[1:]
    if layout == "conj":
        return base.conj()
    if layout == "neg":
        # A real lazy negative bit (the construction tu.to_reference uses).
        return torch._neg_view(base)
    raise AssertionError(f"unknown layout {layout}")


@pytest.mark.lazy_clone
@pytest.mark.parametrize("layout,storage_shape,dtype", _LAYOUT_CASES)
def test__lazy_clone_preserves_layout(layout, storage_shape, dtype):
    inp = _layout_input(layout, storage_shape, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._lazy_clone(ref_inp)
    res_out = flag_gems._lazy_clone(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_clone_storage(res_out, inp, ref_out)


# Copy-on-write isolation in both directions: a storage relocation must not
# alias the source storage, which the value comparison alone cannot observe.
# (dtype, shape, clone_value, source_value)
_ISOLATION_ROWS = [
    (torch.float32, (128, 128), 2.0, 7.0),
    (torch.int64, (256,), 2, 7),
    (torch.bool, (64, 64), True, False),
    (torch.complex64, (64, 64), 2.0, 7.0),
    (torch.float8_e5m2, (32, 32), 2.0, 7.0),
]
_ISOLATION_CASES = [row for row in _ISOLATION_ROWS if _dtype_supported(row[0])]


@pytest.mark.lazy_clone
@pytest.mark.parametrize("dtype,shape,clone_value,source_value", _ISOLATION_CASES)
def test__lazy_clone_storage_isolation(dtype, shape, clone_value, source_value):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    inp_before = tu.to_reference(inp.detach())
    ref_inp = tu.to_reference(inp.detach())

    res_out = flag_gems._lazy_clone(inp)
    ref_out = torch.ops.aten._lazy_clone(ref_inp)

    # Forward result first, before either clone is overwritten.
    tu.assert_result_equal(res_out, ref_out)

    # Writing through the clone must not reach the source tensor.
    res_out.fill_(clone_value)
    ref_out.fill_(clone_value)
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, inp_before)

    # Writing the source must not reach the already produced clone.
    res_clone_state = tu.to_reference(res_out.detach())
    inp.fill_(source_value)
    ref_inp.fill_(source_value)
    tu.assert_result_equal(res_out, res_clone_state)
    tu.assert_result_equal(res_out, ref_out)


# Backward: a pure storage relocation passes a non-uniform upstream gradient
# through unchanged, so both the native comparison and the upstream comparison
# are exact. The original leaf is the differentiable input, not the clone.
# complex64/complex128 are exempt for this dimension: the native operator raises
# RuntimeError("_lazy_clone does not support automatic differentiation for
# outputs with complex dtype.") at the forward call as soon as the input
# requires grad, so no gradient form exists for complex inputs; their forward
# coverage is kept in the value, layout and isolation cases above.
_BACKWARD_ROWS = [
    (torch.float32, (1024, 1024)),
    (torch.float32, (20, 320, 15)),
    (torch.float16, (1024, 1024)),
    (torch.bfloat16, (20, 320, 15)),
    (torch.float64, (256, 256)),
    (torch.float8_e4m3fn, (128, 128)),
    (torch.float8_e5m2, (128, 128)),
]
_BACKWARD_CASES = tu.selected_cases(
    [row for row in _BACKWARD_ROWS if _dtype_supported(row[0])], quick=[]
)


@pytest.mark.lazy_clone
@pytest.mark.parametrize("dtype,shape", _BACKWARD_CASES)
def test__lazy_clone_backward(dtype, shape):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).detach().requires_grad_(True)
    ref_inp = tu.to_reference(inp.detach()).requires_grad_(True)
    upstream = tu.make_input(dtype, shape, ["-1", "1"])

    ref_out = torch.ops.aten._lazy_clone(ref_inp)
    res_out = flag_gems._lazy_clone(inp)
    tu.assert_result_equal(res_out, ref_out)

    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]
    ref_grad = torch.autograd.grad(
        ref_out, ref_inp, grad_outputs=tu.to_reference(upstream)
    )[0]

    tu.assert_result_equal(res_grad, ref_grad)
    tu.assert_result_equal(res_grad, upstream)


_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases([dtype for dtype in _DTYPES if dtype.is_floating_point]),
    quick=[],
)


@pytest.mark.lazy_clone
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test__lazy_clone_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario).reshape(1, 5)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._lazy_clone(ref_inp)
    res_out = flag_gems._lazy_clone(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_clone_storage(res_out, inp, ref_out)


# Empty inputs: zero-filled so no undefined allocation is ever read.
_EMPTY_CASES = [
    (torch.float32, (0,)),
    (torch.float32, (0, 3)),
    (torch.float32, (4, 0, 2)),
    (torch.int32, (0, 0)),
    (torch.int8, (0, 5)),
]


@pytest.mark.lazy_clone
@pytest.mark.parametrize("dtype,shape", _EMPTY_CASES)
def test__lazy_clone_empty(dtype, shape):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._lazy_clone(ref_inp)
    res_out = flag_gems._lazy_clone(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_clone_storage(res_out, inp, ref_out)


# Metadata labels only: the bad argument is built inside the test, so import,
# collection and --list-cases allocate no tensor.
_NON_TENSOR_ARGS = [
    pytest.param("not a tensor", id="str"),
    pytest.param(1.5, id="float"),
    pytest.param(None, id="none"),
    pytest.param("list", id="list"),
]


@pytest.mark.lazy_clone
@pytest.mark.parametrize("arg", _NON_TENSOR_ARGS)
def test__lazy_clone_rejects_non_tensor(arg):
    if arg == "list":
        arg = [torch.zeros(2, device=flag_gems.device)]
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._lazy_clone(arg)


@pytest.mark.lazy_clone
def test__lazy_clone_rejects_sparse_storage():
    inp = torch.zeros(4, 4, device=flag_gems.device).to_sparse()
    with pytest.raises((RuntimeError, NotImplementedError, TypeError)):
        flag_gems._lazy_clone(inp)
