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

"""Correctness tests for ``aten::_test_optional_floatlist``.

The native operator is a CPU-only ATen test helper: accelerator operands raise
"Could not run 'aten::_test_optional_floatlist' with arguments from the 'CUDA'
backend". Both the reference and the injected candidate therefore receive real
CPU operands, which is this operator's actual contract.

Call forms covered here:
  * ``op(values, None)`` returns the operand tensor itself (same object, dtype,
    shape and strides) for every storage dtype and rank;
  * ``op(values, addends)`` returns a fresh buffer with the host ``float[]``
    added to rank-1 ``float32`` values; the schema requires
    ``len(addends) >= values.numel()`` (extra entries are ignored) and the
    native operator has no derivative for this form;
  * ``.out`` writes through the caller's buffer and returns that buffer.

``addends`` has no schema default, so the argument is always required.
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu

# The identity call form preserves any storage dtype the operator accepts.
IDENTITY_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.float64,
    torch.bool,
    torch.complex64,
    torch.complex128,
]

# Every spec rank applies to the identity form; the empty tensor is an extra
# cheap boundary.
IDENTITY_SHAPES = tu.selected_shapes() + [(0,)]

# The list call form requires rank-1 values, so only 1-D shapes apply; the quick
# 3-dim shape is adapted to rank 1.
ADDENDS_SHAPES = [(0,), (1,), (19,), (256,)]

# (addend, add extra entries past numel, sequence type). Zero / positive /
# negative / fractional addends; entries past numel are ignored and a tuple
# is accepted exactly like a list.
ADDENDS_ROWS = tu.selected_cases(
    [
        (0.0, False, list),
        (1.0, False, list),
        (-1.0, False, list),
        (0.5, True, list),
        (2.5, True, tuple),
    ],
    quick=[
        (0.0, False, list),
        (1.0, False, list),
        (-1.0, False, list),
        (0.5, True, list),
        (2.5, True, tuple),
    ],
)

# Out-buffer layouts: a plain tensor, a view at a nonzero storage offset and a
# strided view, so writing past the view (or compacting the storage) is caught.
OUT_LAYOUTS = tu.selected_cases(
    ["contiguous", "offset", "strided"], quick=["contiguous", "offset", "strided"]
)
_OUT_SIZE = 4
_OUT_FILL = 7.0

# Positive special values stay default-only; derivative rejection stays in quick.
SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(IDENTITY_DTYPES), quick=[])
ADDENDS_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases([torch.float32]), quick=[]
)
BACKWARD_DTYPES = [
    dtype for dtype in IDENTITY_DTYPES if dtype.is_floating_point or dtype.is_complex
]


def _cpu_input(dtype, shape, value_range):
    """A native CPU operand: the kernel has no accelerator registration."""
    return tu.make_input(dtype, shape, value_range).cpu()


def _out_storage(layout):
    """Sentinel-filled storage for one out-buffer layout."""
    size = _OUT_SIZE + 4 if layout == "offset" else _OUT_SIZE
    if layout == "strided":
        size = 2 * _OUT_SIZE
    return torch.full((size,), _OUT_FILL, dtype=torch.float32)


def _out_view(layout, storage):
    """The (possibly offset, possibly strided) out buffer inside ``storage``."""
    if layout == "offset":
        return storage[2 : 2 + _OUT_SIZE]
    if layout == "strided":
        return storage[::2]
    return storage


@pytest.mark.test_optional_floatlist
@pytest.mark.parametrize("shape", IDENTITY_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", IDENTITY_DTYPES)
def test__test_optional_floatlist_identity(shape, value_range, dtype):
    inp = _cpu_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_optional_floatlist(ref_inp, None)
    res_out = flag_gems._test_optional_floatlist(inp, None)

    # The identity form hands the operand tensor back unchanged.
    assert res_out is inp
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.test_optional_floatlist
@pytest.mark.parametrize("shape", IDENTITY_SHAPES)
@pytest.mark.parametrize("use_keyword", [False, True])
def test__test_optional_floatlist_identity_call_forms(shape, use_keyword):
    # Both the positional ``None`` and the ``addends=None`` keyword form are
    # native-valid; the argument itself is required (no schema default).
    inp = _cpu_input(torch.float32, shape, ("-1", "1"))
    ref_inp = tu.to_reference(inp)

    if use_keyword:
        ref_out = torch.ops.aten._test_optional_floatlist(ref_inp, addends=None)
        res_out = flag_gems._test_optional_floatlist(inp, addends=None)
    else:
        ref_out = torch.ops.aten._test_optional_floatlist(ref_inp, None)
        res_out = flag_gems._test_optional_floatlist(inp, None)

    assert res_out is inp
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.test_optional_floatlist
@pytest.mark.parametrize("shape", ADDENDS_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("addend_case", ADDENDS_ROWS)
def test__test_optional_floatlist_addends(shape, value_range, addend_case):
    addend, extra_entries, sequence = addend_case
    inp = _cpu_input(torch.float32, shape, value_range)
    ref_inp = tu.to_reference(inp)
    snapshot = tu.to_reference(inp)
    length = inp.numel() + (3 if extra_entries else 0)
    addends = sequence([addend] * length)

    ref_out = torch.ops.aten._test_optional_floatlist(ref_inp, addends)
    res_out = flag_gems._test_optional_floatlist(inp, addends)

    # The list form returns a fresh buffer and leaves the operand untouched.
    assert res_out is not inp
    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_equal(inp, snapshot)


@pytest.mark.test_optional_floatlist
@pytest.mark.parametrize("addends", [[1.5, -0.5, 2.0, 0.0], (1.5, -0.5, 2.0, 0.0)])
def test__test_optional_floatlist_elementwise_addends(addends):
    inp = _cpu_input(torch.float32, (4,), ("-1", "1"))
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_optional_floatlist(ref_inp, addends)
    res_out = flag_gems._test_optional_floatlist(inp, addends)

    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    assert not torch._C._is_alias_of(res_out, inp)


@pytest.mark.test_optional_floatlist
@pytest.mark.parametrize("layout", OUT_LAYOUTS)
@pytest.mark.parametrize("with_addends", [False, True])
def test__test_optional_floatlist_out_buffer(layout, with_addends):
    inp = _cpu_input(torch.float32, (_OUT_SIZE,), ("-1", "1"))
    ref_inp = tu.to_reference(inp)
    addends = [1.5, -0.5, 2.0, 0.0] if with_addends else None

    ref_storage = _out_storage(layout)
    ref_buffer = _out_view(layout, ref_storage)
    res_storage = _out_storage(layout)
    res_buffer = _out_view(layout, res_storage)

    ref_out = torch.ops.aten._test_optional_floatlist.out(
        ref_inp, addends, out=ref_buffer
    )
    res_out = flag_gems._test_optional_floatlist(inp, addends, out=res_buffer)

    # Independently built equivalent buffer: positions outside the out view keep
    # the sentinel fill, so a write past the view cannot go unnoticed.
    expected_storage = _out_storage(layout)
    expected_view = _out_view(layout, expected_storage)
    expected_view.copy_(ref_out)

    assert res_out is res_buffer
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(res_storage, ref_storage)
    tu.assert_result_equal(res_storage, expected_storage)
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()


@pytest.mark.test_optional_floatlist
@pytest.mark.parametrize("with_addends", [False, True])
def test__test_optional_floatlist_out_resize(with_addends):
    # A zero-element out buffer is the size-changing form the native operator
    # still accepts: the buffer is reshaped to the output shape in place.
    inp = _cpu_input(torch.float32, (_OUT_SIZE,), ("-1", "1"))
    ref_inp = tu.to_reference(inp)
    addends = [1.5, -0.5, 2.0, 0.0] if with_addends else None
    ref_buffer = torch.empty(0, dtype=torch.float32)
    res_buffer = torch.empty(0, dtype=torch.float32)

    ref_out = torch.ops.aten._test_optional_floatlist.out(
        ref_inp, addends, out=ref_buffer
    )
    res_out = flag_gems._test_optional_floatlist(inp, addends, out=res_buffer)

    assert res_out is res_buffer
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.test_optional_floatlist
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test__test_optional_floatlist_special_values(dtype, scenario):
    # NaN / Inf / mixed payloads for every supported floating dtype, compared as
    # whole outputs; the shared helper widens FP8 only after the candidate ran
    # and keeps matching NaNs equal.
    inp = tu.make_special_input(dtype, scenario).cpu()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_optional_floatlist(ref_inp, None)
    res_out = flag_gems._test_optional_floatlist(inp, None)

    assert res_out is inp
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.test_optional_floatlist
@pytest.mark.parametrize("scenario", [case[1] for case in ADDENDS_SPECIAL_CASES])
def test__test_optional_floatlist_addends_special_values(scenario):
    inp = tu.make_special_input(torch.float32, scenario).cpu()
    ref_inp = tu.to_reference(inp)
    addends = [0.5] * inp.numel()

    ref_out = torch.ops.aten._test_optional_floatlist(ref_inp, addends)
    res_out = flag_gems._test_optional_floatlist(inp, addends)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.test_optional_floatlist
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test_test_optional_floatlist_rejects_backward(dtype):
    # A non-leaf view reaches the operator's missing derivative; a leaf identity
    # would make autograd stop at the output and never exercise this contract.
    base = tu.make_input(dtype, (16,), ["-1", "1"]).cpu().requires_grad_(True)
    ref_base = tu.to_reference(base.detach()).requires_grad_(True)
    inp, ref_inp = base[::2], ref_base[::2]
    upstream = tu.make_input(dtype, inp.shape, ["-1", "1"]).cpu()
    ref_out = torch.ops.aten._test_optional_floatlist(ref_inp, None)
    res_out = flag_gems._test_optional_floatlist(inp, None)
    tu.assert_result_equal(res_out, ref_out)
    with pytest.raises(
        RuntimeError,
        match="derivative for aten::_test_optional_floatlist is not implemented",
    ):
        torch.autograd.grad(ref_out, ref_base, grad_outputs=upstream)
    with pytest.raises(
        RuntimeError,
        match="derivative for aten::_test_optional_floatlist is not implemented",
    ):
        torch.autograd.grad(res_out, base, grad_outputs=upstream)


@pytest.mark.test_optional_floatlist
@pytest.mark.parametrize("shape", [(2, 3), ()])
def test__test_optional_floatlist_rejects_non_1d_values(shape):
    values = _cpu_input(torch.float32, shape, ("-1", "1"))
    addends = [0.0] * max(1, values.numel())
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_optional_floatlist(values, addends)


@pytest.mark.test_optional_floatlist
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.int64, torch.bool]
)
def test__test_optional_floatlist_rejects_non_float32_values(dtype):
    values = _cpu_input(dtype, (4,), ("-1", "1"))
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_optional_floatlist(values, [0.0] * values.numel())


@pytest.mark.test_optional_floatlist
@pytest.mark.parametrize("length", [0, 2])
def test__test_optional_floatlist_rejects_short_addends(length):
    # Fewer addends than elements is invalid (the empty tensor is the only
    # shapes for which an empty list is accepted, and it is covered above).
    values = _cpu_input(torch.float32, (4,), ("-1", "1"))
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_optional_floatlist(values, [0.0] * length)


@pytest.mark.test_optional_floatlist
@pytest.mark.parametrize("addends", [["a"] * 4, [None] * 4, "abcd"])
def test__test_optional_floatlist_rejects_non_numeric_addends(addends):
    values = _cpu_input(torch.float32, (4,), ("-1", "1"))
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_optional_floatlist(values, addends)


@pytest.mark.test_optional_floatlist
@pytest.mark.parametrize("values", [[1.0, 2.0], "abc", None])
def test__test_optional_floatlist_rejects_non_tensor_values(values):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_optional_floatlist(values, None)


@pytest.mark.test_optional_floatlist
@pytest.mark.parametrize("with_addends", [False, True])
def test__test_optional_floatlist_out_rejects_buffer_dtype(with_addends):
    if with_addends:
        # The accumulator is float32, so a float16 buffer is rejected.
        values = _cpu_input(torch.float32, (3,), ("-1", "1"))
        addends = [0.0] * 3
        buffer = torch.empty(3, dtype=torch.float16)
    else:
        # The identity path requires the buffer dtype to match the operand.
        values = _cpu_input(torch.float16, (3,), ("-1", "1"))
        addends = None
        buffer = torch.empty(3, dtype=torch.float32)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_optional_floatlist(values, addends, out=buffer)


@pytest.mark.test_optional_floatlist
def test__test_optional_floatlist_rejects_missing_addends():
    # Optional describes the value (None), not whether the argument can be omitted.
    values = torch.zeros(2, dtype=torch.float32, device="cpu")
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_optional_floatlist(values)
