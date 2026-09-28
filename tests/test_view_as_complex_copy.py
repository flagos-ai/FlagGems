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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# aten::view_as_complex_copy(real) -> complex materializes a complex tensor from
# real pairs. Native requirements: rank >= 1, a last dimension of size 2 with
# stride 1, every other stride divisible by 2, and an even element storage offset.
_REAL_TO_COMPLEX = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}

# Only the three real float types have a native kernel, so every other dtype is a
# negative case. Rows are gated by the shared static capability flags read from the
# runtime device (tests/accuracy_utils.py), so collecting this file allocates no
# tensor on a device that cannot allocate the dtype; bool/int8/uint8/int32 have no
# device capability question.
_VAC_UNSUPPORTED_DTYPES = [
    dtype
    for dtype, allocatable in (
        (torch.bfloat16, utils.bf16_is_supported),
        (torch.float8_e4m3fn, utils.fp8_is_supported),
        (torch.float8_e5m2, utils.fp8_is_supported),
        (torch.bool, True),
        (torch.int8, True),
        (torch.uint8, True),
        (torch.int32, True),
        (torch.int64, utils.int64_is_supported),
        (torch.complex64, True),
    )
    if allocatable
]

_VAC_DTYPES = [torch.float16, torch.float32] + (
    [torch.float64] if utils.fp64_is_supported else []
)

# Every valid input ends in a size-2 dimension, so each spec shape gets that
# trailing dimension. The rank-0 spec shape is not callable at all ("Input tensor
# must have one or more dimensions"); its (2,) form stands in as the smallest
# valid input and covers the 0-dim output case, while the negative shape test
# keeps rejecting a genuine rank-0 tensor.
_VAC_SHAPES = [shape + (2,) for shape in tu.selected_shapes()]

# Empty tensors keep the trailing size-2 dimension, so they are valid inputs.
_VAC_EMPTY_SHAPES = tu.selected_cases([(0, 2), (3, 0, 2)], quick=[])

# Non-contiguous inputs are valid while the last dimension keeps size 2, stride 1,
# an even offset and even other strides, so each row slices a size-2 window out of
# a wider base tensor instead of building a contiguous copy.
_VAC_STRIDED_CASES = tu.selected_cases([((8, 16), 2), ((4, 2, 8), 4)], quick=[])

# Native-valid layout variants that no broadcast rule describes: an expanded
# (zero-stride) dimension and an outer transpose whose last dimension keeps stride
# 1 are both accepted inputs for this unary operator.
_VAC_LAYOUT_CASES = tu.selected_cases(
    [
        ("expand", (1, 3, 2), (4, 3, 2)),
        ("expand_flat", (1, 2), (5, 2)),
        ("transpose", (3, 4, 8), 2),
    ],
    quick=[],
)

_VAC_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_VAC_DTYPES), quick=[])

_VAC_ALIAS_SHAPES = tu.selected_cases([(64, 2), (8, 16, 2)], quick=[])

# (parent shape, index) pairs whose out buffer is a non-contiguous or offset view
# of a sentinel-filled parent: stride 2 with offset 1, stride 1 with offset 1, and
# a column view with stride 5 and offset 1.
_VAC_OUT_VIEW_CASES = tu.selected_cases(
    [
        ((16,), (slice(1, 16, 2),)),
        ((10,), (slice(1, 9),)),
        ((8, 5), (slice(None), 1)),
    ],
    quick=[],
)

# Rows where the input and the out buffer are both non-contiguous: the input is a
# size-2 window (through _layout_view) with last stride 1, even outer strides and an
# even offset, and the out buffer is an offset/step view of a sentinel-filled
# parent. Row two is base(3,4,8).transpose(0,1)[..., 2:4] -> (4,3,2) written into
# parent(4,6)[:, 1:6:2] -> (4,3).
_VAC_OUT_STRIDED_INPUT_CASES = tu.selected_cases(
    [
        ("window", (8, 16), 4, (16,), (slice(1, 16, 2),)),
        ("transpose", (3, 4, 8), 2, (4, 6), (slice(None), slice(1, 6, 2))),
    ],
    quick=[],
)

_VAC_INVALID_SHAPES = [
    (6, 3),  # last dimension of size 3
    (4, 1),  # last dimension of size 1
    (),  # rank 0
]

# Odd outer strides are rejected even when the last stride is 1 and the offset is
# even: "Tensor must have a stride divisible by 2 for all but last dimension".
_VAC_ODD_OUTER_STRIDES = [3, 5]

_VAC_WRONG_OUT_DTYPES = [torch.float32] + (
    [torch.complex128] if utils.fp64_is_supported else []
)

# Native reports every invalid input with RuntimeError, so the negative tests match
# that contract exactly instead of also accepting AssertionError, which unrelated
# internal checks would raise.
# Backward coverage is impossible: on the target device a requires_grad input raises
# RuntimeError("view_as_complex_copy does not support automatic differentiation for
# outputs with complex dtype.").


def _layout_view(tensor, kind, spec):
    """Apply one native-valid non-contiguous layout (candidate or reference)."""
    if kind.startswith("expand"):
        return tensor.expand(spec)
    if kind == "window":
        return tensor[..., spec : spec + 2]
    return tensor.transpose(0, 1)[..., spec : spec + 2]


def _ramp_input(dtype, shape):
    """Base tensor with strictly increasing values.

    Adjacent entries of a size-2 window therefore always differ, so a result whose
    real and imaginary parts were mixed up could not compare equal.
    """
    return (
        torch.linspace(-1.0, 1.0, math.prod(shape), device=flag_gems.device)
        .to(dtype)
        .reshape(shape)
    )


def _assert_copy_result(res_out, ref_out, inp, ref_inp):
    """Compare a copy result and assert the two properties the copy contract adds.

    tu.assert_result_equal covers dtype, shape and exact values. The aliasing
    aten::view_as_complex shares the input storage while aten::view_as_complex_copy
    must allocate fresh storage, and the candidate must leave its input untouched,
    so both are checked against the independent reference input. Empty tensors
    report a null data pointer on both sides and are skipped.
    """
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    assert res_out.device == inp.device
    if inp.numel() > 0:
        assert res_out.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("shape", _VAC_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VAC_DTYPES)
def test_view_as_complex_copy_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.view_as_complex_copy(ref_inp)
    res_out = flag_gems.view_as_complex_copy(inp)

    _assert_copy_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("shape", _VAC_SHAPES)
@pytest.mark.parametrize("dtype", _VAC_DTYPES)
def test_view_as_complex_copy_out(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    # Sentinel fill: any element the kernel fails to write stays at -7.5+3.25j.
    out = torch.full(
        shape[:-1],
        complex(-7.5, 3.25),
        dtype=_REAL_TO_COMPLEX[dtype],
        device=flag_gems.device,
    )
    ref_out = tu.to_reference(out)

    torch.ops.aten.view_as_complex_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.view_as_complex_copy(inp, out=out)

    assert res_ret is out
    tu.assert_result_equal(out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("parent_shape, index", _VAC_OUT_VIEW_CASES)
@pytest.mark.parametrize("dtype", _VAC_DTYPES)
def test_view_as_complex_copy_out_into_offset_view(parent_shape, index, dtype):
    inp = tu.make_input(dtype, (8, 2), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    # Sentinels fill the whole parent, so comparing the parent also verifies that
    # the kernel wrote only the selected view and left the padding alone.
    parent = torch.full(
        parent_shape,
        complex(-7.5, 3.25),
        dtype=_REAL_TO_COMPLEX[dtype],
        device=flag_gems.device,
    )
    ref_parent = tu.to_reference(parent)
    out = parent[index]
    ref_out = ref_parent[index]

    torch.ops.aten.view_as_complex_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.view_as_complex_copy(inp, out=out)

    assert res_ret is out
    tu.assert_result_equal(parent, ref_parent)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize(
    "kind, base_shape, offset, parent_shape, index", _VAC_OUT_STRIDED_INPUT_CASES
)
@pytest.mark.parametrize("dtype", _VAC_DTYPES)
def test_view_as_complex_copy_out_with_strided_input(
    kind, base_shape, offset, parent_shape, index, dtype
):
    base = _ramp_input(dtype, base_shape)
    ref_base = tu.to_reference(base)
    inp = _layout_view(base, kind, offset)
    ref_inp = _layout_view(ref_base, kind, offset)
    parent = torch.full(
        parent_shape,
        complex(-7.5, 3.25),
        dtype=_REAL_TO_COMPLEX[dtype],
        device=flag_gems.device,
    )
    ref_parent = tu.to_reference(parent)
    out = parent[index]
    ref_out = ref_parent[index]

    torch.ops.aten.view_as_complex_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.view_as_complex_copy(inp, out=out)

    assert res_ret is out
    tu.assert_result_equal(parent, ref_parent)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("shape", _VAC_EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", _VAC_DTYPES)
def test_view_as_complex_copy_empty(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.view_as_complex_copy(ref_inp)
    res_out = flag_gems.view_as_complex_copy(inp)

    _assert_copy_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("base_shape, offset", _VAC_STRIDED_CASES)
@pytest.mark.parametrize("dtype", _VAC_DTYPES)
def test_view_as_complex_copy_strided_input(base_shape, offset, dtype):
    base = tu.make_input(dtype, base_shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base[..., offset : offset + 2]
    ref_inp = ref_base[..., offset : offset + 2]

    ref_out = torch.ops.aten.view_as_complex_copy(ref_inp)
    res_out = flag_gems.view_as_complex_copy(inp)

    _assert_copy_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("kind, base_shape, spec", _VAC_LAYOUT_CASES)
@pytest.mark.parametrize("dtype", _VAC_DTYPES)
def test_view_as_complex_copy_layout(kind, base_shape, spec, dtype):
    base = tu.make_input(dtype, base_shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = _layout_view(base, kind, spec)
    ref_inp = _layout_view(ref_base, kind, spec)

    ref_out = torch.ops.aten.view_as_complex_copy(ref_inp)
    res_out = flag_gems.view_as_complex_copy(inp)

    _assert_copy_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("dtype, scenario", _VAC_SPECIAL_CASES)
def test_view_as_complex_copy_special_values(dtype, scenario):
    payload = tu.make_special_input(dtype, scenario)
    # Both lanes carry the special payload, so nan/inf has to survive in each
    # component of the complex result.
    inp = torch.stack((payload, payload.flip(0)), dim=-1)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.view_as_complex_copy(ref_inp)
    res_out = flag_gems.view_as_complex_copy(inp)

    _assert_copy_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("shape", _VAC_ALIAS_SHAPES)
@pytest.mark.parametrize("dtype", _VAC_DTYPES)
def test_view_as_complex_copy_does_not_alias_input(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.view_as_complex_copy(ref_inp)
    res_out = flag_gems.view_as_complex_copy(inp)

    _assert_copy_result(res_out, ref_out, inp, ref_inp)

    # Overwriting the result afterwards must not change the input: a real copy
    # owns its storage, unlike the aliasing aten::view_as_complex view.
    res_out.fill_(0)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("dtype", _VAC_UNSUPPORTED_DTYPES)
def test_view_as_complex_copy_rejects_unsupported_dtype(dtype):
    inp = torch.zeros((4, 2), dtype=dtype, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.view_as_complex_copy(inp)


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("shape", _VAC_INVALID_SHAPES)
def test_view_as_complex_copy_rejects_invalid_shape(shape):
    inp = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.view_as_complex_copy(inp)


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("dtype", _VAC_DTYPES)
def test_view_as_complex_copy_rejects_odd_storage_offset(dtype):
    base = torch.zeros((4, 8), dtype=dtype, device=flag_gems.device)
    inp = base[:, 1:3]  # storage offset 1

    with pytest.raises(RuntimeError):
        flag_gems.view_as_complex_copy(inp)


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("dtype", _VAC_DTYPES)
def test_view_as_complex_copy_rejects_non_unit_last_stride(dtype):
    inp = torch.zeros((2, 8), dtype=dtype, device=flag_gems.device).t()

    with pytest.raises(RuntimeError):
        flag_gems.view_as_complex_copy(inp)


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("stride", _VAC_ODD_OUTER_STRIDES)
def test_view_as_complex_copy_rejects_odd_outer_stride(stride):
    base = torch.zeros(4 * stride, dtype=torch.float32, device=flag_gems.device)
    inp = base.as_strided((2, 2), (stride, 1), 0)

    with pytest.raises(RuntimeError):
        flag_gems.view_as_complex_copy(inp)


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("out_dtype", _VAC_WRONG_OUT_DTYPES)
def test_view_as_complex_copy_out_rejects_wrong_dtype(out_dtype):
    inp = torch.zeros((8, 2), dtype=torch.float32, device=flag_gems.device)
    out = torch.zeros((8,), dtype=out_dtype, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.view_as_complex_copy(inp, out=out)
