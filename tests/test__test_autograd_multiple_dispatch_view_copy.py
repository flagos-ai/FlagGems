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

"""Correctness tests for ``aten::_test_autograd_multiple_dispatch_view_copy``.

``flag_gems._test_autograd_multiple_dispatch_view_copy`` flattens ``self`` into a
fresh rank-1 contiguous copy: the result never aliases the input, a lazy
negation/conjugate bit on the source is materialized, and only a layout that
still covers one contiguous subspace can be flattened. A transposed, permuted,
expanded or gap-sliced operand therefore raises the native "view size is not
compatible" error and is covered as a negative case. The schema takes a single
tensor operand and no scalar, so the spec's broadcast, scalar-operand and
parameter-sweep dimensions do not apply.
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu

# Static capability flags, read while this module is imported: no tensor is
# allocated and no operator is called at collection time. A dtype whose flag is
# off is dropped from the grid during collection instead of being skipped later.
_DTYPE_CAPABILITY = {
    torch.float64: "support_fp64",
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _supported(dtypes):
    return [
        dtype
        for dtype in dict.fromkeys(dtypes)
        if dtype not in _DTYPE_CAPABILITY
        or getattr(flag_gems.runtime.device, _DTYPE_CAPABILITY[dtype])
    ]


# The nine required dtypes plus native-valid copies beyond the required set.
_COPY_DTYPES = _supported(
    list(tu.REQUIRED_DTYPES) + [torch.int16, torch.bool, torch.complex64, torch.float64]
)
# The native op raises "does not support automatic differentiation for outputs
# with complex dtype", so only the real floating types are graded.
# The device-dispatched backward adds one on CUDA and needs numeric addition:
# the native FP8 path raises "ufunc_add_CUDA not implemented". Forward FP8 copy
# coverage remains complete; a CPU gradient would be a different oracle.
_BACKWARD_DTYPES = _supported(
    [torch.float32, torch.float16, torch.bfloat16, torch.float64]
)
_LAYOUT_DTYPES = _supported([torch.float32, torch.int32])
_OUT_DTYPES = _supported(
    [
        torch.float32,
        torch.float16,
        torch.int32,
        torch.bool,
        torch.float8_e5m2,
        torch.complex64,
    ]
)

_VALUE_RANGE = ["-1", "1"]

_LAYOUT_BASE = (8, 16, 12)
_LAYOUT_BUILDERS = {
    "contiguous": lambda t: t,
    "last_dim_stride": lambda t: t[..., ::2],
    "narrow_offset": lambda t: t[1:3],
    "unsqueeze": lambda t: t.unsqueeze(0),
    "flat_offset": lambda t: t.reshape(-1)[3:],
}

# Every base/layout pair keeps one contiguous subspace; the 2-D row repeats the
# strided layout at a different rank.
_STRIDED_ROWS = tu.selected_cases(
    [
        (_LAYOUT_BASE, "last_dim_stride"),
        (_LAYOUT_BASE, "narrow_offset"),
        (_LAYOUT_BASE, "unsqueeze"),
        (_LAYOUT_BASE, "flat_offset"),
        ((1024, 1024), "last_dim_stride"),
    ],
    quick=[
        (_LAYOUT_BASE, "last_dim_stride"),
        (_LAYOUT_BASE, "narrow_offset"),
        (_LAYOUT_BASE, "unsqueeze"),
        (_LAYOUT_BASE, "flat_offset"),
    ],
)

# Lazy-bit sources: the strided and offset storage cases add the layouts the
# plain contiguous operand cannot reach.
_LAZY_ROWS = [
    ("neg", torch.float32, "contiguous"),
    ("neg", torch.float32, "last_dim_stride"),
    ("neg", torch.float32, "narrow_offset"),
    ("conj", torch.complex64, "contiguous"),
    ("conj", torch.complex64, "last_dim_stride"),
    ("conj", torch.complex64, "narrow_offset"),
]

# Zero-element operands: neither side holds storage, so the fresh-buffer pointer
# check is skipped for them (see _assert_materialized_copy).
_EMPTY_ROWS = [(0,), (2, 0, 3), (0, 3, 4)]

# Positive special values and backward are default-mode only.
_BACKWARD_ROWS = tu.selected_cases([(), (2, 3), (2, 3, 4)], quick=[])
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_COPY_DTYPES), quick=[])

_OUT_KINDS = ["exact", "offset", "strided", "empty"]

# The operator cannot flatten a layout whose elements span two contiguous
# subspaces; the last entry keeps a nonzero gap after the last dimension.
_UNFLATTENABLE_BUILDERS = {
    "transposed": lambda t: t.transpose(0, 1),
    "permuted": lambda t: t.permute(2, 0, 1),
    "expanded": lambda t: t[:1].expand(8, 16, 12),
    "gap_sliced": lambda t: t[..., 1:],
}

_INVALID_ARGUMENT_ROWS = [
    ("non_tensor_operand", (RuntimeError, TypeError)),
    ("missing_operand", RuntimeError),
]

_INVALID_OUT_ROWS = [
    ("dtype_mismatch", RuntimeError),
    ("non_tensor", (RuntimeError, TypeError)),
]


def _assert_materialized_copy(res_out, inp):
    """The result is a fresh flat buffer, not a view of the input."""
    assert res_out.stride() == (1,)
    if res_out.numel():
        # An empty tensor holds nullptr storage on both sides, so pointer
        # inequality only proves a fresh buffer for a non-empty result.
        assert res_out.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()


def _out_buffers(kind, dtype, numel):
    """Output buffers for one layout, plus the storage around a view buffer.

    ``exact`` is pre-sized, ``offset`` and ``strided`` are views of a larger
    buffer, and ``empty`` is the supported (non-deprecated) resize form.
    """
    if kind == "empty":
        return (
            torch.empty(0, dtype=dtype, device=flag_gems.device),
            tu.to_reference(torch.empty(0, dtype=dtype, device=flag_gems.device)),
            None,
            None,
        )
    if kind == "exact":
        base = tu.make_input(dtype, (numel,), _VALUE_RANGE)
        return base, tu.to_reference(base), None, None
    if kind == "offset":
        base = tu.make_input(dtype, (numel + 6,), _VALUE_RANGE)
        ref_base = tu.to_reference(base)
        return base[3 : 3 + numel], ref_base[3 : 3 + numel], base, ref_base
    if kind == "strided":
        base = tu.make_input(dtype, (2 * numel,), _VALUE_RANGE)
        ref_base = tu.to_reference(base)
        return base[::2], ref_base[::2], base, ref_base
    raise ValueError(f"Unknown out-buffer layout {kind!r}")


@pytest.mark.test_autograd_multiple_dispatch_view_copy
@pytest.mark.parametrize("dtype", _COPY_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", tu.selected_shapes())
def test__test_autograd_multiple_dispatch_view_copy(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_autograd_multiple_dispatch_view_copy(ref_inp)
    res_out = flag_gems._test_autograd_multiple_dispatch_view_copy(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_materialized_copy(res_out, inp)


@pytest.mark.test_autograd_multiple_dispatch_view_copy
@pytest.mark.parametrize("dtype", _COPY_DTYPES)
@pytest.mark.parametrize("shape", _EMPTY_ROWS)
def test__test_autograd_multiple_dispatch_view_copy_empty(shape, dtype):
    inp = tu.make_input(dtype, shape, ["0", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_autograd_multiple_dispatch_view_copy(ref_inp)
    res_out = flag_gems._test_autograd_multiple_dispatch_view_copy(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_materialized_copy(res_out, inp)


@pytest.mark.test_autograd_multiple_dispatch_view_copy
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
@pytest.mark.parametrize("base_shape,layout", _STRIDED_ROWS)
def test__test_autograd_multiple_dispatch_view_copy_strided_input(
    base_shape, layout, dtype
):
    inp = _LAYOUT_BUILDERS[layout](tu.make_input(dtype, base_shape, _VALUE_RANGE))
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_autograd_multiple_dispatch_view_copy(ref_inp)
    res_out = flag_gems._test_autograd_multiple_dispatch_view_copy(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_materialized_copy(res_out, inp)


@pytest.mark.test_autograd_multiple_dispatch_view_copy
@pytest.mark.parametrize("dtype", _OUT_DTYPES)
@pytest.mark.parametrize("out_kind", _OUT_KINDS)
def test__test_autograd_multiple_dispatch_view_copy_out(out_kind, dtype):
    inp = tu.make_input(dtype, (2, 19, 7), _VALUE_RANGE)
    ref_inp = tu.to_reference(inp)
    buffer, ref_buffer, base, ref_base = _out_buffers(out_kind, dtype, inp.numel())
    # A pre-sized buffer is filled in place; the zero-element buffer is resized.
    storage = None if out_kind == "empty" else buffer.untyped_storage().data_ptr()

    ref_out = torch.ops.aten._test_autograd_multiple_dispatch_view_copy.out(
        ref_inp, out=ref_buffer
    )
    res_out = flag_gems._test_autograd_multiple_dispatch_view_copy(inp, out=buffer)

    assert res_out is buffer
    if storage is not None:
        assert buffer.untyped_storage().data_ptr() == storage
    tu.assert_result_equal(res_out, ref_out)
    if base is not None:
        # A view buffer is written in place, so the surrounding storage keeps
        # the values it had before the call.
        tu.assert_result_equal(base, ref_base)


@pytest.mark.test_autograd_multiple_dispatch_view_copy
@pytest.mark.parametrize("lazy_bit,dtype,layout", _LAZY_ROWS)
def test__test_autograd_multiple_dispatch_view_copy_materializes_lazy_view(
    lazy_bit, dtype, layout
):
    base = tu.make_input(dtype, _LAYOUT_BASE, _VALUE_RANGE)
    ref_base = tu.to_reference(base)
    build = _LAYOUT_BUILDERS[layout]
    if lazy_bit == "neg":
        inp, ref_inp = torch._neg_view(build(base)), torch._neg_view(build(ref_base))
    else:
        inp, ref_inp = build(base).conj(), build(ref_base).conj()

    ref_out = torch.ops.aten._test_autograd_multiple_dispatch_view_copy(ref_inp)
    res_out = flag_gems._test_autograd_multiple_dispatch_view_copy(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_materialized_copy(res_out, inp)
    assert not res_out.is_neg() and not res_out.is_conj()


@pytest.mark.test_autograd_multiple_dispatch_view_copy
def test__test_autograd_multiple_dispatch_view_copy_does_not_alias_input():
    """The copy owns its storage and the source operand stays read-only."""
    inp = tu.make_input(torch.float32, _LAYOUT_BASE, _VALUE_RANGE)
    ref_inp = tu.to_reference(inp)
    inp_before = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_autograd_multiple_dispatch_view_copy(ref_inp)
    res_out = flag_gems._test_autograd_multiple_dispatch_view_copy(inp)

    # The forward values are compared before any write below, so the isolation
    # probe cannot mask a wrong result.
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, inp_before)

    res_out.fill_(1.0)
    tu.assert_result_equal(inp, inp_before)
    inp.fill_(-3.0)
    tu.assert_result_equal(res_out, tu.to_reference(torch.ones_like(res_out)))


@pytest.mark.test_autograd_multiple_dispatch_view_copy
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
@pytest.mark.parametrize("shape", _BACKWARD_ROWS)
def test__test_autograd_multiple_dispatch_view_copy_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, _VALUE_RANGE).requires_grad_(True)
    ref_inp = inp.detach().clone().requires_grad_(True)
    grad_out = tu.make_input(dtype, (inp.numel(),), _VALUE_RANGE)
    ref_grad_out = grad_out

    ref_out = torch.ops.aten._test_autograd_multiple_dispatch_view_copy(ref_inp)
    res_out = flag_gems._test_autograd_multiple_dispatch_view_copy(inp)
    tu.assert_result_equal(res_out, tu.to_reference(ref_out))

    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad_out)
    (res_grad,) = torch.autograd.grad(res_out, inp, grad_outputs=grad_out)

    # The candidate gradient must come back on the device the operator ran on.
    assert res_grad.device == inp.device
    tu.assert_result_equal(res_grad, tu.to_reference(ref_grad))


@pytest.mark.test_autograd_multiple_dispatch_view_copy
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test__test_autograd_multiple_dispatch_view_copy_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_autograd_multiple_dispatch_view_copy(ref_inp)
    res_out = flag_gems._test_autograd_multiple_dispatch_view_copy(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_materialized_copy(res_out, inp)


@pytest.mark.test_autograd_multiple_dispatch_view_copy
@pytest.mark.parametrize("layout", list(_UNFLATTENABLE_BUILDERS))
def test__test_autograd_multiple_dispatch_view_copy_rejects_unflattenable_layout(
    layout,
):
    inp = _UNFLATTENABLE_BUILDERS[layout](
        tu.make_input(torch.float32, _LAYOUT_BASE, _VALUE_RANGE)
    )
    with pytest.raises(RuntimeError):
        flag_gems._test_autograd_multiple_dispatch_view_copy(inp)


@pytest.mark.test_autograd_multiple_dispatch_view_copy
@pytest.mark.parametrize("bad_argument,expected", _INVALID_ARGUMENT_ROWS)
def test__test_autograd_multiple_dispatch_view_copy_rejects_invalid_arguments(
    bad_argument, expected
):
    args = ([1.0, 2.0],) if bad_argument == "non_tensor_operand" else ()
    with pytest.raises(expected):
        flag_gems._test_autograd_multiple_dispatch_view_copy(*args)


@pytest.mark.test_autograd_multiple_dispatch_view_copy
@pytest.mark.parametrize("bad_out,expected", _INVALID_OUT_ROWS)
def test__test_autograd_multiple_dispatch_view_copy_rejects_invalid_out(
    bad_out, expected
):
    inp = tu.make_input(torch.float32, _LAYOUT_BASE, _VALUE_RANGE)
    if bad_out == "dtype_mismatch":
        buffer = torch.empty(inp.numel(), dtype=torch.float16, device=flag_gems.device)
    else:
        buffer = 42
    with pytest.raises(expected):
        flag_gems._test_autograd_multiple_dispatch_view_copy(inp, out=buffer)
