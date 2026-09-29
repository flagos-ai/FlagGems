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

# clone materializes a fresh copy: it never shares storage with the input and it
# clears the input's lazy conjugate / negation bits. The dtype lists are
# filtered with the suite's static backend capability flags during case
# construction, so collection neither probes the device nor allocates.
_DTYPE_FLAGS = {
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
    torch.bfloat16: utils.bf16_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float64: utils.fp64_is_supported,
}


def _gated(dtypes):
    return [dtype for dtype in dtypes if _DTYPE_FLAGS.get(dtype, True)]


_CLONE_DTYPES = _gated(
    tu.REQUIRED_DTYPES + [torch.float64, torch.bool, torch.complex64]
)
_CLONE_FLOAT_DTYPES = [dtype for dtype in _CLONE_DTYPES if dtype.is_floating_point]

# clone takes one tensor operand and no scalar argument, so the spec's broadcast
# and tensor-vs-scalar dimensions do not apply to this operator.

# Empty tensors are valid copy operands, so they extend the spec shape grid; the
# quick level stays on the (2, 19, 7) smoke shape.
_GRID_SHAPES = tu.selected_cases(
    tu.REQUIRED_SHAPES + [(0,), (0, 5), (3, 0, 4)], quick=tu.QUICK_SHAPES
)

_LAYOUT_DTYPES = tu.selected_cases(
    _gated(
        [
            torch.float32,
            torch.bfloat16,
            torch.int8,
            torch.float8_e4m3fn,
            torch.complex64,
        ]
    ),
    quick=[],
)

# (input shape, input stride pattern, requested memory_format). "preserve_format"
# and "contiguous_format" are explicit calls; the None rows omit the argument and
# therefore also cover the schema default. These supplementary layouts are
# default-only: quick mode runs the main grid, the out smoke row and the
# negative cases.
_LAYOUT_ROWS = tu.selected_cases(
    [
        ((16, 32, 8), "dense", None),
        ((16, 32, 8), "dense", "preserve_format"),
        ((16, 32, 8), "dense", "contiguous_format"),
        ((16, 32, 8), "offset_slice", None),
        ((16, 32, 8), "offset_slice", "preserve_format"),
        ((16, 32, 8), "strided", None),
        ((16, 32, 8), "transposed", None),
        ((16, 32, 8), "transposed", "preserve_format"),
        ((16, 32, 8), "transposed", "contiguous_format"),
        ((16, 32, 8), "expanded", None),
        ((16, 32, 8), "expanded", "contiguous_format"),
        ((2, 3, 4, 5), "dense", "channels_last"),
        ((2, 3, 4, 5), "channels_last", None),
        ((2, 3, 4, 5), "channels_last", "preserve_format"),
        ((2, 3, 4, 5), "channels_last", "contiguous_format"),
        ((2, 3, 5, 4, 6), "dense", "channels_last_3d"),
        ((2, 3, 5, 4, 6), "channels_last_3d", None),
        ((2, 3, 5, 4, 6), "channels_last_3d", "preserve_format"),
    ],
    quick=[],
)

# (out shape, requested memory_format, out buffer layout)
_OUT_ROWS = tu.selected_cases(
    [
        ((2, 19, 7), None, "dense"),
        ((16, 32, 8), None, "dense"),
        ((16, 32, 8), "preserve_format", "dense"),
        ((16, 32, 8), "contiguous_format", "dense"),
        ((2, 3, 4, 5), "channels_last", "channels_last"),
        ((2, 3, 4, 5), None, "channels_last"),
        ((2, 3, 5, 4, 6), "channels_last_3d", "channels_last_3d"),
        ((2, 3, 5, 4, 6), None, "channels_last_3d"),
    ],
    quick=[((2, 19, 7), None, "dense")],
)

_OUT_STRIDED_DTYPES = tu.selected_cases(
    _gated(
        [
            torch.float32,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.complex64,
            torch.int8,
        ]
    ),
    quick=[],
)

_FRESH_ROWS = tu.selected_cases(
    [
        ((16, 32), "dense"),
        ((16, 32), "offset_slice"),
        ((16, 32), "transposed"),
        ((0, 4), "dense"),
    ],
    quick=[],
)
_FRESH_DTYPES = tu.selected_cases(
    _gated([torch.float32, torch.bfloat16, torch.int32]), quick=[]
)

# torch._neg_view has no float8 CUDA kernel (RuntimeError: "neg_cuda" is not
# implemented for 'Float8E4m3fn'), so the lazy-bit cases use the dtypes that can
# actually carry the lazy bit.
_LAZY_BIT_CANDIDATES = [
    ("conj", torch.complex64),
    ("conj", torch.float32),
    ("neg", torch.float32),
    ("neg", torch.bfloat16),
    ("neg", torch.complex64),
]
_LAZY_BIT_CASES = tu.selected_cases(
    [
        (kind, dtype)
        for kind, dtype in _LAZY_BIT_CANDIDATES
        if _DTYPE_FLAGS.get(dtype, True)
    ],
    quick=[],
)

_BACKWARD_ROWS = tu.selected_cases(
    [
        ((16, 32), "dense", None),
        ((4, 8, 16), "dense", None),
        ((16, 32), "transposed", None),
        ((4, 8, 16), "offset_slice", None),
        ((16, 32), "dense", "preserve_format"),
        ((4, 8, 16), "dense", "contiguous_format"),
    ],
    quick=[],
)

_BACKWARD_DTYPES = tu.selected_cases(
    _gated(
        [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float64,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.complex64,
        ]
    ),
    quick=[],
)

# The shared generator already knows that float8_e4m3fn cannot represent inf
# (nan-only) while float8_e5m2 keeps nan, inf and mixed payloads.
_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases(_CLONE_FLOAT_DTYPES), quick=[]
)

_MEMORY_FORMAT_KWARGS = {
    "preserve_format": torch.preserve_format,
    "contiguous_format": torch.contiguous_format,
    "channels_last": torch.channels_last,
    "channels_last_3d": torch.channels_last_3d,
}

# channels_last requires rank 4 and channels_last_3d requires rank 5.
_WRONG_RANK_FORMATS = [
    (torch.channels_last, (2, 3, 4)),
    (torch.channels_last, (2, 3)),
    (torch.channels_last_3d, (2, 3, 4)),
    (torch.channels_last_3d, (2, 3, 5, 4)),
]


def _memory_format_kwargs(memory_format):
    """None means "call without the argument" (the schema default)."""
    if memory_format is None:
        return {}
    return {"memory_format": _MEMORY_FORMAT_KWARGS[memory_format]}


def _sentinel_value(dtype):
    """A fill distinguishable from the [-1, 1] input range these tests build.

    Integer and float tensors use 7, which no element of such an input can
    equal. bool inputs are drawn from {False, True}, so True is the only
    available choice there; the comparison against the native result is what
    establishes the write, not the fill value itself.
    """
    return True if dtype == torch.bool else 7


def _copy_fill(dtype):
    """A different value, written through the copy, for aliasing checks."""
    return False if dtype == torch.bool else 3


def _sentinel_tensor(dtype, shape, layout, device):
    base = torch.full(shape, _sentinel_value(dtype), dtype=dtype, device=device)
    if layout == "channels_last":
        return base.contiguous(memory_format=torch.channels_last)
    if layout == "channels_last_3d":
        return base.contiguous(memory_format=torch.channels_last_3d)
    return base


def _strided_stride(shape):
    """Contiguous strides with one element of padding on the leading axis."""
    stride = [1] * len(shape)
    acc = 1
    for axis in range(len(shape) - 1, -1, -1):
        stride[axis] = acc
        acc *= shape[axis]
    stride[0] += 1
    return tuple(stride)


def _layout_tensors(dtype, shape, layout, value_range):
    """Build a view in the requested layout and return (view, parent).

    ``parent`` is the real allocation the view reads from, so a test can prove
    that the copy never writes back into it - including the elements hidden by
    an offset, a stride gap or an expansion.
    """
    if layout == "dense":
        base = tu.make_input(dtype, shape, value_range)
        return base, base
    if layout in ("channels_last", "channels_last_3d"):
        memory_format = (
            torch.channels_last if layout == "channels_last" else torch.channels_last_3d
        )
        base = tu.make_input(dtype, shape, value_range).contiguous(
            memory_format=memory_format
        )
        return base, base
    if layout == "transposed":
        base = tu.make_input(
            dtype, (shape[1], shape[0]) + tuple(shape[2:]), value_range
        )
        return base.transpose(0, 1), base
    if layout == "offset_slice":
        # Element 0 of the flat parent is the hidden padding for this view.
        base = tu.make_input(dtype, (math.prod(shape) + 1,), value_range)
        return base[1:].view(shape), base
    if layout == "strided":
        base = tu.make_input(dtype, (math.prod(shape) + shape[0],), value_range)
        return torch.as_strided(base, shape, _strided_stride(shape)), base
    # expanded: a size-1 base axis expanded to a larger extent gives that axis
    # stride 0, so the copy has to gather repeated elements.
    axis = next(index for index, dim in enumerate(shape) if dim > 1)
    base_shape = list(shape)
    base_shape[axis] = 1
    base = tu.make_input(dtype, tuple(base_shape), value_range)
    return base.expand(shape), base


@pytest.mark.clone
@pytest.mark.parametrize("shape", _GRID_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _CLONE_DTYPES)
def test_clone(shape, value_range, dtype):
    # Called without memory_format, which also covers the schema default.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.clone(ref_inp)
    res_out = flag_gems.clone(inp)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    # The copy of a dense input is a dense, offset-free allocation of its own.
    assert res_out.is_contiguous()
    assert res_out.storage_offset() == 0
    # clone never writes back into its input.
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.clone
@pytest.mark.parametrize("shape,input_layout,memory_format", _LAYOUT_ROWS)
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test_clone_layout(shape, input_layout, memory_format, dtype):
    inp, parent = _layout_tensors(dtype, shape, input_layout, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_parent = ref_inp if parent is inp else tu.to_reference(parent)
    kwargs = _memory_format_kwargs(memory_format)

    ref_out = torch.ops.aten.clone(ref_inp, **kwargs)
    res_out = flag_gems.clone(inp, **kwargs)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    # preserve_format keeps the strides of a dense, non-overlapping input and
    # falls back to contiguous for overlapping or strided ones; channels_last
    # and contiguous_format re-lay the fresh copy. Native decides all of that,
    # so layout is compared against the reference rather than hard-coded.
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    # Neither the view nor the allocation behind it may change.
    tu.assert_result_equal(inp, ref_inp)
    if ref_parent is not ref_inp:
        tu.assert_result_equal(parent, ref_parent)


@pytest.mark.clone
@pytest.mark.parametrize("shape,memory_format,out_layout", _OUT_ROWS)
@pytest.mark.parametrize("dtype", _CLONE_DTYPES)
def test_clone_out(shape, memory_format, out_layout, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    # Both buffers start outside the input value range, so a copy that skips an
    # element shows up in the buffer comparison.
    buf = _sentinel_tensor(dtype, shape, out_layout, flag_gems.device)
    ref_buf = _sentinel_tensor(dtype, shape, out_layout, ref_inp.device)
    kwargs = _memory_format_kwargs(memory_format)

    ref_out = torch.ops.aten.clone.out(ref_inp, out=ref_buf, **kwargs)
    storage = (buf.data_ptr(), buf.storage_offset())
    res_out = flag_gems.clone(inp, out=buf, **kwargs)

    assert res_out is buf
    assert (buf.data_ptr(), buf.storage_offset()) == storage
    assert res_out.device == buf.device
    assert res_out.stride() == ref_out.stride()
    tu.assert_result_equal(buf, ref_buf)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.clone
@pytest.mark.parametrize("dtype", _OUT_STRIDED_DTYPES)
def test_clone_out_strided_buffer(dtype):
    inp = tu.make_input(dtype, (2, 16, 8), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    # A non-unit-stride view with a nonzero storage offset inside a larger
    # parent allocation: the copy must fill exactly that view.
    parent = _sentinel_tensor(dtype, (4, 32, 16), "dense", flag_gems.device)
    ref_parent = _sentinel_tensor(dtype, (4, 32, 16), "dense", ref_inp.device)
    buf = parent[1:3, :16, ::2]
    ref_buf = ref_parent[1:3, :16, ::2]
    assert not buf.is_contiguous()
    parent_storage = parent.untyped_storage().data_ptr()

    # Native call is needed only for its effect on ref_parent, so its return is
    # not bound.
    torch.ops.aten.clone.out(ref_inp, out=ref_buf)
    res_out = flag_gems.clone(inp, out=buf)

    assert res_out is buf
    assert res_out.device == buf.device
    assert res_out.stride() == ref_buf.stride()
    assert res_out.storage_offset() == ref_buf.storage_offset()
    # The result must still be a view of the caller's allocation: replacing the
    # buffer storage would satisfy "res_out is buf" while dropping the write.
    assert buf.untyped_storage().data_ptr() == parent_storage
    # ref_parent holds the natively written view plus the untouched sentinel
    # padding, so comparing the whole parent also proves the padding survived.
    tu.assert_result_equal(parent, ref_parent)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.clone
@pytest.mark.parametrize("shape,input_layout", _FRESH_ROWS)
@pytest.mark.parametrize("dtype", _FRESH_DTYPES)
def test_clone_copies_into_fresh_storage(shape, input_layout, dtype):
    inp, parent = _layout_tensors(dtype, shape, input_layout, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.clone(ref_inp)
    res_out = flag_gems.clone(inp)

    assert res_out.device == inp.device
    # Values first: an uninitialised or zero-filled output must fail here.
    tu.assert_result_equal(res_out, ref_out)
    if inp.numel():
        assert res_out.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()

    # Writing through the input must not reach the copy.
    inp.fill_(_sentinel_value(dtype))
    tu.assert_result_equal(res_out, ref_out)

    # The post-write state, captured independently so the check below is not a
    # tautology.
    expected_inp = tu.to_reference(inp)
    expected_parent = None if parent is inp else tu.to_reference(parent)

    # Writing through the copy must not reach the input or its hidden padding.
    res_out.fill_(_copy_fill(dtype))
    tu.assert_result_equal(inp, expected_inp)
    if expected_parent is not None:
        tu.assert_result_equal(parent, expected_parent)


@pytest.mark.clone
@pytest.mark.parametrize("kind,dtype", _LAZY_BIT_CASES)
def test_clone_resolves_lazy_view_bits(kind, dtype):
    # The parent is snapshotted before the lazy view is built, so a write into
    # the shared base cannot hide behind the view.
    base = tu.make_input(dtype, (8, 16), ["-1", "1"])
    ref_base = tu.to_reference(base)
    if kind == "conj":
        inp, ref_inp = base.conj(), ref_base.conj()
    else:
        inp, ref_inp = torch._neg_view(base), torch._neg_view(ref_base)

    ref_out = torch.ops.aten.clone(ref_inp)
    res_out = flag_gems.clone(inp)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    # clone materializes the values, so the lazy bits match the reference's.
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.is_neg() == ref_out.is_neg()
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(base, ref_base)


@pytest.mark.clone
@pytest.mark.parametrize("shape,input_layout,memory_format", _BACKWARD_ROWS)
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
def test_clone_backward(shape, input_layout, memory_format, dtype):
    inp, parent = _layout_tensors(dtype, shape, input_layout, ["-1", "1"])
    inp.requires_grad_(True)
    # A non-uniform upstream gradient distinguishes a real gradient copy from a
    # gradient of ones.
    upstream = tu.make_input(dtype, shape, ["0", "max"])
    ref_inp = tu.to_reference(inp)
    ref_parent = None if parent is inp else tu.to_reference(parent)
    ref_upstream = tu.to_reference(upstream)
    kwargs = _memory_format_kwargs(memory_format)

    ref_out = torch.ops.aten.clone(ref_inp, **kwargs)
    res_out = flag_gems.clone(inp, **kwargs)
    tu.assert_result_equal(res_out, ref_out)

    # clone returns a fresh copy, so its gradient is exactly the upstream
    # gradient: the comparison is exact, not tolerance based.
    ref_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)[0]
    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]
    assert res_grad.device == inp.device
    tu.assert_result_equal(res_grad, ref_grad)
    tu.assert_result_equal(inp, ref_inp)
    if ref_parent is not None:
        tu.assert_result_equal(parent, ref_parent)


@pytest.mark.clone
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_clone_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.clone(ref_inp)
    res_out = flag_gems.clone(inp)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.clone
def test_clone_rejects_non_tensor():
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.clone(3.14)


@pytest.mark.clone
@pytest.mark.parametrize("memory_format,shape", _WRONG_RANK_FORMATS)
def test_clone_rejects_memory_format_for_wrong_rank(memory_format, shape):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.clone(inp, memory_format=memory_format)


@pytest.mark.clone
def test_clone_out_rejects_dtype_mismatch():
    inp = tu.make_input(torch.float32, (2, 3), ["-1", "1"])
    buf = torch.empty((2, 3), dtype=torch.float16, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.clone(inp, out=buf)
