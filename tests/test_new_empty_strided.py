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

from . import test_utils as tu

# aten::new_empty_strided allocates uninitialized storage for an explicit size and
# stride; `self` contributes dtype/device only and is never aliased. The normal path
# computes no element, so the result payload is undefined and is read only under
# deterministic fill, where the allocator defines it. `size` and `stride` are sequence
# arguments, so there is no operand broadcast, no scalar-operand form and no backward
# workload: the result is a fresh allocation, not a function of `self`.

_DTYPE_CAPABILITY = {
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    # Static capability flags, read at import time: no tensor is allocated, no operator
    # is called, and nothing is probed or skipped at run time.
    flag = _DTYPE_CAPABILITY.get(dtype)
    return (
        True if flag is None else bool(getattr(flag_gems.runtime.device, flag, False))
    )


# complex64 rides the 32-bit float path and needs no capability flag of its own.
SUPPORTED_DTYPES = [
    dtype
    for dtype in tu.REQUIRED_DTYPES
    + [torch.float64, torch.bool, torch.complex64, torch.complex128]
    if _dtype_supported(dtype)
]

# Zero-extent requests are valid and keep their explicit stride metadata.
_ZERO_SHAPES = [(0,), (2, 0, 3)]
_LAYOUTS = ["contiguous", "spaced", "zero_stride"]


def _contiguous_stride(size):
    running = 1
    strides = []
    for dim in reversed(size):
        strides.append(running)
        running *= dim
    return tuple(reversed(strides))


def _stride_for(size, layout):
    dense = _contiguous_stride(size)
    if layout == "contiguous":
        return dense
    if layout == "spaced":
        # Doubled contiguous strides: gaps between elements are valid metadata.
        return tuple(step * 2 for step in dense)
    if layout == "zero_stride":
        # A zero outer stride is valid for every non-scalar rank, rank 1 and
        # zero-extent requests included; a 0-dim tensor's stride is the empty tuple.
        return () if not size else (0,) + dense[1:]
    raise ValueError("unsupported layout " + repr(layout))


def _layouts_for(size):
    if not size:
        return ["contiguous"]
    return list(_LAYOUTS)


def _assert_allocation(res_out, ref_out, inp, size, stride, dtype):
    # The payload is undefined, so only the metadata the native operator defines is
    # compared; the native storage byte size already is the element span.
    assert tuple(res_out.shape) == tuple(size)
    assert tuple(res_out.shape) == tuple(ref_out.shape)
    assert res_out.stride() == ref_out.stride() == tuple(stride)
    assert res_out.dtype == dtype
    assert res_out.dtype == ref_out.dtype
    assert res_out.device == inp.device
    assert res_out.layout == torch.strided
    assert res_out.storage_offset() == 0
    assert res_out._is_view() is False
    assert res_out.untyped_storage().nbytes() == ref_out.untyped_storage().nbytes()
    if res_out.numel():
        assert res_out.data_ptr() != inp.data_ptr()


# One row per (requested size, requested stride layout).
_ALL_ROWS = [
    (tuple(shape), layout)
    for shape in list(tu.selected_shapes()) + _ZERO_SHAPES
    for layout in _layouts_for(tuple(shape))
]
# Every cheap boundary stays in quick: all small shapes and the zero-extent requests
# with each layout, across the full dtype set.
_QUICK_SHAPES = [(), (1,), (256,), (2, 19, 7), (0,), (2, 0, 3)]
_QUICK_ROWS = [
    (tuple(shape), layout)
    for shape in _QUICK_SHAPES
    for layout in _layouts_for(tuple(shape))
]
_GRID_ROWS = tu.selected_cases(_ALL_ROWS, quick=_QUICK_ROWS)


@pytest.mark.new_empty_strided
@pytest.mark.parametrize("shape,layout", _GRID_ROWS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_new_empty_strided_allocation(shape, layout, value_range, dtype):
    size = tuple(shape)
    stride = _stride_for(size, layout)
    inp = tu.make_input(dtype, size, value_range)
    inp_before = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_empty_strided(inp_before, list(size), list(stride))
    res_out = flag_gems.new_empty_strided(inp, list(size), list(stride))

    _assert_allocation(res_out, ref_out, inp, size, stride, dtype)
    tu.assert_result_equal(inp, inp_before)


# (requested size, stride, buffer kind, storage retained). The native out
# overload overwrites even retained buffers with undefined data in ordinary
# mode. Only deterministic fill defines the result payload.
_OUT_ROWS = [
    ((2, 3), (3, 1), "matching", True),
    ((0, 3), (3, 1), "matching", True),
    ((), (), "matching", True),
    ((3, 1), (1, 1), "matching", True),
    ((4, 3), (1, 4), "transposed", True),
    ((2, 3), (6, 2), "stride_ignored", True),
    ((2, 3), (3, 1), "int8_buffer", True),
    ((2, 3), (3, 1), "flat_buffer", True),
    ((2, 3), (6, 2), "zero_size_buffer", False),
    ((2, 3), (3, 1), "offset_window", True),
]


def _make_out_buffer(kind, size, dtype):
    base = None
    if kind == "transposed":
        buf = torch.empty(size[1], size[0], dtype=dtype, device=flag_gems.device).t()
    elif kind == "int8_buffer":
        # The out buffer's dtype is authoritative for the result.
        buf = torch.empty(size, dtype=torch.int8, device=flag_gems.device)
    elif kind == "flat_buffer":
        buf = torch.empty(math.prod(size), dtype=dtype, device=flag_gems.device)
    elif kind == "zero_size_buffer":
        buf = torch.empty(0, dtype=dtype, device=flag_gems.device)
    elif kind == "offset_window":
        # A window inside a larger allocation; the guard bytes around it are outside
        # the requested extent and are compared through the parent storage.
        base = torch.empty(math.prod(size) + 2, dtype=dtype, device=flag_gems.device)
        buf = base[2:]
    else:
        buf = torch.empty(size, dtype=dtype, device=flag_gems.device)
    if base is not None:
        base.fill_(False if base.dtype == torch.bool else 7)
    buf.fill_(False if buf.dtype == torch.bool else 7)
    return buf, base


@pytest.mark.new_empty_strided
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("size,stride,kind,keeps_storage", _OUT_ROWS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_new_empty_strided_out_reuses_buffer(
    size, stride, kind, keeps_storage, dtype, deterministic
):
    inp = tu.make_input(dtype, (3, 4), ["-1", "1"])
    inp_before = tu.to_reference(inp)
    buf, base = _make_out_buffer(kind, size, dtype)
    ref_buf, ref_base = _make_out_buffer(kind, size, dtype)
    storage_ptr = buf.untyped_storage().data_ptr()
    storage_nbytes = buf.untyped_storage().nbytes()

    # Distinct sentinels detect writes; deterministic mode defines newly
    # allocated bytes, while ordinary mode checks geometry and guard bytes.
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.utils.deterministic.fill_uninitialized_memory = True
    try:
        ref_out = torch.ops.aten.new_empty_strided.out(
            inp_before, list(size), list(stride), out=ref_buf
        )
        res_out = flag_gems.new_empty_strided(inp, list(size), list(stride), out=buf)
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous_fill
        torch.use_deterministic_algorithms(
            previous_enabled, warn_only=previous_warn_only
        )

    # The out overload hands back the buffer object itself.
    assert res_out is buf
    assert tuple(res_out.shape) == tuple(ref_out.shape)
    assert res_out.stride() == ref_out.stride()
    assert res_out.dtype == ref_out.dtype
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.device == inp.device
    if keeps_storage:
        # A buffer that already satisfies the request retains its storage; a zero-size
        # buffer is reallocated by the resize instead.
        assert res_out.untyped_storage().data_ptr() == storage_ptr
        assert res_out.untyped_storage().nbytes() == storage_nbytes
    if deterministic:
        tu.assert_result_equal(res_out, ref_out)
    if base is not None:
        if deterministic:
            tu.assert_result_equal(base, ref_base)
        else:
            tu.assert_result_equal(base[:2], ref_base[:2])
    tu.assert_result_equal(inp, inp_before)


# Optional-keyword coverage. pin_memory=True is rejected for a non-CPU source and is a
# negative row below; its valid CPU form is a separate test.
_KWARG_ROWS = [
    ("strided_layout", {"layout": torch.strided}),
    ("explicit_device", {"device": flag_gems.device}),
    ("pin_memory_false", {"pin_memory": False}),
]


@pytest.mark.new_empty_strided
@pytest.mark.parametrize(
    "kwargs",
    [row[1] for row in _KWARG_ROWS],
    ids=[row[0] for row in _KWARG_ROWS],
)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_new_empty_strided_optional_kwargs(kwargs, dtype):
    size, stride = (2, 19, 7), (133, 7, 1)
    inp = tu.make_input(dtype, size, ["-1", "1"])
    inp_before = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_empty_strided(
        inp_before, list(size), list(stride), **kwargs
    )
    res_out = flag_gems.new_empty_strided(inp, list(size), list(stride), **kwargs)

    _assert_allocation(res_out, ref_out, inp, size, stride, dtype)
    tu.assert_result_equal(inp, inp_before)


_DTYPE_OVERRIDE_CASES = [
    dtype
    for dtype in [torch.int32, torch.uint8, torch.float64, torch.bool]
    if _dtype_supported(dtype)
]


@pytest.mark.new_empty_strided
@pytest.mark.parametrize("override", _DTYPE_OVERRIDE_CASES, ids=lambda d: str(d))
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_new_empty_strided_dtype_override(override, dtype):
    size, stride = (2, 19, 7), (133, 7, 1)
    inp = tu.make_input(dtype, (2, 2), ["-1", "1"])
    inp_before = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_empty_strided(
        inp_before, list(size), list(stride), dtype=override
    )
    res_out = flag_gems.new_empty_strided(inp, list(size), list(stride), dtype=override)

    _assert_allocation(res_out, ref_out, inp, size, stride, override)
    tu.assert_result_equal(inp, inp_before)


# Pinning is defined for a dense CPU source: both the oracle and the candidate receive
# the same explicit CPU tensor and the same requested CPU device.
@pytest.mark.new_empty_strided
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_new_empty_strided_pin_memory_on_cpu_source(dtype):
    size, stride = (2, 19, 7), (133, 7, 1)
    inp = torch.empty(size, dtype=dtype, device="cpu")
    ref_inp = inp.clone()

    ref_out = torch.ops.aten.new_empty_strided(
        ref_inp, list(size), list(stride), pin_memory=True, device=torch.device("cpu")
    )
    res_out = flag_gems.new_empty_strided(
        inp, list(size), list(stride), pin_memory=True, device=torch.device("cpu")
    )

    assert res_out.is_pinned()
    assert tuple(res_out.shape) == tuple(ref_out.shape) == size
    assert res_out.stride() == ref_out.stride() == tuple(stride)
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.device.type == "cpu"
    assert res_out.untyped_storage().nbytes() == ref_out.untyped_storage().nbytes()


# `self` is metadata-only, so a transposed, offset-window, expanded or
# gapped source cannot change the allocation.
_SOURCE_LAYOUT_ROWS = [
    ("transposed", (4, 6)),
    ("offset_window", (5, 6)),
    ("expanded", (1, 6)),
    ("column_step", (4, 12)),
]


def _apply_source_layout(base, layout):
    if layout == "transposed":
        return base.t()
    if layout == "offset_window":
        return base[2:5]
    if layout == "expanded":
        return base.expand(3, base.shape[-1])
    if layout == "column_step":
        return base[:, ::2]
    raise ValueError("unsupported source layout " + repr(layout))


@pytest.mark.new_empty_strided
@pytest.mark.parametrize("layout,storage_shape", _SOURCE_LAYOUT_ROWS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_new_empty_strided_source_layout_independent(layout, storage_shape, dtype):
    size, stride = (3, 4), (4, 1)
    base = tu.make_input(dtype, storage_shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = _apply_source_layout(base, layout)
    ref_inp = _apply_source_layout(ref_base, layout)
    base_before = tu.to_reference(base)

    ref_out = torch.ops.aten.new_empty_strided(ref_inp, list(size), list(stride))
    res_out = flag_gems.new_empty_strided(inp, list(size), list(stride))

    _assert_allocation(res_out, ref_out, inp, size, stride, dtype)
    tu.assert_result_equal(base, base_before)


# Default-only: the source carries special values, but it contributes dtype/device
# only, so its payload must neither propagate to the undefined result nor be written.
_SPECIAL_SOURCE_CASES = tu.selected_cases(
    tu.special_value_cases(
        [dtype for dtype in SUPPORTED_DTYPES if dtype.is_floating_point]
    ),
    quick=[],
)


@pytest.mark.new_empty_strided
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_SOURCE_CASES)
def test_new_empty_strided_special_source_values(dtype, scenario):
    size, stride = (2, 3), (3, 1)
    inp = tu.make_special_input(dtype, scenario)
    inp_before = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_empty_strided(inp_before, list(size), list(stride))
    res_out = flag_gems.new_empty_strided(inp, list(size), list(stride))

    _assert_allocation(res_out, ref_out, inp, size, stride, dtype)
    tu.assert_result_equal(inp, inp_before)


# A fresh allocation owns its storage: writing the result (a write only, the undefined
# payload is never read) must not reach the source.
_ISOLATION_ROWS = [((2, 3), "contiguous"), ((2, 3), "spaced"), ((3, 4), "zero_stride")]


@pytest.mark.new_empty_strided
@pytest.mark.parametrize("shape,layout", _ISOLATION_ROWS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_new_empty_strided_result_storage_isolated(shape, layout, dtype):
    size = tuple(shape)
    stride = _stride_for(size, layout)
    inp = tu.make_input(dtype, size, ["-1", "1"])
    inp_before = tu.to_reference(inp)

    res_out = flag_gems.new_empty_strided(inp, list(size), list(stride))
    res_out.fill_(1)

    tu.assert_result_equal(inp, inp_before)


@pytest.mark.new_empty_strided
@pytest.mark.parametrize(
    "dtype",
    [
        dtype
        for dtype in SUPPORTED_DTYPES
        if dtype.is_floating_point or dtype.is_complex
    ],
)
def test_new_empty_strided_result_has_no_autograd(dtype):
    inp = tu.make_input(dtype, (3, 4), ["-1", "1"]).requires_grad_(True)
    inp_before = tu.to_reference(inp)

    res_out = flag_gems.new_empty_strided(inp, [2, 3], [3, 1])

    # An allocation is not a function of `self`, so it stays out of the graph.
    assert res_out.requires_grad is False
    assert res_out.grad_fn is None
    tu.assert_result_equal(inp, inp_before)


@pytest.mark.new_empty_strided
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_new_empty_strided_deterministic_fill(dtype):
    # Deterministic algorithms with fill_uninitialized_memory are the one path that
    # defines the payload of a fresh allocation, so the two results are compared
    # exactly (NaN for floats, the dtype maximum for integers, True for bool).
    size, stride = (2, 3), (3, 1)
    inp = tu.make_input(dtype, (2, 2), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.utils.deterministic.fill_uninitialized_memory = True
    try:
        ref_out = torch.ops.aten.new_empty_strided(ref_inp, list(size), list(stride))
        res_out = flag_gems.new_empty_strided(inp, list(size), list(stride))
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous_fill
        torch.use_deterministic_algorithms(
            previous_enabled, warn_only=previous_warn_only
        )

    tu.assert_result_equal(res_out, ref_out)


# Negative rows run in both modes: invalid arguments must be rejected before any
# allocation is attempted.
_INVALID_ROWS = [
    ([2, 3], [1], {}),  # size/stride rank mismatch
    (3, [1], {}),  # size is not a sequence
    ([2.5, 3], [3, 1], {}),  # non-integer extent
    ([-2, 3], [3, 1], {}),  # negative extent
    ([2, 3], [-1, 2], {}),  # negative stride
    ([2, 3], [1 << 62, 1], {}),  # stride overflow
    ([2, 3], [3, 1], {"layout": torch.sparse_coo}),  # no sparse allocation backend
]
if torch.device(flag_gems.device).type != "cpu":
    # pin_memory is a host-memory contract: a non-CPU source is rejected. On a
    # CPU-configured build the same call is valid and is covered positively above.
    _INVALID_ROWS.append(([2, 3], [3, 1], {"pin_memory": True}))


@pytest.mark.new_empty_strided
@pytest.mark.parametrize("size,stride,kwargs", _INVALID_ROWS)
def test_new_empty_strided_rejects_invalid_size_or_stride(size, stride, kwargs):
    inp = tu.make_input(torch.float32, (3, 4), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError, ValueError, NotImplementedError)):
        flag_gems.new_empty_strided(inp, size, stride, **kwargs)


@pytest.mark.new_empty_strided
@pytest.mark.parametrize("self_operand", [3.14, [[1.0, 2.0]]], ids=["scalar", "list"])
def test_new_empty_strided_rejects_non_tensor_self(self_operand):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.new_empty_strided(self_operand, [2], [1])


@pytest.mark.new_empty_strided
def test_new_empty_strided_out_rejects_non_tensor_buffer():
    inp = tu.make_input(torch.float32, (3, 4), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.new_empty_strided(inp, [2, 3], [3, 1], out=[1, 2])


@pytest.mark.new_empty_strided
def test_new_empty_strided_out_rejects_dtype_override():
    # The out schema takes no dtype argument: the buffer's dtype is authoritative.
    inp = tu.make_input(torch.float32, (3, 4), ["-1", "1"])
    buf = torch.empty(2, 3, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.new_empty_strided(inp, [2, 3], [3, 1], out=buf, dtype=torch.float64)
