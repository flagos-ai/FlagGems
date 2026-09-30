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

"""Correctness tests for ``aten::empty_like`` (``default`` and native ``.out``).

``empty_like`` only allocates storage, so its output elements are undefined and
every comparison is on allocation metadata against ``torch.ops.aten.empty_like``
called on the same input. Broadcast and backward do not apply: the schema has a
single tensor input, computes no elements and builds no autograd graph, which
test_empty_like_allocates_independent_tensor checks instead. Output contents are
read only in test_empty_like_deterministic_fill, where the deterministic fill
makes them defined.
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu


def _assert_allocation_metadata(res_out, ref_out, inp, requested_device=None):
    # empty_like defines no element values, so only metadata is comparable.
    assert res_out.size() == ref_out.size()
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.dtype == ref_out.dtype
    # The candidate output follows its own input, never the reference, which
    # to_reference may have moved to CPU. flag_gems.device may omit the index.
    device = inp.device if requested_device is None else requested_device
    if device.index is None:
        assert res_out.device.type == device.type
    else:
        assert res_out.device == device


# Allocation is dtype-agnostic, so no kernel capability gate applies. Native
# probes accepted REQUIRED_DTYPES plus bool, float64 and complex64; complex128
# needs no element kernel either and is included for the same reason.
SUPPORTED_DTYPES = tu.REQUIRED_DTYPES + [
    torch.bool,
    torch.float64,
    torch.complex64,
    torch.complex128,
]


@pytest.mark.empty_like
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_empty_like_allocation_metadata(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.empty_like(ref_inp)
    res_out = flag_gems.empty_like(inp)

    _assert_allocation_metadata(res_out, ref_out, inp)


# (shape, input permutation, input memory format, operator kwargs). ``None``
# leaves the input untouched and ``{}`` omits the argument, which is the schema
# default preserve_format call; all four enum values appear where the input rank
# permits them. These rows are the memory-format boundary set, so all of them run
# in both the default and the quick level.
_FORMAT_ROWS = [
    ((16, 128, 64, 60), None, None, {}),
    ((16, 128, 64, 60), (2, 0, 1, 3), None, {"memory_format": torch.preserve_format}),
    ((16, 128, 64, 60), (2, 0, 1, 3), None, {"memory_format": torch.contiguous_format}),
    (
        (16, 128, 64, 60),
        None,
        torch.channels_last,
        {"memory_format": torch.channels_last},
    ),
    ((16, 128, 64, 60), None, None, {"memory_format": torch.channels_last}),
    ((16, 7, 57, 32, 29), None, None, {"memory_format": torch.channels_last_3d}),
    (
        (16, 7, 57, 32, 29),
        (4, 0, 1, 2, 3),
        None,
        {"memory_format": torch.contiguous_format},
    ),
]


def _format_input(shape, permutation, memory_format):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    if permutation is not None:
        inp = inp.permute(permutation)
    if memory_format is not None:
        inp = inp.contiguous(memory_format=memory_format)
    return inp


@pytest.mark.empty_like
@pytest.mark.parametrize("case", _FORMAT_ROWS)
def test_empty_like_memory_format(case):
    shape, permutation, input_format, kwargs = case
    inp = _format_input(shape, permutation, input_format)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.empty_like(ref_inp, **kwargs)
    res_out = flag_gems.empty_like(inp, **kwargs)

    _assert_allocation_metadata(res_out, ref_out, inp)


# View inputs: the reference strides must be reproduced for non-contiguous
# inputs, zero strides are not carried over, and the result always gets fresh
# storage at offset 0 (a narrowed input's offset is dropped). All rows are small
# view-layout boundaries and run in both levels.
_VIEW_ROWS = [
    ("transposed", (4, 6)),
    ("narrowed_nonzero_offset", (8, 8)),
    ("strided_slice", (8, 8)),
    ("expanded_zero_stride", (1, 6)),
    ("permuted_4d", (2, 3, 4, 5)),
    ("zero_size", (0, 3)),
    ("scalar", ()),
]


def _view_input(name, shape):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    if name == "transposed":
        return inp.transpose(0, 1)
    if name == "narrowed_nonzero_offset":
        return inp.narrow(0, 3, 4)
    if name == "strided_slice":
        return inp[:, ::2]
    if name == "expanded_zero_stride":
        return inp.expand(6, 6)
    if name == "permuted_4d":
        return inp.permute(2, 0, 3, 1)
    return inp


@pytest.mark.empty_like
@pytest.mark.parametrize("case", _VIEW_ROWS)
def test_empty_like_preserves_view_layout(case):
    name, shape = case
    inp = _view_input(name, shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.empty_like(ref_inp)
    res_out = flag_gems.empty_like(inp)

    _assert_allocation_metadata(res_out, ref_out, inp)


_INDEPENDENCE_ROWS = [
    (torch.float32, (4, 6), True),
    (torch.float64, (2, 3, 4), True),
    (torch.int8, (64,), False),
]


@pytest.mark.empty_like
@pytest.mark.parametrize("dtype,shape,needs_grad", _INDEPENDENCE_ROWS)
def test_empty_like_allocates_independent_tensor(dtype, shape, needs_grad):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    if needs_grad:
        inp = inp.clone().requires_grad_(True)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.empty_like(ref_inp)
    res_out = flag_gems.empty_like(inp)

    _assert_allocation_metadata(res_out, ref_out, inp)
    # Fresh storage: no alias with the input and no view on it.
    assert not res_out._is_view()
    assert res_out.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()
    # An allocation never propagates autograd state; this stands in for the
    # backward dimension, which does not exist for empty_like.
    assert res_out.requires_grad == ref_out.requires_grad
    assert (res_out.grad_fn is None) == (ref_out.grad_fn is None)


@pytest.mark.empty_like
@pytest.mark.parametrize("lazy_kind", ["conj", "neg"])
def test_empty_like_drops_lazy_conj_and_neg(lazy_kind):
    if lazy_kind == "conj":
        inp = tu.make_input(torch.complex64, (4, 6), ["-1", "1"]).conj()
    else:
        inp = torch._neg_view(tu.make_input(torch.float32, (4, 6), ["-1", "1"]))
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.empty_like(ref_inp)
    res_out = flag_gems.empty_like(inp)

    # A fresh allocation materializes neither lazy bit.
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.is_neg() == ref_out.is_neg()
    _assert_allocation_metadata(res_out, ref_out, inp)


# (operator kwargs, explicitly requested output device or None)
_OVERRIDE_ROWS = [
    ({"dtype": torch.int32}, None),
    ({"dtype": torch.float64, "layout": torch.strided}, None),
    ({"device": torch.device(flag_gems.device)}, torch.device(flag_gems.device)),
    ({"pin_memory": False}, None),
    (
        {
            "dtype": None,
            "layout": None,
            "device": None,
            "pin_memory": None,
            "memory_format": None,
        },
        None,
    ),
]
_OVERRIDE_IDS = [
    "dtype_override",
    "dtype_and_layout_override",
    "device_override",
    "pin_memory_false",
    "explicit_none_arguments",
]


@pytest.mark.empty_like
@pytest.mark.parametrize("kwargs,device", _OVERRIDE_ROWS, ids=_OVERRIDE_IDS)
def test_empty_like_dtype_device_overrides(kwargs, device):
    inp = tu.make_input(torch.float32, (16, 128), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.empty_like(ref_inp, **kwargs)
    res_out = flag_gems.empty_like(inp, **kwargs)

    assert res_out.dtype == ref_out.dtype
    _assert_allocation_metadata(res_out, ref_out, inp, requested_device=device)


# The .out overload is natively callable (aten::empty_like.out(x, out=buf)
# returns ``buf`` itself), so it is called directly on both sides. Buffer contents
# are not asserted: the overload does not define them.
_OUT_DTYPES = [torch.float32, torch.int8, torch.float8_e4m3fn]


@pytest.mark.empty_like
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _OUT_DTYPES)
def test_empty_like_out(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    ref_buf = torch.empty(shape, dtype=dtype, device=ref_inp.device)
    act_buf = torch.empty(shape, dtype=dtype, device=inp.device)

    ref_out = torch.ops.aten.empty_like.out(ref_inp, out=ref_buf)
    res_out = flag_gems.empty_like(inp, out=act_buf)

    assert res_out is act_buf
    _assert_allocation_metadata(res_out, ref_out, inp)


# Out-buffer forms the input dtype does not constrain: the buffer keeps its own
# dtype, and a buffer smaller than the input is resized in place by the native
# overload (its deprecated Resize path).
# (input dtype, buffer dtype, input shape, buffer shape, transpose, memory_format)
_OUT_VARIANT_ROWS = [
    (torch.float32, torch.float64, (4, 6), (4, 6), False, None),
    (torch.int8, torch.float32, (4, 6), (4, 6), False, None),
    (torch.float32, torch.float32, (4, 6), (2,), False, None),
    (torch.float32, torch.float32, (4, 6), (6, 4), True, torch.contiguous_format),
]


@pytest.mark.empty_like
@pytest.mark.parametrize("case", _OUT_VARIANT_ROWS)
def test_empty_like_out_buffer_variants(case):
    in_dtype, buf_dtype, shape, buf_shape, transpose, memory_format = case
    inp = tu.make_input(in_dtype, shape, ["-1", "1"])
    if transpose:
        inp = inp.transpose(0, 1)
    ref_inp = tu.to_reference(inp)
    ref_buf = torch.empty(buf_shape, dtype=buf_dtype, device=ref_inp.device)
    act_buf = torch.empty(buf_shape, dtype=buf_dtype, device=inp.device)
    kwargs = {} if memory_format is None else {"memory_format": memory_format}

    ref_out = torch.ops.aten.empty_like.out(ref_inp, out=ref_buf, **kwargs)
    res_out = flag_gems.empty_like(inp, out=act_buf, **kwargs)

    assert res_out is act_buf
    _assert_allocation_metadata(res_out, ref_out, inp)


# The only path that writes elements. With deterministic algorithms enabled and
# the fill flag on, a fresh allocation is filled, so its contents are defined and
# must match exactly (tu.assert_result_equal matches the NaN fill via equal_nan);
# the comparison is unconditional - reading the candidate buffer to decide
# whether to assert would hide a missing or wrong fill. All three switches are
# saved, set and restored here so no other test inherits a modified
# deterministic/fill state. Float dtypes only: the NaN fill needs floating
# storage. The tensors stay small, so this branch is also exercised in quick.
_FILL_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.empty_like
@pytest.mark.parametrize("dtype", _FILL_DTYPES)
def test_empty_like_deterministic_fill(dtype):
    inp = tu.make_input(dtype, (64, 64), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    saved_enabled = torch.are_deterministic_algorithms_enabled()
    saved_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    saved_fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.utils.deterministic.fill_uninitialized_memory = True
        ref_out = torch.ops.aten.empty_like(ref_inp)
        res_out = flag_gems.empty_like(inp)
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = saved_fill
        torch.use_deterministic_algorithms(saved_enabled, warn_only=saved_warn_only)

    tu.assert_result_equal(res_out, ref_out)
    _assert_allocation_metadata(res_out, ref_out, inp)


# NaN/Inf input contents cannot reach an allocation, so they are covered through
# the unchanged allocation contract (never by comparing undefined results). The
# matrix comes from the shared helper, which already encodes e4m3fn as nan-only
# and e5m2 as nan/inf/mixed. Default-only, as positive special-value cases are.
_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases(
        [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float64,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]
    ),
    quick=[],
)


@pytest.mark.empty_like
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_empty_like_special_value_inputs(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.empty_like(ref_inp)
    res_out = flag_gems.empty_like(inp)

    _assert_allocation_metadata(res_out, ref_out, inp)


# Native rejections: a string memory_format/dtype is a schema mismatch;
# channels_last / channels_last_3d on a 2-D input require rank 4 / rank 5;
# sparse layout is not implemented for the strided factory; pin_memory=True on a
# device tensor is rejected ('Only dense CPU tensors can be pinned'), which is why
# the True side is covered here as a rejection instead. Every row is collected in
# both the default and the quick level.
_NEGATIVE_ROWS = [
    ("memory_format_string", {"memory_format": "contiguous"}),
    ("dtype_string", {"dtype": "float32"}),
    ("channels_last_on_2d", {"memory_format": torch.channels_last}),
    ("channels_last_3d_on_2d", {"memory_format": torch.channels_last_3d}),
    ("sparse_layout", {"layout": torch.sparse_coo}),
] + (
    [("pin_memory_on_device", {"pin_memory": True})]
    if flag_gems.device != "cpu"
    else []
)


@pytest.mark.empty_like
@pytest.mark.parametrize("case", _NEGATIVE_ROWS, ids=[row[0] for row in _NEGATIVE_ROWS])
def test_empty_like_rejects_invalid_arguments(case):
    _, kwargs = case
    inp = tu.make_input(torch.float32, (2, 6), ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems.empty_like(inp, **kwargs)


@pytest.mark.empty_like
def test_empty_like_rejects_non_tensor_input():
    with pytest.raises(RuntimeError):
        flag_gems.empty_like([1.0, 2.0])
