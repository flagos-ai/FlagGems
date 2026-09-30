# Copyright 2025 The FlagGems Authors.
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

"""Correctness tests for aten::new_empty.

new_empty is an allocation operator: it returns a fresh buffer of the requested
``size`` while dtype/device/layout default to the source tensor.  The payload of
the result is undefined, so the value-range grid only varies the source and every
assertion is made on allocation metadata (shape / dtype / device / layout /
stride / storage offset / storage identity) or on the untouched source;
undefined filler is never compared.  The one documented case in which the buffer
is written instead of left uninitialized -- torch's deterministic-algorithms fill
-- has its own test.

Two spec dimensions cannot be expressed for this operator and are stated here
instead of being silently dropped: there are no operand pairs, so there is no
broadcast workload (the only other argument is ``size``), and the fresh
allocation carries no autograd history (probed: requires_grad False and
grad_fn None even for a source that tracks gradients), so there is no backward
workload.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Every gate below is a static accuracy_utils capability flag, so collection
# neither allocates a tensor nor probes the native operator.
_DTYPE_GATES = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
}
_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2] if utils.fp8_is_supported else []
_BF16_DTYPES = [torch.bfloat16] if utils.bf16_is_supported else []
_INT64_DTYPES = [torch.int64] if utils.int64_is_supported else []
_FP64_DTYPES = [torch.float64] if utils.fp64_is_supported else []

NEW_EMPTY_DTYPES = (
    [d for d in tu.REQUIRED_DTYPES if _DTYPE_GATES.get(d, True)]
    + [torch.bool, torch.int16, torch.complex64]
    + _FP64_DTYPES
)

_SMALL_SOURCE = (4,)
# make_input resolves its range argument through the shared symbolic table, so
# auxiliary inputs must use a shared descriptor, not a literal bound pair.
_SOURCE_RANGE = tu.selected_ranges()[0]
_TARGET_SIZE = (2, 3)
_PAD = 3
_ZERO_SIZE_SHAPES = [(0,), (2, 0, 4), (0, 0)]

# Every semantic boundary below keeps a cheap representative row in quick: the
# dtype inheritance, the size forms, the source layouts, the autograd state, the
# optional-parameter forms, the sparse layouts, the out-buffer forms and the
# pinned host allocation.  Only the extra positive special values stay default
# only, and every negative case runs in both modes.
_INHERITS_DTYPE_CASES = NEW_EMPTY_DTYPES
_SIZE_FORMS = [(), (2, 3), torch.Size([2, 3]), [True, 2]]
_SOURCE_VARIANT_DTYPES = [torch.float32, torch.int8]
_SOURCE_VARIANT_PARENT = (4, 12)
_SOURCE_VARIANTS = ["contiguous", "noncontiguous", "offset", "expanded"]
_DIFFERENTIABLE_DTYPES = [torch.float32] + _FP64_DTYPES
_DENSE_LAYOUT_CASES = [{}, {"layout": None}, {"layout": torch.strided}]
_SPECIAL_DTYPES = [torch.float16, torch.float32] + _BF16_DTYPES + _FP8_DTYPES
_SPECIAL_CASES = tu.selected_cases(
    list(tu.special_value_cases(_SPECIAL_DTYPES)), quick=[]
)

_DEVICE_IS_HOST = torch.device(flag_gems.device).type == "cpu"
# An indexed request is honoured exactly; an un-indexed accelerator request
# resolves to the current device index.
_DEVICE_OVERRIDES = (
    [torch.device("cpu")]
    if _DEVICE_IS_HOST
    else [torch.device("cpu"), torch.device(flag_gems.device)]
)
_EXPLICIT_DEVICE_DTYPES = [torch.float32] + _INT64_DTYPES
_EXPLICIT_DEVICE_CASES = [
    (device, dtype) for device in _DEVICE_OVERRIDES for dtype in _EXPLICIT_DEVICE_DTYPES
]

_EXPLICIT_DTYPE_SOURCES = [torch.float16, torch.float32] + _INT64_DTYPES
_EXPLICIT_DTYPES = (
    [torch.float32, torch.int32, torch.uint8, torch.complex64]
    + _BF16_DTYPES
    + _INT64_DTYPES
    + _FP8_DTYPES
)
_EXPLICIT_DTYPE_CASES = [
    (src, out) for src in _EXPLICIT_DTYPE_SOURCES for out in _EXPLICIT_DTYPES
]

# Sparse factories materialise the structural index storage of the layout; the
# probed COO/CSR/CSC empty tensors keep int64 indices, so eligibility follows the
# backend int64 capability.  _SPARSE_DTYPES only widens the payload choice.
_SPARSE_LAYOUTS = (
    [torch.sparse_coo, torch.sparse_csr, torch.sparse_csc]
    if utils.int64_is_supported
    else []
)
_SPARSE_CONSTRUCTORS = {
    torch.sparse_coo: lambda dense: dense.to_sparse(),
    torch.sparse_csr: lambda dense: dense.to_sparse_csr(),
    torch.sparse_csc: lambda dense: dense.to_sparse_csc(),
}
_BLOCK_LAYOUTS = [torch.sparse_bsr, torch.sparse_bsc]
_SPARSE_SHAPE = (2, 3)
_SPARSE_DTYPES = [torch.float32] + _INT64_DTYPES

_OUT_SIZES = tu.selected_cases([(256,), (20, 320, 15)], quick=[(256,)])
# The .out overload has no dtype argument, so the buffer keeps its own dtype and
# the source dtype is irrelevant to the result.
_OUT_CASES = [
    (torch.float32, torch.float32),
    (torch.int32, torch.int32),
    (torch.float32, torch.int32),
]
# (0,) grows an empty buffer, (1,) grows a short one, (5, 5) shrinks.
_OUT_RESIZE_BUFFERS = [(0,), (1,), (5, 5)]
# (row offset of the out view inside its padded parent, requested size).  The
# offset-0 row is the plain strided view, the offset-1 row adds a non-zero
# storage offset.
_OUT_STRIDED_CASES = [(0, _TARGET_SIZE), (1, _TARGET_SIZE)]

# pin_memory only ever describes host storage: the positive case is a property of
# the requested CPU output and of the pinned allocator the surrounding torch build
# ships, not of the accelerator the source lives on.  The rejection is a separate
# native rule ('Only dense CPU tensors can be pinned') and only meaningful while
# the configured device is not the host.
_PINNED_HOST_DTYPES = [torch.float32]
_PINNED_DEVICE_REJECTION_DTYPES = [] if _DEVICE_IS_HOST else [torch.float32]

# Argument-type mismatches are rejected by the aten schema matcher with
# RuntimeError and by Python-level argument parsing with TypeError; an invalid
# extent value is RuntimeError in both layers.
_ARGUMENT_TYPE_ERRORS = (TypeError, RuntimeError)
_SIZE_ERROR_CASES = [
    (4, _ARGUMENT_TYPE_ERRORS),
    ((1.5, 2), _ARGUMENT_TYPE_ERRORS),
    (((2, 3),), _ARGUMENT_TYPE_ERRORS),
    ((-1,), RuntimeError),
    ((2, -3), RuntimeError),
]
_SELF_ERROR_CASES = [
    pytest.param(None, _ARGUMENT_TYPE_ERRORS, id="self_none"),
    pytest.param([1, 2], _ARGUMENT_TYPE_ERRORS, id="self_list"),
]


def _sparse_source(dtype, layout):
    """Build an empty sparse source directly, without a dense payload read."""
    dense = torch.zeros(_SPARSE_SHAPE, dtype=dtype, device=flag_gems.device)
    return _SPARSE_CONSTRUCTORS[layout](dense)


def _check_allocation(res_out, ref_out, inp):
    """Compare the observable allocation metadata against the native result.

    Both tensors hold unrelated undefined payloads, so only the geometry the
    operator actually specifies is compared; the output device is checked
    against the candidate's own input rather than the reference tensor.
    """
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.device == inp.device
    assert res_out.layout == ref_out.layout
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()


def _source_with_parent(dtype, variant):
    # The parent and a pre-call snapshot are returned next to the view so the
    # test can detect writes that land outside the view but inside its storage.
    parent = tu.make_input(dtype, _SOURCE_VARIANT_PARENT, _SOURCE_RANGE)
    # tu.to_reference clones the storage independently, so the snapshot is a true
    # point-in-time copy under either configured reference device.
    snapshot = tu.to_reference(parent)
    if variant == "contiguous":
        view = parent
    elif variant == "noncontiguous":
        view = parent.t()
    elif variant == "offset":
        view = parent[1:, 3:]
    elif variant == "expanded":
        view = parent[:1, :1].expand(_SOURCE_VARIANT_PARENT)
    else:
        raise AssertionError(f"unknown source variant {variant}")
    return view, parent, snapshot


@pytest.mark.new_empty
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", NEW_EMPTY_DTYPES)
def test_new_empty(shape, value_range, dtype):
    # The source payload is never read: its dtype and device only supply the
    # defaults of the allocation.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_empty(ref_inp, shape)
    res_out = flag_gems.new_empty(inp, shape)

    _check_allocation(res_out, ref_out, inp)
    assert res_out.shape == inp.shape
    assert res_out.dtype == inp.dtype
    assert res_out.layout == torch.strided
    assert res_out.storage_offset() == 0
    assert res_out.is_contiguous()
    # a factory result owns fresh storage, never an alias of the source
    assert res_out.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_empty
@pytest.mark.parametrize("shape", _ZERO_SIZE_SHAPES)
@pytest.mark.parametrize("dtype", NEW_EMPTY_DTYPES)
def test_new_empty_zero_size(shape, dtype):
    inp = tu.make_input(dtype, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_empty(ref_inp, shape)
    res_out = flag_gems.new_empty(inp, shape)

    _check_allocation(res_out, ref_out, inp)
    assert res_out.shape == torch.Size(shape)
    assert res_out.numel() == 0
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_empty
@pytest.mark.parametrize("size", _SIZE_FORMS)
def test_new_empty_size_forms(size):
    inp = tu.make_input(torch.float32, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_empty(ref_inp, size)
    res_out = flag_gems.new_empty(inp, size)

    # a bool extent denotes 1, so the requested size is normalised first
    expected = torch.Size(int(extent) for extent in size)
    _check_allocation(res_out, ref_out, inp)
    assert res_out.shape == expected
    assert res_out.dtype == inp.dtype
    assert res_out.is_contiguous()
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_empty
@pytest.mark.parametrize("dtype", _INHERITS_DTYPE_CASES)
def test_new_empty_inherits_source_metadata(dtype):
    inp = tu.make_input(dtype, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_empty(ref_inp, _TARGET_SIZE)
    res_out = flag_gems.new_empty(inp, _TARGET_SIZE)

    _check_allocation(res_out, ref_out, inp)
    assert res_out.shape == torch.Size(_TARGET_SIZE)
    # with no explicit override the output follows the source dtype and device
    assert res_out.dtype == inp.dtype
    assert res_out.device == inp.device
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_empty
@pytest.mark.parametrize("src_dtype,out_dtype", _EXPLICIT_DTYPE_CASES)
def test_new_empty_explicit_dtype(src_dtype, out_dtype):
    inp = tu.make_input(src_dtype, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_empty(ref_inp, _TARGET_SIZE, dtype=out_dtype)
    res_out = flag_gems.new_empty(inp, _TARGET_SIZE, dtype=out_dtype)

    _check_allocation(res_out, ref_out, inp)
    assert res_out.dtype == out_dtype
    assert res_out.shape == torch.Size(_TARGET_SIZE)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_empty
@pytest.mark.parametrize("device,dtype", _EXPLICIT_DEVICE_CASES)
def test_new_empty_explicit_device(device, dtype):
    inp = tu.make_input(dtype, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_empty(ref_inp, _TARGET_SIZE, device=device)
    res_out = flag_gems.new_empty(inp, _TARGET_SIZE, device=device)

    assert res_out.shape == ref_out.shape == torch.Size(_TARGET_SIZE)
    assert res_out.dtype == ref_out.dtype == dtype
    if torch.device(device).type == inp.device.type:
        assert res_out.device == inp.device
    else:
        assert res_out.device == torch.device(device)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_empty
@pytest.mark.parametrize("variant", _SOURCE_VARIANTS)
@pytest.mark.parametrize("dtype", _SOURCE_VARIANT_DTYPES)
def test_new_empty_source_geometry_is_irrelevant(variant, dtype):
    inp, parent, snapshot = _source_with_parent(dtype, variant)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_empty(ref_inp, _TARGET_SIZE)
    res_out = flag_gems.new_empty(inp, _TARGET_SIZE)

    # The requested size alone defines the output, and the source storage must be
    # untouched: the whole parent is compared against its pre-call snapshot, so a
    # write outside the view but inside its storage is still detected.
    _check_allocation(res_out, ref_out, inp)
    assert res_out.shape == torch.Size(_TARGET_SIZE)
    assert res_out.is_contiguous()
    tu.assert_result_equal(parent, snapshot)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_empty
@pytest.mark.parametrize("dtype", _DIFFERENTIABLE_DTYPES)
def test_new_empty_output_is_not_differentiable(dtype):
    inp = tu.make_input(dtype, _SMALL_SOURCE, _SOURCE_RANGE).requires_grad_(True)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_empty(ref_inp, _TARGET_SIZE)
    res_out = flag_gems.new_empty(inp, _TARGET_SIZE)

    # the allocation contract still holds for a source that tracks gradients
    _check_allocation(res_out, ref_out, inp)
    assert res_out.shape == torch.Size(_TARGET_SIZE)
    # uninitialized storage is independent of the source, so autograd must not
    # track it and there is no backward workload for this operator
    assert not res_out.requires_grad
    assert res_out.grad_fn is None
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_empty
@pytest.mark.parametrize("layout_kwargs", _DENSE_LAYOUT_CASES)
def test_new_empty_dense_layout(layout_kwargs):
    inp = tu.make_input(torch.float32, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_empty(ref_inp, _TARGET_SIZE, **layout_kwargs)
    res_out = flag_gems.new_empty(inp, _TARGET_SIZE, **layout_kwargs)

    _check_allocation(res_out, ref_out, inp)
    assert res_out.layout == torch.strided
    assert res_out.is_contiguous()
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_empty
@pytest.mark.parametrize("layout", _SPARSE_LAYOUTS)
@pytest.mark.parametrize("dtype", _SPARSE_DTYPES)
def test_new_empty_sparse_layout(layout, dtype):
    inp = tu.make_input(dtype, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_empty(ref_inp, _SPARSE_SHAPE, layout=layout)
    res_out = flag_gems.new_empty(inp, _SPARSE_SHAPE, layout=layout)

    # the sparse payload is undefined too; the structure the layout parameter
    # determines is compared instead
    assert res_out.layout == ref_out.layout == layout
    assert res_out.shape == ref_out.shape == torch.Size(_SPARSE_SHAPE)
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.device == inp.device
    assert res_out._nnz() == ref_out._nnz()
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_empty
@pytest.mark.parametrize("layout", _SPARSE_LAYOUTS)
@pytest.mark.parametrize("dtype", _SPARSE_DTYPES)
def test_new_empty_sparse_source_inherits_layout(layout, dtype):
    inp = _sparse_source(dtype, layout)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_empty(ref_inp, _SPARSE_SHAPE)
    res_out = flag_gems.new_empty(inp, _SPARSE_SHAPE)

    # a sparse source propagates its layout to the fresh allocation
    assert res_out.layout == ref_out.layout == layout
    assert res_out.shape == ref_out.shape == torch.Size(_SPARSE_SHAPE)
    assert res_out._nnz() == ref_out._nnz()
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_empty
@pytest.mark.parametrize("layout", _SPARSE_LAYOUTS)
def test_new_empty_sparse_source_to_strided(layout):
    inp = _sparse_source(torch.float32, layout)
    ref_inp = tu.to_reference(inp)

    # an explicit strided layout overrides the source's sparse layout
    ref_out = torch.ops.aten.new_empty(ref_inp, _SPARSE_SHAPE, layout=torch.strided)
    res_out = flag_gems.new_empty(inp, _SPARSE_SHAPE, layout=torch.strided)

    assert res_out.layout == ref_out.layout == torch.strided
    assert res_out.shape == ref_out.shape == torch.Size(_SPARSE_SHAPE)
    assert res_out.is_contiguous()
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_empty
@pytest.mark.parametrize("size", _OUT_SIZES)
@pytest.mark.parametrize("src_dtype,out_dtype", _OUT_CASES)
def test_new_empty_out_returns_same_buffer(size, src_dtype, out_dtype):
    inp = tu.make_input(src_dtype, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)
    ref_buf = torch.full(size, 7, dtype=out_dtype, device=ref_inp.device)
    act_buf = torch.full(size, 7, dtype=out_dtype, device=inp.device)

    ref_out = torch.ops.aten.new_empty.out(ref_inp, size, out=ref_buf)
    res_out = flag_gems.new_empty(inp, size, out=act_buf)

    # the out overload writes into the caller's object and returns that object
    assert res_out is act_buf
    assert res_out.shape == ref_out.shape == torch.Size(size)
    assert res_out.dtype == ref_out.dtype == out_dtype
    assert res_out.device == inp.device
    # A matching initialized buffer keeps its defined payload.
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_empty
@pytest.mark.parametrize("buffer_shape", _OUT_RESIZE_BUFFERS)
def test_new_empty_out_resizes_buffer(buffer_shape):
    inp = tu.make_input(torch.float32, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)
    ref_buf = torch.full(buffer_shape, 7.0, dtype=torch.float32, device=ref_inp.device)
    act_buf = torch.full(buffer_shape, 7.0, dtype=torch.float32, device=inp.device)

    ref_out = torch.ops.aten.new_empty.out(ref_inp, _TARGET_SIZE, out=ref_buf)
    res_out = flag_gems.new_empty(inp, _TARGET_SIZE, out=act_buf)

    # the supplied buffer is adapted to the requested size and returned, not
    # replaced by a freshly allocated tensor
    assert res_out is act_buf
    assert res_out.shape == ref_out.shape == torch.Size(_TARGET_SIZE)
    assert res_out.dtype == torch.float32
    assert res_out.device == inp.device
    assert res_out.is_contiguous()
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_empty
@pytest.mark.parametrize("row_offset,size", _OUT_STRIDED_CASES)
def test_new_empty_out_strided_view(row_offset, size):
    inp = tu.make_input(torch.float32, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    rows, cols = size
    parent_cols = cols + _PAD
    ref_parent = torch.full(
        (rows + 1, parent_cols), 7.0, dtype=torch.float32, device=ref_inp.device
    )
    act_parent = torch.full(
        (rows + 1, parent_cols), 7.0, dtype=torch.float32, device=inp.device
    )
    ref_buf = ref_parent[row_offset : row_offset + rows, :cols]
    act_buf = act_parent[row_offset : row_offset + rows, :cols]

    ref_out = torch.ops.aten.new_empty.out(ref_inp, size, out=ref_buf)
    res_out = flag_gems.new_empty(inp, size, out=act_buf)

    assert res_out is act_buf
    assert res_out.shape == ref_out.shape == torch.Size(size)
    assert res_out.dtype == ref_out.dtype == torch.float32
    assert res_out.device == inp.device
    # the non-contiguous buffer keeps its parent storage and its own geometry
    assert act_buf.storage_offset() == row_offset * parent_cols
    assert act_buf.stride() == ref_buf.stride()
    assert not act_buf.is_contiguous()
    assert (
        act_buf.untyped_storage().data_ptr() == act_parent.untyped_storage().data_ptr()
    )
    # The matching initialized view and its surrounding guards stay defined.
    tu.assert_result_equal(act_parent, ref_parent)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_empty
@pytest.mark.parametrize("dtype", _PINNED_HOST_DTYPES)
def test_new_empty_host_pinned_output(dtype):
    inp = tu.make_input(dtype, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_empty(
        ref_inp, _TARGET_SIZE, device=torch.device("cpu"), pin_memory=True
    )
    res_out = flag_gems.new_empty(
        inp, _TARGET_SIZE, device=torch.device("cpu"), pin_memory=True
    )

    assert res_out.shape == ref_out.shape == torch.Size(_TARGET_SIZE)
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.device == torch.device("cpu")
    assert res_out.is_pinned()
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_empty
@pytest.mark.parametrize("dtype", _PINNED_DEVICE_REJECTION_DTYPES)
def test_new_empty_rejects_pinned_device_output(dtype):
    inp = tu.make_input(dtype, _SMALL_SOURCE, _SOURCE_RANGE)

    # pinning is a host-memory property; requesting it for accelerator storage is
    # rejected by the native allocator
    with pytest.raises(RuntimeError):
        flag_gems.new_empty(inp, _TARGET_SIZE, device=inp.device, pin_memory=True)


@pytest.mark.new_empty
@pytest.mark.parametrize("dtype", NEW_EMPTY_DTYPES)
def test_new_empty_deterministic_uninitialized_fill(dtype):
    # This is the one documented case in which new_empty writes the new storage
    # instead of leaving it uninitialized, so the payload becomes observable and
    # is compared against the native result exactly.  torch's deterministic
    # switch is process-global (a plain function, not a context manager), so the
    # enabled/warn_only/fill settings are all restored in a finally block for the
    # rest of the session.
    inp = tu.make_input(dtype, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    was_deterministic = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill_flag = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    try:
        ref_out = torch.ops.aten.new_empty(ref_inp, _TARGET_SIZE)
        res_out = flag_gems.new_empty(inp, _TARGET_SIZE)
    finally:
        torch.use_deterministic_algorithms(was_deterministic, warn_only=was_warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = fill_flag

    assert res_out.shape == ref_out.shape == torch.Size(_TARGET_SIZE)
    assert res_out.dtype == ref_out.dtype == dtype
    # the fill is part of the contract here: NaN matches NaN, INT_MAX matches
    # INT_MAX
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_empty
@pytest.mark.parametrize("size,error", _SIZE_ERROR_CASES)
def test_new_empty_rejects_invalid_size(size, error):
    inp = tu.make_input(torch.float32, _SMALL_SOURCE, _SOURCE_RANGE)

    with pytest.raises(error):
        flag_gems.new_empty(inp, size)


@pytest.mark.new_empty
@pytest.mark.parametrize("self_,error", _SELF_ERROR_CASES)
def test_new_empty_rejects_invalid_self(self_, error):
    with pytest.raises(error):
        flag_gems.new_empty(self_, _TARGET_SIZE)


@pytest.mark.new_empty
def test_new_empty_rejects_invalid_dtype():
    inp = tu.make_input(torch.float32, _SMALL_SOURCE, _SOURCE_RANGE)

    with pytest.raises(_ARGUMENT_TYPE_ERRORS):
        flag_gems.new_empty(inp, _TARGET_SIZE, dtype="not_a_dtype")


@pytest.mark.new_empty
def test_new_empty_rejects_invalid_layout():
    inp = tu.make_input(torch.float32, _SMALL_SOURCE, _SOURCE_RANGE)

    with pytest.raises(_ARGUMENT_TYPE_ERRORS):
        flag_gems.new_empty(inp, _TARGET_SIZE, layout="not_a_layout")


@pytest.mark.new_empty
@pytest.mark.parametrize("layout", _BLOCK_LAYOUTS)
def test_new_empty_rejects_block_sparse_layout(layout):
    inp = tu.make_input(torch.float32, _SMALL_SOURCE, _SOURCE_RANGE)

    # the compressed empty-tensor helper only accepts non-block layouts
    with pytest.raises(RuntimeError):
        flag_gems.new_empty(inp, _SPARSE_SHAPE, layout=layout)


@pytest.mark.new_empty
@pytest.mark.parametrize("layout", [torch.sparse_csr, torch.sparse_csc])
def test_new_empty_rejects_1d_compressed_size(layout):
    inp = tu.make_input(torch.float32, _SMALL_SOURCE, _SOURCE_RANGE)

    # batched sparse compressed tensors need at least a 2-D size
    with pytest.raises(RuntimeError):
        flag_gems.new_empty(inp, (4,), layout=layout)


@pytest.mark.new_empty
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_new_empty_special_value_source(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_empty(ref_inp, _TARGET_SIZE)
    res_out = flag_gems.new_empty(inp, _TARGET_SIZE)

    # NaN/Inf source payload is never read, and the source must survive intact
    _check_allocation(res_out, ref_out, inp)
    assert res_out.shape == torch.Size(_TARGET_SIZE)
    tu.assert_result_equal(inp, ref_inp)
