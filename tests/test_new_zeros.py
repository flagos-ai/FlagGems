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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# new_zeros is a factory: the source only supplies the default dtype and device
# of the returned buffer, while size and the keyword overrides describe it.
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

NEW_ZEROS_DTYPES = (
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

# Optional-parameter, call-form and autograd coverage stays in the default suite:
# quick keeps the main grid, the zero-size boundary, the designated .out smoke and
# every negative.
_INHERITS_DTYPE_CASES = tu.selected_cases(NEW_ZEROS_DTYPES, quick=[])
_SIZE_FORMS = tu.selected_cases([(2, 3), torch.Size([2, 3]), [True, 2]], quick=[])
_SOURCE_VARIANT_DTYPES = [torch.float32, torch.int8]
_SOURCE_VARIANT_PARENT = (4, 12)
_SOURCE_VARIANTS = ["contiguous", "noncontiguous", "offset", "expanded"]
_DIFFERENTIABLE_DTYPES = tu.selected_cases([torch.float32] + _FP64_DTYPES, quick=[])
_DENSE_LAYOUT_CASES = tu.selected_cases(
    [{}, {"layout": None}, {"layout": torch.strided}], quick=[]
)
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
_EXPLICIT_DEVICE_CASES = tu.selected_cases(
    [
        (device, dtype)
        for device in _DEVICE_OVERRIDES
        for dtype in _EXPLICIT_DEVICE_DTYPES
    ],
    quick=[],
)

_EXPLICIT_DTYPE_SOURCES = [torch.float16, torch.float32] + _INT64_DTYPES
_EXPLICIT_DTYPES = (
    [torch.float32, torch.int32, torch.uint8, torch.complex64]
    + _BF16_DTYPES
    + _INT64_DTYPES
    + _FP8_DTYPES
)
_EXPLICIT_DTYPE_CASES = tu.selected_cases(
    [(src, out) for src in _EXPLICIT_DTYPE_SOURCES for out in _EXPLICIT_DTYPES],
    quick=[],
)

# A bool extent is not a rejection: the schema coerces [True, 2] to (1, 2).
# Argument-type mismatches are rejected by the aten schema matcher with
# RuntimeError and by Python-level argument parsing with TypeError; out-of-range
# extents are RuntimeError in both layers.
_ARGUMENT_TYPE_ERRORS = (TypeError, RuntimeError)
_SIZE_ERROR_CASES = [
    (5, _ARGUMENT_TYPE_ERRORS),
    ((1.5, 2), _ARGUMENT_TYPE_ERRORS),
    ((2, 3.0), _ARGUMENT_TYPE_ERRORS),
    ((-1,), RuntimeError),
    ((2, -3), RuntimeError),
]

# Sparse factories materialise structural index storage next to the payload.
# Probed on the accelerator: COO keeps int64 indices and CSR/CSC keeps int64
# crow/ccol indices, so layout eligibility follows the backend int64 capability
# while _SPARSE_DTYPES only widens the payload choice.  The output indices stay
# exactly the native ones; they are never narrowed to int32.
_SPARSE_LAYOUTS = (
    [torch.sparse_coo, torch.sparse_csr, torch.sparse_csc]
    if utils.int64_is_supported
    else []
)
_BLOCK_LAYOUTS = [torch.sparse_bsr, torch.sparse_bsc]
_SPARSE_SHAPE = (2, 3)
_SPARSE_DTYPES = [torch.float32] + _INT64_DTYPES

_OUT_DTYPES = [torch.float32, torch.int32]
_OUT_SIZES = tu.selected_cases([(256,), (20, 320, 15)], quick=[(256,)])
_OUT_CROSS_DTYPE_CASES = tu.selected_cases(
    [
        (torch.float32, torch.int32),
        (torch.float16, torch.float32),
        (torch.int32, torch.uint8),
        (torch.float32, torch.complex64),
    ]
    + [(torch.float32, out) for out in _BF16_DTYPES + _FP64_DTYPES + _INT64_DTYPES],
    quick=[],
)
_OUT_RESIZE_BUFFERS = tu.selected_cases([(0,), (1,), (5, 5)], quick=[])
# (row offset of the out view inside its padded parent, requested size).  The
# offset-0 row is the plain strided view, the offset-1 row adds a non-zero
# storage offset.
_OUT_STRIDED_CASES = tu.selected_cases([(0, _TARGET_SIZE), (1, _TARGET_SIZE)], quick=[])

# pin_memory only ever describes host storage: the positive case is a property of
# the requested CPU output and of the pinned allocator the surrounding torch build
# ships (the same allocator the existing pin_memory tests rely on), not of the
# accelerator the source lives on.  The rejection is a separate native rule
# ('Only dense CPU tensors can be pinned') and only meaningful while the
# configured device is not the host.
_PINNED_HOST_DTYPES = tu.selected_cases([torch.float32], quick=[])
_PINNED_DEVICE_REJECTION_DTYPES = [] if _DEVICE_IS_HOST else [torch.float32]


def _source_with_parent(dtype, variant):
    # The parent and a pre-call snapshot are returned next to the view so the
    # test can detect writes that land outside the view but inside its storage.
    parent = tu.make_input(dtype, _SOURCE_VARIANT_PARENT, _SOURCE_RANGE)
    # Reference placement keeps the shared comparison device contract:
    # tu.to_reference clones the storage independently, so the snapshot is a
    # true point-in-time copy under either configured reference device.
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


@pytest.mark.new_zeros
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", NEW_ZEROS_DTYPES)
def test_new_zeros(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_zeros(ref_inp, shape)
    res_out = flag_gems.new_zeros(inp, shape)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    assert res_out.is_contiguous()
    if inp.numel() > 0 and res_out.numel() > 0:
        # a fresh buffer, never an alias of the source storage
        assert res_out.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize(
    "shape", tu.selected_cases(_ZERO_SIZE_SHAPES, quick=[(2, 0, 4)])
)
@pytest.mark.parametrize("dtype", NEW_ZEROS_DTYPES)
def test_new_zeros_zero_size(shape, dtype):
    inp = tu.make_input(dtype, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_zeros(ref_inp, shape)
    res_out = flag_gems.new_zeros(inp, shape)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("size", _SIZE_FORMS)
def test_new_zeros_size_forms(size):
    inp = tu.make_input(torch.float32, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_zeros(ref_inp, size)
    res_out = flag_gems.new_zeros(inp, size)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("dtype", _INHERITS_DTYPE_CASES)
def test_new_zeros_inherits_source_metadata(dtype):
    inp = tu.make_input(dtype, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_zeros(ref_inp, _TARGET_SIZE)
    res_out = flag_gems.new_zeros(inp, _TARGET_SIZE)

    tu.assert_result_equal(res_out, ref_out)
    # with no explicit override the output follows the source dtype and device
    assert res_out.dtype == inp.dtype
    assert res_out.device == inp.device
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("src_dtype,out_dtype", _EXPLICIT_DTYPE_CASES)
def test_new_zeros_explicit_dtype(src_dtype, out_dtype):
    inp = tu.make_input(src_dtype, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_zeros(ref_inp, _TARGET_SIZE, dtype=out_dtype)
    res_out = flag_gems.new_zeros(inp, _TARGET_SIZE, dtype=out_dtype)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.dtype == out_dtype
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("device,dtype", _EXPLICIT_DEVICE_CASES)
def test_new_zeros_explicit_device(device, dtype):
    inp = tu.make_input(dtype, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_zeros(ref_inp, _TARGET_SIZE, device=device)
    res_out = flag_gems.new_zeros(inp, _TARGET_SIZE, device=device)

    # A device override makes the reference result follow the requested device
    # while the shared comparison buffers live in reference placement, so the
    # oracle is placed with the same shared mechanism.  The requested device is
    # asserted on the candidate directly below.
    tu.assert_result_equal(res_out, tu.to_reference(ref_out))
    if torch.device(device).type == inp.device.type:
        assert res_out.device == inp.device
    else:
        assert res_out.device == torch.device(device)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("variant", tu.selected_cases(_SOURCE_VARIANTS, quick=[]))
@pytest.mark.parametrize("dtype", _SOURCE_VARIANT_DTYPES)
def test_new_zeros_source_geometry_is_irrelevant(variant, dtype):
    inp, parent, snapshot = _source_with_parent(dtype, variant)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_zeros(ref_inp, _TARGET_SIZE)
    res_out = flag_gems.new_zeros(inp, _TARGET_SIZE)

    tu.assert_result_equal(res_out, ref_out)
    # The requested size alone defines the output, and the source storage must be
    # untouched: the whole parent is compared against its pre-call snapshot, so a
    # write outside the view but inside its storage is still detected.
    tu.assert_result_equal(parent, snapshot)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("dtype", _DIFFERENTIABLE_DTYPES)
def test_new_zeros_output_is_not_differentiable(dtype):
    inp = tu.make_input(dtype, _SMALL_SOURCE, _SOURCE_RANGE).requires_grad_(True)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_zeros(ref_inp, _TARGET_SIZE)
    res_out = flag_gems.new_zeros(inp, _TARGET_SIZE)

    # the forward contract still holds for a source that tracks gradients
    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    # zeros carry no dependency on the source, so autograd must not track them
    assert not res_out.requires_grad
    assert res_out.grad_fn is None
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("layout_kwargs", _DENSE_LAYOUT_CASES)
def test_new_zeros_dense_layout(layout_kwargs):
    inp = tu.make_input(torch.float32, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_zeros(ref_inp, _TARGET_SIZE, **layout_kwargs)
    res_out = flag_gems.new_zeros(inp, _TARGET_SIZE, **layout_kwargs)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.layout == torch.strided
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("layout", tu.selected_cases(_SPARSE_LAYOUTS, quick=[]))
@pytest.mark.parametrize("dtype", _SPARSE_DTYPES)
def test_new_zeros_sparse_layout(layout, dtype):
    inp = tu.make_input(dtype, _SPARSE_SHAPE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_zeros(ref_inp, _SPARSE_SHAPE, layout=layout)
    res_out = flag_gems.new_zeros(inp, _SPARSE_SHAPE, layout=layout)

    # the exact comparison covers the sparse layout and its index/pointer storage
    tu.assert_result_equal(res_out, ref_out)
    assert res_out.layout == layout
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("size", _OUT_SIZES)
@pytest.mark.parametrize("dtype", _OUT_DTYPES)
def test_new_zeros_out_overload(size, dtype):
    inp = tu.make_input(dtype, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)
    ref_buf = torch.ones(size, dtype=dtype, device=ref_inp.device)
    act_buf = torch.ones(size, dtype=dtype, device=inp.device)

    ref_out = torch.ops.aten.new_zeros.out(ref_inp, size, out=ref_buf)
    res_out = flag_gems.new_zeros(inp, size, out=act_buf)

    tu.assert_result_equal(res_out, ref_out)
    # the supplied (ones-filled) buffer is zeroed in place and returned
    assert res_out is act_buf
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("src_dtype,out_dtype", _OUT_CROSS_DTYPE_CASES)
def test_new_zeros_out_cross_dtype(src_dtype, out_dtype):
    inp = tu.make_input(src_dtype, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)
    ref_buf = torch.ones(_TARGET_SIZE, dtype=out_dtype, device=ref_inp.device)
    act_buf = torch.ones(_TARGET_SIZE, dtype=out_dtype, device=inp.device)

    ref_out = torch.ops.aten.new_zeros.out(ref_inp, _TARGET_SIZE, out=ref_buf)
    res_out = flag_gems.new_zeros(inp, _TARGET_SIZE, out=act_buf)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out is act_buf
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("buffer_shape", _OUT_RESIZE_BUFFERS)
def test_new_zeros_out_resizes_buffer(buffer_shape):
    inp = tu.make_input(torch.float32, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)
    ref_buf = torch.ones(buffer_shape, dtype=torch.float32, device=ref_inp.device)
    act_buf = torch.ones(buffer_shape, dtype=torch.float32, device=inp.device)

    ref_out = torch.ops.aten.new_zeros.out(ref_inp, _TARGET_SIZE, out=ref_buf)
    res_out = flag_gems.new_zeros(inp, _TARGET_SIZE, out=act_buf)

    tu.assert_result_equal(res_out, ref_out)
    # the supplied buffer is adapted to the requested size and returned, not
    # replaced by a freshly allocated tensor
    assert res_out is act_buf
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("row_offset,size", _OUT_STRIDED_CASES)
def test_new_zeros_out_strided_buffer(row_offset, size):
    inp = tu.make_input(torch.float32, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    rows, cols = size
    ref_parent = torch.full(
        (rows + 1, cols + _PAD), 7.0, dtype=torch.float32, device=ref_inp.device
    )
    act_parent = torch.full(
        (rows + 1, cols + _PAD), 7.0, dtype=torch.float32, device=inp.device
    )
    ref_buf = ref_parent[row_offset : row_offset + rows, :cols]
    act_buf = act_parent[row_offset : row_offset + rows, :cols]

    ref_out = torch.ops.aten.new_zeros.out(ref_inp, size, out=ref_buf)
    res_out = flag_gems.new_zeros(inp, size, out=act_buf)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out is act_buf
    assert act_buf.storage_offset() == row_offset * (cols + _PAD)
    assert not act_buf.is_contiguous()
    # The view must keep writing into its parent storage: a candidate that
    # swapped in fresh storage would leave the parent at its sentinels, so the
    # whole parent is compared against the reference parent (zeros inside the
    # requested view, sentinels in the padding columns and the untouched rows).
    assert (
        act_buf.untyped_storage().data_ptr() == act_parent.untyped_storage().data_ptr()
    )
    tu.assert_result_equal(act_parent, ref_parent)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("dtype", _PINNED_HOST_DTYPES)
def test_new_zeros_host_pinned_output(dtype):
    inp = tu.make_input(dtype, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_zeros(
        ref_inp, _TARGET_SIZE, device=torch.device("cpu"), pin_memory=True
    )
    res_out = flag_gems.new_zeros(
        inp, _TARGET_SIZE, device=torch.device("cpu"), pin_memory=True
    )

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == torch.device("cpu")
    assert res_out.is_pinned()
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("dtype", _PINNED_DEVICE_REJECTION_DTYPES)
def test_new_zeros_rejects_pinned_device_output(dtype):
    inp = tu.make_input(dtype, _SMALL_SOURCE, _SOURCE_RANGE)

    with pytest.raises(RuntimeError):
        flag_gems.new_zeros(inp, _TARGET_SIZE, device=inp.device, pin_memory=True)


@pytest.mark.new_zeros
@pytest.mark.parametrize("size,error", _SIZE_ERROR_CASES)
def test_new_zeros_rejects_invalid_size(size, error):
    inp = tu.make_input(torch.float32, _SMALL_SOURCE, _SOURCE_RANGE)

    with pytest.raises(error):
        flag_gems.new_zeros(inp, size)


@pytest.mark.new_zeros
def test_new_zeros_rejects_invalid_dtype():
    inp = tu.make_input(torch.float32, _SMALL_SOURCE, _SOURCE_RANGE)

    with pytest.raises(_ARGUMENT_TYPE_ERRORS):
        flag_gems.new_zeros(inp, _TARGET_SIZE, dtype="not_a_dtype")


@pytest.mark.new_zeros
@pytest.mark.parametrize("layout", _BLOCK_LAYOUTS)
def test_new_zeros_rejects_block_sparse_layout(layout):
    inp = tu.make_input(torch.float32, _SMALL_SOURCE, _SOURCE_RANGE)

    # the compressed empty-tensor helper only accepts non-block layouts
    with pytest.raises(RuntimeError):
        flag_gems.new_zeros(inp, _SPARSE_SHAPE, layout=layout)


@pytest.mark.new_zeros
@pytest.mark.parametrize("layout", [torch.sparse_csr, torch.sparse_csc])
def test_new_zeros_rejects_1d_compressed_size(layout):
    inp = tu.make_input(torch.float32, _SMALL_SOURCE, _SOURCE_RANGE)

    # batched sparse compressed tensors need at least a 2-D size
    with pytest.raises(RuntimeError):
        flag_gems.new_zeros(inp, (4,), layout=layout)


@pytest.mark.new_zeros
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_new_zeros_special_value_source(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_zeros(ref_inp, _TARGET_SIZE)
    res_out = flag_gems.new_zeros(inp, _TARGET_SIZE)

    # NaN/Inf source contents must not leak into the freshly zeroed output
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
