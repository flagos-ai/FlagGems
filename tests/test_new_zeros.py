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


def _cpu_values(inp):
    # Preserve lazy conjugate/negative values even when a typed XPU transfer
    # drops a lazy flag without applying its value transformation.
    nbytes = inp.untyped_storage().nbytes()
    raw = torch.empty(0, dtype=torch.uint8, device=inp.device).set_(
        inp.untyped_storage(), 0, (nbytes,), (1,)
    )
    raw = raw.to("cpu").clone()
    out = torch.empty(0, dtype=inp.dtype, device="cpu").set_(
        raw.untyped_storage(), inp.storage_offset(), inp.size(), inp.stride()
    )
    if inp.is_conj():
        out = out.conj()
    if inp.is_neg():
        out = torch._neg_view(out)
    assert out.dtype == inp.dtype and out.shape == inp.shape
    assert out.stride() == inp.stride() and out.storage_offset() == inp.storage_offset()
    assert (out.is_conj(), out.is_neg()) == (inp.is_conj(), inp.is_neg())
    return out


def _assert_result_equal(result, reference, *, device=None):
    # Validate the original ABI before copying any values for observation.
    assert result.dtype == reference.dtype
    assert result.shape == reference.shape and result.layout == reference.layout
    expected_device = torch.device(flag_gems.device if device is None else device)
    if expected_device.index is None and expected_device.type not in ("cpu", "meta"):
        expected_device = torch.device(
            expected_device.type, flag_gems.runtime.torch_device_fn.current_device()
        )
    assert result.device == expected_device
    assert reference.device == (
        torch.device("cpu") if utils.TO_CPU else expected_device
    )
    if flag_gems.vendor_name == "kunlunxin" and (
        result.is_complex()
        or result.is_neg()
        or result.is_conj()
        or reference.is_neg()
        or reference.is_conj()
    ):
        result, reference = _cpu_values(result), _cpu_values(reference)
    elif flag_gems.vendor_name in ("ascend", "mthreads") and result.is_complex():
        result, reference = result.cpu(), reference.cpu()
    tu.assert_result_equal(result, reference)


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


def _skip_unsupported_cpu_fp8_zero_shape(dtype, shape):
    # Older CPU runtimes retain a small nonempty FP8 zero path, but these
    # original grid shapes dispatch to an unavailable fill_cpu kernel.
    if (
        flag_gems.vendor_name == "hygon"
        and utils.TO_CPU
        and dtype in _FP8_DTYPES
        and tuple(shape)
        in {
            (1024, 1024),
            (20, 320, 15),
            (16, 128, 64, 60),
            (16, 7, 57, 32, 29),
            (0,),
            (2, 0, 4),
            (0, 0),
        }
    ):
        pytest.skip("Hygon CPU new_zeros has no FP8 fill for this shape")


def _out_sentinel(shape, dtype, device):
    if flag_gems.vendor_name == "kunlunxin" and dtype.is_complex:
        # Preserve the original all-one sentinel where device full is absent.
        return torch.full(shape, 1, dtype=dtype, device="cpu").to(device)
    return torch.full(shape, 1, dtype=dtype, device=device)


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
    [
        pytest.param(
            src,
            out,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name == "ascend" and out in _FP8_DTYPES,
                reason="Ascend native new_zeros cannot fill FP8 output payloads",
            ),
        )
        for src in _EXPLICIT_DTYPE_SOURCES
        for out in _EXPLICIT_DTYPES
    ],
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


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    pytest.param(
        dtype,
        value_range,
        marks=pytest.mark.skipif(
            flag_gems.vendor_name == "ascend"
            and dtype in (torch.float8_e4m3fn, torch.float8_e5m2),
            reason="Ascend cannot construct or compare FP8 payloads with device kernels",
        ),
    )
    for dtype in NEW_ZEROS_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8
    or value_range not in (["0", "1"], ["-1", "0"], ["min", "0"])
]


@pytest.mark.new_zeros
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(
    "dtype,value_range",
    _DTYPE_RANGE_CASES,
)
def test_new_zeros(shape, value_range, dtype):
    _skip_unsupported_cpu_fp8_zero_shape(dtype, shape)
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_zeros(ref_inp, shape)
    res_out = flag_gems.new_zeros(inp, shape)

    _assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    assert res_out.is_contiguous()
    if inp.numel() > 0 and res_out.numel() > 0:
        # a fresh buffer, never an alias of the source storage
        assert res_out.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()
    _assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize(
    "shape", tu.selected_cases(_ZERO_SIZE_SHAPES, quick=[(2, 0, 4)])
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            dtype,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name == "ascend"
                and dtype in (torch.float8_e4m3fn, torch.float8_e5m2),
                reason="Ascend cannot construct or compare FP8 payloads with device kernels",
            ),
        )
        for dtype in NEW_ZEROS_DTYPES
    ],
)
def test_new_zeros_zero_size(shape, dtype):
    _skip_unsupported_cpu_fp8_zero_shape(dtype, shape)
    inp = tu.make_input(dtype, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_zeros(ref_inp, shape)
    res_out = flag_gems.new_zeros(inp, shape)

    _assert_result_equal(res_out, ref_out)
    _assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("size", _SIZE_FORMS)
def test_new_zeros_size_forms(size):
    inp = tu.make_input(torch.float32, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_zeros(ref_inp, size)
    res_out = flag_gems.new_zeros(inp, size)

    _assert_result_equal(res_out, ref_out)
    _assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            dtype,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name == "ascend"
                and dtype in (torch.float8_e4m3fn, torch.float8_e5m2),
                reason="Ascend cannot construct or compare FP8 payloads with device kernels",
            ),
        )
        for dtype in _INHERITS_DTYPE_CASES
    ],
)
def test_new_zeros_inherits_source_metadata(dtype):
    inp = tu.make_input(dtype, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_zeros(ref_inp, _TARGET_SIZE)
    res_out = flag_gems.new_zeros(inp, _TARGET_SIZE)

    _assert_result_equal(res_out, ref_out)
    # with no explicit override the output follows the source dtype and device
    assert res_out.dtype == inp.dtype
    assert res_out.device == inp.device
    _assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("src_dtype,out_dtype", _EXPLICIT_DTYPE_CASES)
def test_new_zeros_explicit_dtype(src_dtype, out_dtype):
    inp = tu.make_input(src_dtype, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_zeros(ref_inp, _TARGET_SIZE, dtype=out_dtype)
    res_out = flag_gems.new_zeros(inp, _TARGET_SIZE, dtype=out_dtype)

    _assert_result_equal(res_out, ref_out)
    assert res_out.dtype == out_dtype
    _assert_result_equal(inp, ref_inp)


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
    _assert_result_equal(res_out, tu.to_reference(ref_out), device=device)
    if torch.device(device).type == inp.device.type:
        assert res_out.device == inp.device
    else:
        assert res_out.device == torch.device(device)
    _assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("variant", tu.selected_cases(_SOURCE_VARIANTS, quick=[]))
@pytest.mark.parametrize("dtype", _SOURCE_VARIANT_DTYPES)
def test_new_zeros_source_geometry_is_irrelevant(variant, dtype):
    inp, parent, snapshot = _source_with_parent(dtype, variant)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_zeros(ref_inp, _TARGET_SIZE)
    res_out = flag_gems.new_zeros(inp, _TARGET_SIZE)

    _assert_result_equal(res_out, ref_out)
    # The requested size alone defines the output, and the source storage must be
    # untouched: the whole parent is compared against its pre-call snapshot, so a
    # write outside the view but inside its storage is still detected.
    _assert_result_equal(parent, snapshot)
    _assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("dtype", _DIFFERENTIABLE_DTYPES)
def test_new_zeros_output_is_not_differentiable(dtype):
    inp = tu.make_input(dtype, _SMALL_SOURCE, _SOURCE_RANGE).requires_grad_(True)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_zeros(ref_inp, _TARGET_SIZE)
    res_out = flag_gems.new_zeros(inp, _TARGET_SIZE)

    # the forward contract still holds for a source that tracks gradients
    _assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    # zeros carry no dependency on the source, so autograd must not track them
    assert not res_out.requires_grad
    assert res_out.grad_fn is None
    _assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("layout_kwargs", _DENSE_LAYOUT_CASES)
def test_new_zeros_dense_layout(layout_kwargs):
    inp = tu.make_input(torch.float32, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_zeros(ref_inp, _TARGET_SIZE, **layout_kwargs)
    res_out = flag_gems.new_zeros(inp, _TARGET_SIZE, **layout_kwargs)

    _assert_result_equal(res_out, ref_out)
    assert res_out.layout == torch.strided
    _assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("layout", tu.selected_cases(_SPARSE_LAYOUTS, quick=[]))
@pytest.mark.parametrize("dtype", _SPARSE_DTYPES)
def test_new_zeros_sparse_layout(layout, dtype):
    inp = tu.make_input(dtype, _SPARSE_SHAPE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_zeros(ref_inp, _SPARSE_SHAPE, layout=layout)
    res_out = flag_gems.new_zeros(inp, _SPARSE_SHAPE, layout=layout)

    # the exact comparison covers the sparse layout and its index/pointer storage
    _assert_result_equal(res_out, ref_out)
    assert res_out.layout == layout
    _assert_result_equal(inp, ref_inp)


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

    _assert_result_equal(res_out, ref_out)
    # the supplied (ones-filled) buffer is zeroed in place and returned
    assert res_out is act_buf
    _assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("src_dtype,out_dtype", _OUT_CROSS_DTYPE_CASES)
def test_new_zeros_out_cross_dtype(src_dtype, out_dtype):
    inp = tu.make_input(src_dtype, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)
    ref_buf = _out_sentinel(_TARGET_SIZE, out_dtype, ref_inp.device)
    act_buf = _out_sentinel(_TARGET_SIZE, out_dtype, inp.device)

    ref_out = torch.ops.aten.new_zeros.out(ref_inp, _TARGET_SIZE, out=ref_buf)
    res_out = flag_gems.new_zeros(inp, _TARGET_SIZE, out=act_buf)

    _assert_result_equal(res_out, ref_out)
    assert res_out is act_buf
    _assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("buffer_shape", _OUT_RESIZE_BUFFERS)
def test_new_zeros_out_resizes_buffer(buffer_shape):
    inp = tu.make_input(torch.float32, _SMALL_SOURCE, _SOURCE_RANGE)
    ref_inp = tu.to_reference(inp)
    ref_buf = torch.ones(buffer_shape, dtype=torch.float32, device=ref_inp.device)
    act_buf = torch.ones(buffer_shape, dtype=torch.float32, device=inp.device)

    ref_out = torch.ops.aten.new_zeros.out(ref_inp, _TARGET_SIZE, out=ref_buf)
    res_out = flag_gems.new_zeros(inp, _TARGET_SIZE, out=act_buf)

    _assert_result_equal(res_out, ref_out)
    # the supplied buffer is adapted to the requested size and returned, not
    # replaced by a freshly allocated tensor
    assert res_out is act_buf
    _assert_result_equal(inp, ref_inp)


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

    _assert_result_equal(res_out, ref_out)
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
    _assert_result_equal(act_parent, ref_parent)
    _assert_result_equal(inp, ref_inp)


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

    _assert_result_equal(res_out, ref_out, device=torch.device("cpu"))
    assert res_out.device == torch.device("cpu")
    assert res_out.is_pinned()
    _assert_result_equal(inp, ref_inp)


@pytest.mark.new_zeros
@pytest.mark.parametrize("dtype", _PINNED_DEVICE_REJECTION_DTYPES)
def test_new_zeros_rejects_pinned_device_output(dtype):
    inp = tu.make_input(dtype, _SMALL_SOURCE, _SOURCE_RANGE)

    if flag_gems.vendor_name == "mthreads":
        kwargs = {"device": inp.device, "pin_memory": True}
        expected = torch.ops.aten.new_zeros(inp, _TARGET_SIZE, **kwargs)
        actual = flag_gems.new_zeros(inp, _TARGET_SIZE, **kwargs)

        assert actual.device == expected.device == inp.device
        assert actual.is_pinned() == expected.is_pinned()
        _assert_result_equal(actual, tu.to_reference(expected))
        return

    error = (
        (RuntimeError, ValueError)
        if flag_gems.vendor_name == "kunlunxin"
        else RuntimeError
    )
    with pytest.raises(error):
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
@pytest.mark.parametrize(
    "dtype,scenario",
    [
        pytest.param(
            dtype,
            scenario,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name == "ascend"
                and dtype in (torch.float8_e4m3fn, torch.float8_e5m2),
                reason="Ascend cannot construct or compare FP8 payloads with device kernels",
            ),
        )
        for dtype, scenario in _SPECIAL_CASES
    ],
)
def test_new_zeros_special_value_source(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.new_zeros(ref_inp, _TARGET_SIZE)
    res_out = flag_gems.new_zeros(inp, _TARGET_SIZE)

    # NaN/Inf source contents must not leak into the freshly zeroed output
    _assert_result_equal(res_out, ref_out)
    _assert_result_equal(inp, ref_inp)


_MASTER_SIZES = [(), (0,), (1,), (256,), (1024, 1024), (16, 7, 57), (16, 128, 64, 60)]
_MASTER_DTYPES = list(dict.fromkeys(
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
    + _BF16_DTYPES + _FP64_DTYPES + _INT64_DTYPES
))


@pytest.mark.new_zeros
@pytest.mark.parametrize("size", tu.selected_cases(_MASTER_SIZES, quick=[]))
@pytest.mark.parametrize("dtype", _MASTER_DTYPES)
def test_new_zeros_master_size_and_dtype(size, dtype):
    inp = torch.empty(1, dtype=dtype, device=flag_gems.device)
    expected = torch.ops.aten.new_zeros(tu.to_reference(inp), list(size))
    actual = flag_gems.new_zeros(inp, list(size))

    _assert_result_equal(actual, expected)
    assert actual.dtype == inp.dtype and actual.device == inp.device


@pytest.mark.new_zeros
@pytest.mark.parametrize("size", tu.selected_cases(_MASTER_SIZES, quick=[]))
def test_new_zeros_master_inherits_metadata(size):
    inp = torch.empty((2, 3), dtype=torch.float32, device=flag_gems.device)
    expected = torch.ops.aten.new_zeros(tu.to_reference(inp), list(size))
    actual = flag_gems.new_zeros(inp, list(size))

    _assert_result_equal(actual, expected)
    assert actual.layout == inp.layout


@pytest.mark.new_zeros
@pytest.mark.parametrize("source_shape", [(), (1,), (4, 5)])
def test_new_zeros_master_ignores_source_payload(source_shape):
    inp = torch.full(source_shape, float("nan"), device=flag_gems.device)
    reference = tu.to_reference(inp)
    expected = torch.ops.aten.new_zeros(reference, [2, 3])
    actual = flag_gems.new_zeros(inp, [2, 3])

    _assert_result_equal(actual, expected)
    _assert_result_equal(inp, reference)


@pytest.mark.new_zeros
@pytest.mark.parametrize("dtype", _MASTER_DTYPES)
def test_new_zeros_master_dtype_override(dtype):
    inp = torch.empty(2, device=flag_gems.device)
    expected = torch.ops.aten.new_zeros(tu.to_reference(inp), [4, 2], dtype=dtype)
    actual = flag_gems.new_zeros(inp, [4, 2], dtype=dtype)

    _assert_result_equal(actual, expected)
    assert actual.dtype == dtype


@pytest.mark.new_zeros
@pytest.mark.parametrize("size", [(), (0,), (1,), (2, 3), (16, 128, 64, 60)])
def test_new_zeros_master_explicit_device(size):
    inp = torch.empty((2, 3), device=flag_gems.device)
    expected = torch.ops.aten.new_zeros(tu.to_reference(inp), list(size))
    actual = flag_gems.new_zeros(inp, list(size), device=inp.device)

    _assert_result_equal(actual, expected, device=inp.device)


@pytest.mark.new_zeros
@pytest.mark.parametrize("form", [torch.Size, list, tuple, torch.tensor])
@pytest.mark.parametrize("size", [(), (0,), (1,), (2, 3)])
def test_new_zeros_master_size_forms(form, size):
    inp = torch.empty((2, 3), device=flag_gems.device)
    expected = torch.ops.aten.new_zeros(tu.to_reference(inp), list(size))
    size_arg = torch.tensor(size, dtype=torch.int64) if form is torch.tensor else form(size)
    actual = flag_gems.new_zeros(inp, size_arg)

    _assert_result_equal(actual, expected)


@pytest.mark.new_zeros
def test_new_zeros_master_false_extent():
    inp = torch.empty(2, device=flag_gems.device)
    expected = torch.ops.aten.new_zeros(tu.to_reference(inp), [False, 2])
    actual = flag_gems.new_zeros(inp, [False, 2])

    _assert_result_equal(actual, expected)


@pytest.mark.new_zeros
def test_new_zeros_master_true_extent():
    inp = torch.empty(2, device=flag_gems.device)
    expected = torch.ops.aten.new_zeros(tu.to_reference(inp), [True, 2])
    actual = flag_gems.new_zeros(inp, [True, 2])

    _assert_result_equal(actual, expected)


@pytest.mark.new_zeros
@pytest.mark.parametrize("size", [(), (0,), (2, 3), (1024, 1024)])
def test_new_zeros_master_out(size):
    inp = torch.empty(2, device=flag_gems.device)
    out = torch.full(size, 7.0, device=inp.device)
    reference = tu.to_reference(inp)
    ref_out = torch.full(size, 7.0, device=reference.device)
    expected = torch.ops.aten.new_zeros.out(reference, list(size), out=ref_out)
    actual = flag_gems.new_zeros(inp, list(size), out=out)

    assert actual is out
    _assert_result_equal(actual, expected)


@pytest.mark.new_zeros
def test_new_zeros_master_empty_source():
    inp = torch.empty(0, dtype=torch.int32, device=flag_gems.device)
    expected = torch.ops.aten.new_zeros(tu.to_reference(inp), [2, 3])
    actual = flag_gems.new_zeros(inp, [2, 3])

    _assert_result_equal(actual, expected)
    assert actual.dtype == inp.dtype


@pytest.mark.new_zeros
@pytest.mark.parametrize("bad_arg", ["self", "size"])
def test_new_zeros_master_invalid_schema(bad_arg):
    inp = torch.empty(2, device=flag_gems.device)
    args = ([1, 2, 3], [2]) if bad_arg == "self" else (inp, 3.14)
    with pytest.raises((RuntimeError, TypeError, ValueError, AttributeError)):
        flag_gems.new_zeros(*args)
