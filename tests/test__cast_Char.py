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

"""Correctness tests for ``aten::_cast_Char(self, non_blocking=False) -> Tensor``.

The reference and every expected value come from ``torch.ops.aten._cast_Char``;
the active-device native conversion supplies the expectation and its result is
always ``torch.int8``, so the comparison is exact and carries no tolerance.
Out-of-range and non-finite conversions are left to the native operator and are
never hard-coded here.

An int8 operand comes back as the very same tensor object - same storage,
storage offset, stride and shape - for contiguous, transposed, strided, offset,
channels-last and empty inputs.

The spec's ``[-1, 1)`` range truncates to zero for every integer dtype, so the
layout, identity, empty-view and parameter families use a deterministic
position-varying fixture with ``|value| >= 1.25`` instead: a candidate that
returns zeros, clamps, or reads the wrong stride, storage offset or parameter
value stays observably wrong there.

Exempt dimensions, verified against the native operator: there is no ``.out``
overload, the operator takes exactly one tensor operand and therefore cannot
broadcast, and an integral result carries no autograd history, so there is no
backward family.
"""

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_CAST_DTYPE = torch.int8


def _copy_dense_to(inp, device):
    # Copy storage independently and retain the view before observing values.
    raw = (
        torch.empty(0, dtype=torch.uint8, device=inp.device)
        .set_(inp.untyped_storage(), 0, (inp.untyped_storage().nbytes(),), (1,))
        .to(device)
        .clone()
    )
    result = torch.empty(0, dtype=inp.dtype, device=device).set_(
        raw.untyped_storage(), inp.storage_offset(), inp.size(), inp.stride()
    )
    if inp.is_conj():
        result = result.conj()
    if inp.is_neg():
        result = torch._neg_view(result)
    result.requires_grad_(inp.requires_grad)
    assert result.dtype == inp.dtype and result.shape == inp.shape
    assert (
        result.stride() == inp.stride()
        and result.storage_offset() == inp.storage_offset()
    )
    assert (result.is_conj(), result.is_neg()) == (inp.is_conj(), inp.is_neg())
    expected_device = torch.device(device)
    assert result.device.type == expected_device.type
    if expected_device.index is not None:
        assert result.device == expected_device
    if inp.device == result.device and inp.untyped_storage().nbytes():
        assert inp.untyped_storage().data_ptr() != result.untyped_storage().data_ptr()
    return result


def _skip_native_input(dtype, shape):
    nonempty = all(extent != 0 for extent in shape)
    if flag_gems.vendor_name == "ascend" and dtype in _FP8_DTYPES and nonempty:
        pytest.skip("Ascend _cast_Char has no native nonempty FP8 conversion")


def _expected_device():
    device = torch.device(flag_gems.device)
    if device.index is None and device.type not in ("cpu", "meta"):
        device = torch.device(
            device.type, flag_gems.runtime.torch_device_fn.current_device()
        )
    return device


def _check_cast_abi(result, reference, inp):
    assert inp.device == result.device == _expected_device()
    assert result.dtype == reference.dtype == _CAST_DTYPE
    assert result.shape == reference.shape == inp.shape
    assert result.layout == reference.layout == inp.layout
    assert reference.device == (torch.device("cpu") if utils.TO_CPU else inp.device)


def _assert_cast_equal(result, reference, inp):
    # Reject an incorrect original device or ABI before numerical CPU copies.
    _check_cast_abi(result, reference, inp)
    tu.assert_result_equal(result, reference)


def _reference(inp):
    assert inp.device == _expected_device()
    _skip_native_input(inp.dtype, inp.shape)
    reference = (
        _copy_dense_to(inp, "cpu")
        if utils.TO_CPU and inp.layout == torch.strided
        else tu.to_reference(inp)
    )
    # Tensor.to has no portable floating-to-integer value outside this domain.
    # Inspect only the input, never the candidate or the observed discrepancy.
    if utils.TO_CPU:
        values = (
            reference._values() if reference.layout == torch.sparse_coo else reference
        )
        if values.numel() and (
            values.dtype.is_floating_point or values.dtype.is_complex
        ):
            if values.is_complex():
                values = values.real
            values = values.to(torch.float64)
            low, high = float(values.amin()), float(values.amax())
            bounds = torch.iinfo(_CAST_DTYPE)
            if (
                not (math.isfinite(low) and math.isfinite(high))
                or math.trunc(low) < bounds.min
                or math.trunc(high) > bounds.max
            ):
                pytest.skip(
                    "CPU/device floating-to-integer comparison has no portable value "
                    "for nonfinite or truncated out-of-range inputs"
                )
    return reference


def _make_input(dtype, shape, value_range):
    _skip_native_input(dtype, shape)
    return tu.make_input(dtype, shape, value_range)


def _make_special_input(dtype, scenario):
    _skip_native_input(dtype, (5,))
    return tu.make_special_input(dtype, scenario)


_BF16 = utils.bf16_is_supported
_INT64 = utils.int64_is_supported
_FP8 = utils.fp8_is_supported
_FP64 = utils.fp64_is_supported

_REQUIRED_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
)


def _supported(dtype):
    """Whether the device statically advertises kernels for ``dtype``.

    Only the ``accuracy_utils`` capability flags are read here, so no dtype is
    probed at import or collection time.
    """
    if dtype == torch.bfloat16:
        return _BF16
    if dtype == torch.int64:
        return _INT64
    if dtype == torch.float64:
        return _FP64
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return _FP8
    return True


def _dtypes(*extra):
    """The required dtypes plus ``extra``, minus the unsupported ones."""
    dtypes = []
    for dtype in _REQUIRED_DTYPES + extra:
        if dtype not in dtypes and _supported(dtype):
            dtypes.append(dtype)
    return tuple(dtypes)


_GRID_DTYPES = _dtypes(torch.int16, torch.bool)
# Every bool payload is true, so a bool fixture could not distinguish a wrong
# stride, offset or parameter value; the fixture families start after bool.
_FIXTURE_DTYPES = _dtypes(torch.int16)

# Position-varying payload with alternating signs and ``|value| >= 1.25``. The
# native cast leaves every entry a non-zero integer, so a candidate that zeroes,
# clamps or misreads the stride/storage offset is observable. FP8 storage rounds
# much of the payload, so the fixture does not claim distinct stored values
# there; both sides read the identical tensor and the layout and identity rows
# carry the stride/offset sensitivity.
_FIXTURE_VALUES = (
    1.25,
    -2.75,
    3.5,
    -4.25,
    5.75,
    -6.5,
    7.25,
    -8.75,
    9.5,
    -10.25,
    11.75,
    -12.5,
    13.25,
    -14.75,
    15.5,
    -16.25,
    17.75,
    -18.5,
    19.25,
    -20.75,
    21.5,
    -22.25,
    23.75,
    -24.5,
    25.25,
    -26.75,
    27.5,
    -28.25,
    29.75,
    -30.5,
    31.25,
    -32.75,
)


def _cpu_fixture(dtype, count):
    values = _FIXTURE_VALUES
    if dtype == torch.uint8:
        values = tuple(abs(value) for value in values)
    elif not dtype.is_floating_point:
        values = tuple(int(value) for value in values)
    repeats = -(-count // len(values))
    return (
        torch.tensor(values, dtype=torch.float32, device="cpu")
        .repeat(repeats)[:count]
        .to(dtype)
    )


def _fixture_tensor(dtype, count):
    """A deterministic non-zero payload for ``dtype`` of exactly ``count`` values."""
    if flag_gems.vendor_name == "ascend" and dtype in _FP8_DTYPES:
        return _cpu_fixture(dtype, count).to(flag_gems.device)
    values = _FIXTURE_VALUES
    if dtype == torch.uint8:
        values = tuple(abs(value) for value in values)
    elif not dtype.is_floating_point:
        # int() truncates toward zero, keeping the alternating signs.
        values = tuple(int(value) for value in values)
    pattern = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    repeats = -(-count // pattern.numel())
    return pattern.repeat(repeats)[:count]


def _fixture(dtype, shape):
    count = 1
    for dim in shape:
        count *= dim
    return _fixture_tensor(dtype, count).reshape(shape)


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _GRID_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8
    or value_range not in (["0", "1"], ["-1", "0"], ["min", "0"])
]


@pytest.mark.underscore_cast_Char
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype,value_range", _DTYPE_RANGE_CASES)
def test__cast_Char(dtype, shape, value_range):
    torch.manual_seed(0)
    inp = _make_input(dtype, shape, value_range)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten._cast_Char(ref_inp)
    res_out = flag_gems._cast_Char(inp)

    _assert_cast_equal(res_out, ref_out, inp)


_LAYOUTS = ("transpose", "channels_last", "stride_slice", "offset_slice")
_LAYOUT_WORKLOADS = tu.selected_cases(_LAYOUTS, quick=[])


def _layout_input(layout, dtype):
    if layout == "transpose":
        return _fixture(dtype, (8, 10)).transpose(0, 1)
    if layout == "channels_last":
        if flag_gems.vendor_name == "ascend":
            cpu = _cpu_fixture(dtype, 256).reshape(2, 8, 4, 4)
            cpu = cpu.to(torch.float64).to(memory_format=torch.channels_last).to(dtype)
            return _copy_dense_to(cpu, flag_gems.device)
        return _fixture(dtype, (2, 8, 4, 4)).to(memory_format=torch.channels_last)
    if layout == "stride_slice":
        return _fixture(dtype, (8, 24))[:, 1::3]
    return _fixture(dtype, (8, 5)).reshape(-1)[4:36]


@pytest.mark.underscore_cast_Char
@pytest.mark.parametrize("layout", _LAYOUT_WORKLOADS)
@pytest.mark.parametrize("dtype", _FIXTURE_DTYPES)
def test__cast_Char_view_layouts(dtype, layout):
    inp = _layout_input(layout, dtype)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten._cast_Char(ref_inp)
    res_out = flag_gems._cast_Char(inp)

    _assert_cast_equal(res_out, ref_out, inp)


# Same-dtype (int8) forms, where the native operator aliases instead of copying.
_IDENTITY_FORMS = (
    "contiguous",
    "transpose",
    "stride_slice",
    "offset_slice",
    "channels_last",
    "empty_view",
    "scalar",
)
_IDENTITY_WORKLOADS = tu.selected_cases(_IDENTITY_FORMS, quick=[])


def _identity_input(form):
    if form == "contiguous":
        return _fixture(torch.int8, (8, 10))
    if form == "transpose":
        return _fixture(torch.int8, (8, 10)).transpose(0, 1)
    if form == "stride_slice":
        return _fixture(torch.int8, (8, 24))[:, 1::3]
    if form == "offset_slice":
        return _fixture(torch.int8, (8, 5)).reshape(-1)[4:36]
    if form == "channels_last":
        return _layout_input("channels_last", torch.int8)
    if form == "empty_view":
        return _fixture(torch.int8, (0, 10)).transpose(0, 1)
    return _fixture(torch.int8, ()).reshape(())


@pytest.mark.underscore_cast_Char
@pytest.mark.parametrize("form", _IDENTITY_WORKLOADS)
def test__cast_Char_int8_identity(form):
    inp = _identity_input(form)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten._cast_Char(ref_inp)
    res_out = flag_gems._cast_Char(inp)

    _assert_cast_equal(res_out, ref_out, inp)
    # Native hands back the int8 operand itself, so a de-aliased copy is not
    # equivalent even when its values, stride and storage offset match.
    assert res_out is inp


_EMPTY_EXTENT_SHAPES = tu.selected_cases(((0,), (0, 3), (3, 0), (2, 0, 5)), quick=[])


@pytest.mark.underscore_cast_Char
@pytest.mark.parametrize("shape", _EMPTY_EXTENT_SHAPES)
@pytest.mark.parametrize("dtype", _GRID_DTYPES)
def test__cast_Char_empty_extent(dtype, shape):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten._cast_Char(ref_inp)
    res_out = flag_gems._cast_Char(inp)

    _assert_cast_equal(res_out, ref_out, inp)


_EMPTY_VIEWS = ("rows", "cols", "block", "transposed")
_EMPTY_VIEW_WORKLOADS = tu.selected_cases(_EMPTY_VIEWS, quick=[])


def _empty_view_input(kind, dtype):
    if kind == "rows":
        return _fixture(dtype, (0, 8))
    if kind == "cols":
        return _fixture(dtype, (8, 0))
    if kind == "block":
        return _fixture(dtype, (4, 6))[2:2, :]
    return _fixture(dtype, (0, 8)).transpose(0, 1)


@pytest.mark.underscore_cast_Char
@pytest.mark.parametrize("kind", _EMPTY_VIEW_WORKLOADS)
@pytest.mark.parametrize("dtype", _FIXTURE_DTYPES)
def test__cast_Char_empty_view(dtype, kind):
    inp = _empty_view_input(kind, dtype)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten._cast_Char(ref_inp)
    res_out = flag_gems._cast_Char(inp)

    _assert_cast_equal(res_out, ref_out, inp)


# Payloads at the int8 wrap boundary, at the float infinity edge and over the
# full integer range. The expected result is the native cast of the same payload
# in the requested dtype; nothing is clamped or widened here.
_FLOAT_BOUNDARY = (0.0, 0.5, -0.5, 127.0, 128.0, -128.0, -129.0, 300.5, -300.5, -1e-45)
_FLOAT_EXTREME = (3.4e38, -3.4e38, float("inf"), float("-inf"), 1e-45, -1e-45)
_INT_BOUNDARY = (0, 127, 128, -128, -129, 255, 256, -256, 32767, -32768)
_INT_WIDE_BOUNDARY = (
    0,
    127,
    128,
    -128,
    -129,
    1000,
    -1000,
    2**31 - 1,
    -(2**31),
    -300,
    300,
)
_UINT8_BOUNDARY = (0, 127, 128, 200, 255)
_INT8_BOUNDARY = (-128, -1, 0, 1, 127)
_BOOL_BOUNDARY = (True, False, True, True, False)


def _boundary_rows():
    rows = [
        pytest.param(torch.int8, _INT8_BOUNDARY, id="int8"),
        pytest.param(torch.uint8, _UINT8_BOUNDARY, id="uint8"),
        pytest.param(torch.bool, _BOOL_BOUNDARY, id="bool"),
        pytest.param(torch.int16, _INT_BOUNDARY, id="int16"),
        pytest.param(torch.int32, _INT_WIDE_BOUNDARY, id="int32"),
        pytest.param(torch.float16, _FLOAT_BOUNDARY, id="float16"),
        pytest.param(torch.float32, _FLOAT_BOUNDARY, id="float32"),
        pytest.param(torch.float32, _FLOAT_EXTREME, id="float32-extreme"),
    ]
    if _BF16:
        rows.append(pytest.param(torch.bfloat16, _FLOAT_BOUNDARY, id="bfloat16"))
    if _FP64:
        rows.append(pytest.param(torch.float64, _FLOAT_BOUNDARY, id="float64"))
    if _INT64:
        rows.append(pytest.param(torch.int64, _INT_WIDE_BOUNDARY, id="int64"))
    if _FP8:
        rows.append(
            pytest.param(torch.float8_e4m3fn, _FLOAT_BOUNDARY, id="float8-e4m3fn")
        )
        rows.append(pytest.param(torch.float8_e5m2, _FLOAT_BOUNDARY, id="float8-e5m2"))
    return rows


_BOUNDARY_WORKLOADS = tu.selected_cases(_boundary_rows(), quick=[])


@pytest.mark.underscore_cast_Char
@pytest.mark.parametrize("dtype,payload", _BOUNDARY_WORKLOADS)
def test__cast_Char_boundaries(dtype, payload):
    _skip_native_input(dtype, (len(payload),))
    if flag_gems.vendor_name == "ascend" and utils.TO_CPU and dtype == torch.int32:
        pytest.skip("Ascend wide int32-to-int8 narrowing does not match the CPU oracle")
    inp = torch.tensor(payload, dtype=dtype, device=flag_gems.device)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten._cast_Char(ref_inp)
    res_out = flag_gems._cast_Char(inp)

    _assert_cast_equal(res_out, ref_out, inp)


_PARAM_SHAPE = (1024, 1024)
_PARAM_DTYPES = tuple(
    dtype
    for dtype in (
        torch.uint8,
        torch.int8,
        torch.int32,
        torch.float16,
        torch.bfloat16,
        torch.int64,
        torch.float8_e4m3fn,
    )
    if _supported(dtype)
)
_NON_BLOCKING_FORMS = ("false", "true")
_NON_BLOCKING_WORKLOADS = tu.selected_cases(_NON_BLOCKING_FORMS, quick=[])


@pytest.mark.underscore_cast_Char
@pytest.mark.parametrize("form", _NON_BLOCKING_WORKLOADS)
@pytest.mark.parametrize("dtype", _PARAM_DTYPES)
def test__cast_Char_non_blocking(dtype, form):
    # The main grid already calls the omitted schema default; these rows pin the
    # explicit ``False``/``True`` forms on the non-zero 1024x1024 fixture.
    inp = _fixture(dtype, _PARAM_SHAPE)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten._cast_Char(ref_inp, non_blocking=form == "true")
    res_out = flag_gems._cast_Char(inp, non_blocking=form == "true")

    _assert_cast_equal(res_out, ref_out, inp)


_SPECIAL_DTYPES = tuple(dtype for dtype in _dtypes() if dtype.is_floating_point)
_SPECIAL_WORKLOADS = tu.selected_cases(
    [
        pytest.param(
            dtype, scenario, id="{}-{}".format(str(dtype).split(".")[-1], scenario)
        )
        for dtype, scenario in tu.special_value_cases(_SPECIAL_DTYPES)
    ],
    quick=[],
)


@pytest.mark.underscore_cast_Char
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_WORKLOADS)
def test__cast_Char_special_values(dtype, scenario):
    inp = _make_special_input(dtype, scenario)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten._cast_Char(ref_inp)
    res_out = flag_gems._cast_Char(inp)

    _assert_cast_equal(res_out, ref_out, inp)


# Narrowing to an integral dtype drops autograd history, so there is no backward
# family to cover; these rows still pin the forward result for an input that
# requires grad, against the native contract.
_GRAD_DTYPES = tuple(
    dtype
    for dtype in _dtypes(torch.float64)
    if dtype.is_floating_point and dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
)
_GRAD_WORKLOADS = tu.selected_cases(
    [pytest.param(dtype, id=str(dtype).split(".")[-1]) for dtype in _GRAD_DTYPES],
    quick=[],
)


@pytest.mark.underscore_cast_Char
@pytest.mark.parametrize("dtype", _GRAD_WORKLOADS)
def test__cast_Char_grad_enabled_input(dtype):
    inp = _fixture(dtype, (8, 8)).requires_grad_(True)
    ref_inp = tu.to_reference(inp.detach()).requires_grad_(True)

    ref_out = torch.ops.aten._cast_Char(ref_inp)
    res_out = flag_gems._cast_Char(inp)

    _assert_cast_equal(res_out, ref_out, inp)


@pytest.mark.underscore_cast_Char
def test__cast_Char_rejects_out_keyword():
    inp = _make_input(torch.float32, (4, 4), ["-1", "1"])
    out = torch.empty((4, 4), dtype=torch.int8, device=flag_gems.device)

    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Char(inp, out=out)


@pytest.mark.underscore_cast_Char
def test__cast_Char_rejects_non_bool_non_blocking():
    inp = _make_input(torch.float32, (4, 4), ["-1", "1"])

    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Char(inp, non_blocking="yes")


@pytest.mark.underscore_cast_Char
def test__cast_Char_requires_operand():
    # The native wrapper reports the missing operand as a RuntimeError, so both
    # error types are accepted here.
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Char()


def _require_master_dtype(dtype):
    if dtype is torch.float64 and not utils.fp64_is_supported:
        pytest.skip("FP64 is not supported on this backend")
    if dtype is torch.bfloat16 and not utils.bf16_is_supported:
        pytest.skip("BF16 is not supported on this backend")
    # MUSA runs complex casts even though its generic FP64 flag is off.
    if dtype.is_complex and flag_gems.vendor_name != "mthreads":
        if not utils.complex_is_supported or (
            dtype is torch.complex128 and not utils.fp64_is_supported
        ):
            pytest.skip(f"{dtype} is not supported on this backend")


def _master_cast_reference(inp, non_blocking=False):
    return torch.ops.aten._cast_Char(_reference(inp), non_blocking)


def _assert_master_cast_close(inp, actual, expected, *, rtol, atol):
    _check_cast_abi(actual, expected, inp)
    if actual.device != expected.device:
        actual = actual.cpu()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


_MASTER_DTYPES = [
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
]


def _make_master_input(shape, dtype):
    _require_master_dtype(dtype)
    torch.manual_seed(0)
    return torch.randint(-512, 512, shape, device=flag_gems.device).to(dtype)


@pytest.mark.underscore_cast_Char
@pytest.mark.parametrize(
    "shape", [(), (0,), (2, 0, 3), (1,), (1023,), (1025,), (129, 257)]
)
@pytest.mark.parametrize("dtype", _MASTER_DTYPES)
@pytest.mark.parametrize("non_blocking", [False, True])
def test__cast_Char_master(shape, dtype, non_blocking):
    x = _make_master_input(shape, dtype)
    expected = _master_cast_reference(x, non_blocking)
    actual = flag_gems._cast_Char(x, non_blocking)
    _assert_master_cast_close(x, actual, expected, rtol=0, atol=0)
    assert actual.stride() == expected.stride()
    assert not actual.requires_grad
    if dtype == torch.int8:
        assert actual is x
    elif x.numel():
        assert actual.data_ptr() != x.data_ptr()


@pytest.mark.underscore_cast_Char
@pytest.mark.parametrize("dtype", _MASTER_DTYPES)
@pytest.mark.parametrize(
    "layout", ["transpose", "slice", "expand", "channels_last", "permuted_slice"]
)
def test__cast_Char_master_strided(dtype, layout):
    x = _make_master_input((4, 5, 6, 7), dtype)
    if layout == "transpose":
        x = x.transpose(0, 3)
    elif layout == "slice":
        x = x[1:, :, ::2, 1::2]
    elif layout == "expand":
        x = x[:1].expand(9, -1, -1, -1)
    elif layout == "permuted_slice":
        x = x.transpose(0, 3)[::2, :, 1::2, :]
    else:
        x = x.contiguous(memory_format=torch.channels_last)
    expected = _master_cast_reference(x)
    actual = flag_gems._cast_Char(x)
    _assert_master_cast_close(x, actual, expected, rtol=0, atol=0)
    assert actual.stride() == expected.stride()


@pytest.mark.underscore_cast_Char
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test__cast_Char_master_fractional(dtype):
    _require_master_dtype(dtype)
    x = torch.tensor(
        [
            -257.75,
            -256.5,
            -129.5,
            -128.75,
            -1.9,
            -0.5,
            0,
            0.5,
            1.9,
            127.5,
            128.5,
            255.5,
            256.5,
            257.75,
            2**31,
            2**32 + 256,
            1e20,
            -1e20,
            float("inf"),
            -float("inf"),
            float("nan"),
        ],
        device=flag_gems.device,
        dtype=dtype,
    )
    expected = _master_cast_reference(x)
    _assert_master_cast_close(x, flag_gems._cast_Char(x), expected, rtol=0, atol=0)


@pytest.mark.underscore_cast_Char
@pytest.mark.parametrize("requires_grad", [False, True])
def test__cast_Char_master_requires_grad(requires_grad):
    torch.manual_seed(0)
    x = torch.randn(1025, device=flag_gems.device, requires_grad=requires_grad)
    expected = _master_cast_reference(x, True)
    actual = flag_gems._cast_Char(x, True)
    _assert_master_cast_close(x, actual, expected, rtol=0, atol=0)
    assert not actual.requires_grad


@pytest.mark.underscore_cast_Char
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test__cast_Char_master_complex(dtype):
    _require_master_dtype(dtype)
    torch.manual_seed(0)
    x = torch.randn(17, device=flag_gems.device, dtype=dtype).conj()
    expected = _master_cast_reference(x)
    actual = flag_gems._cast_Char(x)
    _assert_master_cast_close(x, actual, expected, rtol=0, atol=0)


@pytest.mark.underscore_cast_Char
def test__cast_Char_master_integer_limits():
    x = torch.tensor(
        [-(2**63), -(2**31) - 1, 2**31, 2**63 - 1], device=flag_gems.device
    )
    expected = _master_cast_reference(x)
    _assert_master_cast_close(x, flag_gems._cast_Char(x), expected, rtol=0, atol=0)


@pytest.mark.underscore_cast_Char
def test__cast_Char_master_negative_view():
    x = torch._neg_view(torch.arange(17, device=flag_gems.device, dtype=torch.float32))
    expected = _master_cast_reference(x)
    _assert_master_cast_close(x, flag_gems._cast_Char(x), expected, rtol=0, atol=0)
