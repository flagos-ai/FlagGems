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
from . import kernelgen_test_utils as tu
from .accuracy_utils import _SKIP_FP64

DTYPES = [
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    pytest.param(torch.float64, marks=_SKIP_FP64),
]


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


def _require_dtype(dtype):
    supported = {
        torch.bfloat16: utils.bf16_is_supported,
        torch.float64: utils.fp64_is_supported,
        torch.int64: utils.int64_is_supported,
        torch.complex64: utils.complex_is_supported,
        torch.complex128: utils.complex_is_supported and utils.fp64_is_supported,
    }
    if not supported.get(dtype, True):
        pytest.skip(f"{dtype} is not supported on this backend")


def make_input(shape, dtype):
    if dtype.is_floating_point:
        return torch.randint(
            -128, 128, shape, dtype=torch.int32, device=flag_gems.device
        ).to(dtype)
    return torch.randint(-512, 512, shape, device=flag_gems.device).to(dtype)


@pytest.mark.cast_Char
@pytest.mark.parametrize(
    "shape", [(), (0,), (2, 0, 3), (1,), (1023,), (1025,), (129, 257)]
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("non_blocking", [False, True])
def test_cast_Char(shape, dtype, non_blocking):
    _require_dtype(dtype)
    x = make_input(shape, dtype)
    expected = torch.ops.aten._cast_Char.default(_reference(x), non_blocking)
    actual = flag_gems.cast_Char(x, non_blocking)
    _assert_cast_equal(actual, expected, x)
    assert actual.stride() == expected.stride()
    assert not actual.requires_grad
    if dtype == torch.int8:
        assert actual is x
    elif x.numel():
        assert actual.data_ptr() != x.data_ptr()


@pytest.mark.cast_Char
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "layout", ["transpose", "slice", "expand", "channels_last", "permuted_slice"]
)
def test_cast_Char_strided(dtype, layout):
    _require_dtype(dtype)
    x = make_input((4, 5, 6, 7), dtype)
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
    expected = torch.ops.aten._cast_Char.default(_reference(x))
    actual = flag_gems.cast_Char(x)
    _assert_cast_equal(actual, expected, x)
    assert actual.stride() == expected.stride()


@pytest.mark.cast_Char
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        pytest.param(torch.float64, marks=_SKIP_FP64),
    ],
)
def test_cast_Char_fractional(dtype):
    _require_dtype(dtype)
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
    expected = torch.ops.aten._cast_Char.default(_reference(x))
    actual = flag_gems.cast_Char(x)
    _assert_cast_equal(actual, expected, x)


@pytest.mark.cast_Char
@pytest.mark.parametrize("requires_grad", [False, True])
def test_cast_Char_requires_grad(requires_grad):
    x = torch.randn(1025, device=flag_gems.device, requires_grad=requires_grad)
    expected = torch.ops.aten._cast_Char.default(_reference(x), True)
    actual = flag_gems.cast_Char(x, True)
    _assert_cast_equal(actual, expected, x)
    assert not actual.requires_grad


@pytest.mark.cast_Char
@pytest.mark.parametrize(
    "dtype", [torch.complex64, pytest.param(torch.complex128, marks=_SKIP_FP64)]
)
def test_cast_Char_complex(dtype):
    _require_dtype(dtype)
    x = torch.randn(17, device=flag_gems.device, dtype=dtype).conj()
    expected = torch.ops.aten._cast_Char.default(_reference(x))
    actual = flag_gems.cast_Char(x)
    _assert_cast_equal(actual, expected, x)


@pytest.mark.cast_Char
def test_cast_Char_integer_limits():
    x = torch.tensor(
        [-(2**63), -(2**31) - 1, 2**31, 2**63 - 1], device=flag_gems.device
    )
    expected = torch.ops.aten._cast_Char.default(_reference(x))
    actual = flag_gems.cast_Char(x)
    _assert_cast_equal(actual, expected, x)


@pytest.mark.cast_Char
def test_cast_Char_negative_view():
    x = torch._neg_view(torch.arange(17, device=flag_gems.device, dtype=torch.float32))
    expected = torch.ops.aten._cast_Char.default(_reference(x))
    actual = flag_gems.cast_Char(x)
    _assert_cast_equal(actual, expected, x)
