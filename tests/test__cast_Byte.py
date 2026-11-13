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
    torch.float64,
]


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


def _native_cast(inp, *args):
    if utils.TO_CPU:
        inp = _copy_dense_to(inp, "cpu")
        values = inp.real if inp.is_complex() else inp
        if values.is_floating_point() and values.numel():
            values = values.double().trunc()
            low, high = float(values.amin()), float(values.amax())
            if not (
                math.isfinite(low) and math.isfinite(high) and 0 <= low <= high <= 255
            ):
                pytest.skip(
                    "CPU/device cast has no portable result for nonfinite or out-of-range inputs"
                )
    return torch.ops.aten._cast_Byte.default(inp, *args)


def _assert_cast(actual, expected, inp):
    assert actual.device == inp.device
    assert actual.dtype == expected.dtype == torch.uint8
    assert actual.shape == expected.shape == inp.shape
    assert actual.stride() == expected.stride()
    torch.testing.assert_close(utils.to_cpu(actual, expected), expected, rtol=0, atol=0)


@pytest.fixture(autouse=True)
def _seed_input():
    torch.manual_seed(0)


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
    # Keep ordinary float shape/layout cases in the portable cast domain.
    # The dedicated boundary cases exercise overflow and nonfinite inputs.
    low, high = (0, 256) if dtype.is_floating_point else (-512, 512)
    return torch.randint(
        low, high, shape, dtype=torch.int32, device=flag_gems.device
    ).to(dtype)


@pytest.mark.underscore_cast_Byte
@pytest.mark.parametrize(
    "shape", [(), (0,), (2, 0, 3), (1,), (1023,), (1025,), (129, 257)]
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("non_blocking", [False, True])
def test_cast_Byte(shape, dtype, non_blocking):
    _require_dtype(dtype)
    x = make_input(shape, dtype)
    expected = _native_cast(x, non_blocking)
    actual = flag_gems._cast_Byte(x, non_blocking)
    _assert_cast(actual, expected, x)
    assert not actual.requires_grad
    if dtype == torch.uint8:
        assert actual is x
    elif x.numel():
        assert actual.data_ptr() != x.data_ptr()


@pytest.mark.underscore_cast_Byte
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "layout", ["transpose", "slice", "expand", "channels_last", "permuted_slice"]
)
def test_cast_Byte_strided(dtype, layout):
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
    expected = _native_cast(x)
    actual = flag_gems._cast_Byte(x)
    _assert_cast(actual, expected, x)


@pytest.mark.underscore_cast_Byte
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_cast_Byte_fractional(dtype):
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
    expected = _native_cast(x)
    _assert_cast(flag_gems._cast_Byte(x), expected, x)


@pytest.mark.underscore_cast_Byte
@pytest.mark.parametrize("requires_grad", [False, True])
def test_cast_Byte_requires_grad(requires_grad):
    x = (torch.rand(1025, device=flag_gems.device) * 255).requires_grad_(requires_grad)
    expected = _native_cast(x, True)
    actual = flag_gems._cast_Byte(x, True)
    _assert_cast(actual, expected, x)
    assert not actual.requires_grad


@pytest.mark.underscore_cast_Byte
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="XPyTorch _cast_Byte does not support complex->uint8 (native and "
    "_to_copy both raise NOT IMPLEMENTED); no device reference exists",
)
def test_cast_Byte_complex(dtype):
    _require_dtype(dtype)
    x = torch.randn(17, device=flag_gems.device, dtype=dtype).conj()
    expected = _native_cast(x)
    actual = flag_gems._cast_Byte(x)
    _assert_cast(actual, expected, x)


@pytest.mark.underscore_cast_Byte
def test_cast_Byte_integer_limits():
    _require_dtype(torch.int64)
    x = torch.tensor(
        [-(2**63), -(2**31) - 1, 2**31, 2**63 - 1], device=flag_gems.device
    )
    expected = _native_cast(x)
    _assert_cast(flag_gems._cast_Byte(x), expected, x)


@pytest.mark.underscore_cast_Byte
def test_cast_Byte_negative_view():
    x = torch._neg_view(torch.arange(17, device=flag_gems.device, dtype=torch.float32))
    expected = _native_cast(x)
    _assert_cast(flag_gems._cast_Byte(x), expected, x)
