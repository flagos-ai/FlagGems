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
    torch.complex64,
    torch.complex128,
]


TARGET = torch.int16


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_CAST_DTYPE = TARGET


def _skip_native_input(dtype, shape):
    nonempty = all((extent != 0 for extent in shape))
    if flag_gems.vendor_name == "ascend" and dtype in _FP8_DTYPES and nonempty:
        pytest.skip("Ascend _cast_Short has no native nonempty FP8 conversion")
    if (
        flag_gems.vendor_name == "kunlunxin"
        and dtype in (torch.bfloat16, torch.complex64)
        and nonempty
    ):
        pytest.skip(
            "Kunlunxin _cast_Short has no native nonempty bfloat16/complex64 conversion"
        )


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
    _check_cast_abi(result, reference, inp)
    tu.assert_result_equal(result, reference)


def _reference(inp):
    assert inp.device == _expected_device()
    _skip_native_input(inp.dtype, inp.shape)
    reference = tu.to_reference(inp)
    if (
        utils.TO_CPU
        and reference.numel()
        and (reference.dtype.is_floating_point or reference.is_complex())
    ):
        values = reference.real if reference.is_complex() else reference
        # Check actual floating inputs with Python integer bounds. In
        # particular, float(INT64_MAX) rounds up to the invalid value 2**63.
        # Integer inputs bypass this path and keep all their original bits.
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
    _require_dtype(dtype)
    if dtype.is_floating_point or dtype.is_complex:
        return torch.randn(shape, device=flag_gems.device, dtype=dtype)
    return torch.randint(-512, 512, shape, device=flag_gems.device).to(dtype)


def assert_cast(x, non_blocking=False):
    expected = torch.ops.aten._cast_Short.default(_reference(x), non_blocking)
    actual = flag_gems.cast_Short(x, non_blocking)
    _assert_cast_equal(actual, expected, x)
    assert actual.stride() == expected.stride()
    assert actual.requires_grad == expected.requires_grad
    if x.dtype == TARGET:
        assert actual is x
    elif x.numel():
        assert actual.data_ptr() != x.data_ptr()
    return actual, expected


@pytest.mark.cast_Short
@pytest.mark.parametrize(
    "shape", [(), (0,), (2, 0, 3), (1,), (1023,), (1025,), (129, 257)]
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("non_blocking", [False, True])
def test_cast_Short(shape, dtype, non_blocking):
    assert_cast(make_input(shape, dtype), non_blocking)


@pytest.mark.cast_Short
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "layout", ["transpose", "slice", "expand", "channels_last", "permuted_slice"]
)
def test_cast_Short_strided(dtype, layout):
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
    assert_cast(x)


@pytest.mark.cast_Short
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("conjugated", [False, True])
def test_cast_Short_complex_real_component(dtype, conjugated):
    # Nonzero imaginary parts must not contribute to the int16 result.
    real = torch.tensor([-32768.0, -2.9, 0.0, 3.9, 32767.0], device=flag_gems.device)
    imag = torch.tensor([7.0, -11.0, 5.0, -13.0, 17.0], device=flag_gems.device)
    x = torch.complex(real, imag).to(dtype)
    if conjugated:
        x = x.conj()
    actual, expected = assert_cast(x)
    torch.testing.assert_close(actual, real.to(torch.int16), rtol=0, atol=0)
    torch.testing.assert_close(expected, real.to(torch.int16), rtol=0, atol=0)
