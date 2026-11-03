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

import pytest
import torch

import flag_gems

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


def make_input(shape, dtype):
    return torch.randint(-512, 512, shape, device=flag_gems.device).to(dtype)


@pytest.mark.cast_Byte
@pytest.mark.parametrize(
    "shape", [(), (0,), (2, 0, 3), (1,), (1023,), (1025,), (129, 257)]
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("non_blocking", [False, True])
def test_cast_Byte(shape, dtype, non_blocking):
    x = make_input(shape, dtype)
    expected = torch.ops.aten._cast_Byte(x, non_blocking)
    actual = flag_gems.cast_Byte(x, non_blocking)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.stride() == expected.stride()
    assert not actual.requires_grad
    if dtype == torch.uint8:
        assert actual is x
    elif x.numel():
        assert actual.data_ptr() != x.data_ptr()


@pytest.mark.cast_Byte
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "layout", ["transpose", "slice", "expand", "channels_last", "permuted_slice"]
)
def test_cast_Byte_strided(dtype, layout):
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
    expected = torch.ops.aten._cast_Byte(x)
    actual = flag_gems.cast_Byte(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.stride() == expected.stride()


@pytest.mark.cast_Byte
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_cast_Byte_fractional(dtype):
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
    torch.testing.assert_close(
        flag_gems.cast_Byte(x), torch.ops.aten._cast_Byte(x), rtol=0, atol=0
    )


@pytest.mark.cast_Byte
@pytest.mark.parametrize("requires_grad", [False, True])
def test_cast_Byte_requires_grad(requires_grad):
    x = torch.randn(1025, device=flag_gems.device, requires_grad=requires_grad)
    expected = torch.ops.aten._cast_Byte(x, True)
    actual = flag_gems.cast_Byte(x, True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not actual.requires_grad


@pytest.mark.cast_Byte
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_cast_Byte_complex(dtype):
    x = torch.randn(17, device=flag_gems.device, dtype=dtype).conj()
    expected = torch.ops.aten._cast_Byte(x)
    actual = flag_gems.cast_Byte(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cast_Byte
def test_cast_Byte_integer_limits():
    x = torch.tensor(
        [-(2**63), -(2**31) - 1, 2**31, 2**63 - 1], device=flag_gems.device
    )
    torch.testing.assert_close(
        flag_gems.cast_Byte(x), torch.ops.aten._cast_Byte(x), rtol=0, atol=0
    )


@pytest.mark.cast_Byte
def test_cast_Byte_negative_view():
    x = torch._neg_view(torch.arange(17, device=flag_gems.device, dtype=torch.float32))
    torch.testing.assert_close(
        flag_gems.cast_Byte(x), torch.ops.aten._cast_Byte(x), rtol=0, atol=0
    )
