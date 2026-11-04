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

from . import accuracy_utils as utils


@pytest.mark.rsqrt
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rsqrt(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.rsqrt(ref_inp)
    res_out = flag_gems.rsqrt(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.rsqrt_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rsqrt_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.rsqrt_(ref_inp)
    res_out = flag_gems.rsqrt_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.rsqrt
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon launch path")
@pytest.mark.parametrize("n", [0, 1, 65535, 65536, 65537, 262144, 1048576])
@pytest.mark.parametrize("layout", ["contiguous", "offset", "strided", "transpose"])
def test_hygon_rsqrt_layout(n, layout):
    backing = torch.rand(2 * n + 17, device=flag_gems.device) + 0.125
    if layout == "offset":
        inp = backing[7 : 7 + n]
    elif layout == "strided":
        inp = backing[7 : 7 + 2 * n : 2]
    elif layout == "transpose":
        inp = backing[:n].reshape(1, n).t()
        if n % 2 == 0:
            inp = backing[:n].reshape(2, n // 2).t()
    else:
        inp = backing[:n]
    before = backing.cpu().clone()
    expected = torch.rsqrt(inp.cpu().double()).float()
    actual = flag_gems.rsqrt(inp)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-6, atol=0)
    assert actual.shape == inp.shape
    assert actual.dtype == inp.dtype
    assert actual.device == inp.device
    torch.testing.assert_close(backing.cpu(), before, rtol=0, atol=0)


@pytest.mark.rsqrt
@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon FP32 special values"
)
@pytest.mark.parametrize("offset", [0, 7])
def test_hygon_rsqrt_special_values(offset):
    # Build subnormal inputs by bits so input generation cannot flush them.
    generator = torch.Generator().manual_seed(1729)
    magnitude = torch.randint(1, 0x00800000, (32768,), generator=generator)
    edge = torch.tensor(
        [
            0,
            -2147483648,
            1,
            2,
            0x007FFFFF,
            0x00800000,
            0x7F7FFFFF,
            0x7F800000,
            -8388608,
            0x7FC00001,
            -4194303,
            0x3F800000,
            -1082130432,
        ],
        dtype=torch.int32,
    )
    bits = torch.cat((edge, magnitude.int(), (magnitude | 0x80000000).int()))
    values = bits.view(torch.float32)
    storage = torch.empty(values.numel() + offset, device=flag_gems.device)
    inp = storage[offset:]
    inp.copy_(values)
    before = inp.cpu().view(torch.int32).clone()
    expected = torch.rsqrt(values.double()).float()
    actual = flag_gems.rsqrt(inp).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=0, equal_nan=True)
    non_nan = ~torch.isnan(expected)
    assert torch.equal(torch.signbit(actual[non_nan]), torch.signbit(expected[non_nan]))
    assert torch.equal(inp.cpu().view(torch.int32), before)


@pytest.mark.rsqrt
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon dispatcher path")
def test_hygon_rsqrt_dispatch():
    inp = torch.rand(65537, device=flag_gems.device) + 0.125
    expected = torch.rsqrt(inp.cpu().double()).float()
    with flag_gems.use_gems(include=["rsqrt"]):
        actual = torch.rsqrt(inp)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-6, atol=0)


@pytest.mark.rsqrt
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon dtype fallback")
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float64, torch.int32]
)
def test_hygon_rsqrt_dtype_fallback(dtype):
    inp = torch.full((65537,), 4, dtype=dtype, device=flag_gems.device)
    actual = flag_gems.rsqrt(inp)
    expected = torch.rsqrt(inp.cpu())
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.rsqrt
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon view fallback")
def test_hygon_rsqrt_negative_view():
    inp = torch._neg_view(-torch.ones(65537, device=flag_gems.device))
    with flag_gems.use_gems(include=["rsqrt"]):
        actual = torch.rsqrt(inp)
    torch.testing.assert_close(actual.cpu(), torch.ones(65537), rtol=0, atol=0)


@pytest.mark.rsqrt
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon autograd fallback")
def test_hygon_rsqrt_autograd():
    inp = torch.full((65537,), 4.0, device=flag_gems.device, requires_grad=True)
    with flag_gems.use_gems(include=["rsqrt"]):
        actual = torch.rsqrt(inp)
        actual.sum().backward()
    torch.testing.assert_close(actual.cpu(), torch.full((65537,), 0.5), rtol=0, atol=0)
    torch.testing.assert_close(
        inp.grad.cpu(), torch.full((65537,), -0.0625), rtol=0, atol=0
    )
