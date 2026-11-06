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

import importlib
import struct

import pytest
import torch

import flag_gems

pytestmark = [
    pytest.mark.neg,
    pytest.mark.skipif(
        flag_gems.vendor_name != "hygon", reason="Hygon neg layout repair"
    ),
]


def _recipe(kind):
    z, nz, one, minus, hole = 0, 0x80000000, 0x3F800000, 0xBF800000, 0x42C60000
    original = [
        struct.unpack("<I", struct.pack("<f", ((i % 65) - 32) / 8))[0]
        for i in range(128)
    ]
    return {
        "transpose_minimal": ([z, nz, one, minus, z, nz], (3, 2), (1, 3), 0),
        "stride2_offset": (
            [hole, z, hole, nz, hole, one, hole, minus, hole],
            (4,),
            (2,),
            1,
        ),
        "slice2d_offset": (
            [hole, z, hole, nz, hole, one, hole, minus, hole, z, hole, nz],
            (2, 2),
            (6, 2),
            1,
        ),
        "expand_both_zeros": ([z, nz, one, minus], (3, 4), (0, 1), 0),
        "worker_transpose": (original, (16, 8), (1, 16), 0),
        "worker_slice": (original, (8, 8), (16, 2), 1),
    }[kind]


def _check_layout(words, shape, stride, offset, monkeypatch):
    signed = [word if word < 2**31 else word - 2**32 for word in words]
    cpu_bits = torch.tensor(signed, dtype=torch.int32, device="cpu")
    cpu_input = cpu_bits.view(torch.float32).as_strided(shape, stride, offset)
    device_bits = cpu_bits.to(flag_gems.device)
    inp = device_bits.view(torch.float32).as_strided(shape, stride, offset)
    provider = importlib.import_module(flag_gems.neg.__module__)
    assert provider.__name__.endswith("_hygon.ops.neg")
    assert not provider._can_use_fast_path(inp)
    assert provider._can_use_strided_fp32_path(inp)

    def unexpected_generic(A):
        pytest.fail("ordinary strided FP32 neg must use the sign-bit expression")

    monkeypatch.setattr(provider, "_generic_neg", unexpected_generic)
    result = flag_gems.neg(inp)
    expected = torch.neg(cpu_input)
    expected_bits = cpu_input.view(torch.int32) ^ -2147483648
    torch.testing.assert_close(
        result.view(torch.int32).cpu(), expected_bits, rtol=0, atol=0
    )
    torch.testing.assert_close(
        result.view(torch.int32).cpu(), expected.view(torch.int32), rtol=0, atol=0
    )
    assert result.shape == inp.shape
    assert result.dtype == inp.dtype
    assert result.device == inp.device
    assert result.stride() == expected.stride()
    assert result.untyped_storage()._cdata != inp.untyped_storage()._cdata
    torch.testing.assert_close(device_bits.cpu(), cpu_bits, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kind",
    [
        "transpose_minimal",
        "stride2_offset",
        "slice2d_offset",
        "expand_both_zeros",
        "worker_transpose",
        "worker_slice",
    ],
)
def test_neg_hygon_strided_signed_zero(kind, monkeypatch):
    _check_layout(*_recipe(kind), monkeypatch)


def test_neg_hygon_strided_special_bits(monkeypatch):
    special = [
        0,
        0x80000000,
        1,
        0x80000001,
        0x007FFFFF,
        0x807FFFFF,
        0x7F800000,
        0xFF800000,
        0x7FC00001,
        0xFFC12345,
        0x7F800001,
        0xFF812345,
    ]
    # Every unused storage word is initialized, including both ends and gaps.
    words = [0x42C60000] * 26
    for index, word in enumerate(special):
        words[1 + 2 * index] = word
    _check_layout(words, (3, 4), (8, 2), 1, monkeypatch)
