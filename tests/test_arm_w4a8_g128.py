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

import platform

import pytest
import torch

import flag_gems
from flag_gems.quantized_linear import pack_rhs_qsi4c128p, w4a8_g128_linear

_ARM_CPU_BACKEND_ACTIVE = (
    platform.machine().lower() in {"arm64", "aarch64"}
    and flag_gems.vendor_name == "arm"
    and flag_gems.device == "cpu"
)


def _asymmetric_a8_reference(input: torch.Tensor):
    values = input.to(torch.float32)
    zeros = torch.zeros(values.shape[0], dtype=torch.float32)
    row_min = torch.minimum(values.amin(dim=1), zeros)
    row_max = torch.maximum(values.amax(dim=1), zeros)
    value_range = row_max - row_min
    multiplier = torch.where(value_range == 0, 1.0, 255.0 / value_range)
    scale = 1.0 / multiplier
    descaled_min = row_min * multiplier
    descaled_max = row_max * multiplier
    choose_min = -128.0 + descaled_min + 127.0 + descaled_max > 0.0
    zero_point = torch.where(choose_min, -128.0 - descaled_min, 127.0 - descaled_max)
    zero_point = torch.round(zero_point.clamp(-128.0, 127.0)).to(torch.int32)
    scaled = values * multiplier[:, None]
    rounded = torch.sign(scaled) * torch.floor(torch.abs(scaled) + 0.5)
    quantized = (rounded.to(torch.int32) + zero_point[:, None]).clamp(-128, 127)
    return quantized, scale, zero_point


def _reference(input, weight, weight_scale):
    quantized, input_scale, zero_point = _asymmetric_a8_reference(input)
    m, k = quantized.shape
    n = weight.shape[0]
    groups = k // 128
    centered = quantized - zero_point[:, None]
    accumulator = torch.einsum(
        "mgk,ngk->mng",
        centered.reshape(m, groups, 128).to(torch.float32),
        weight.reshape(n, groups, 128).to(torch.float32),
    )
    scales = weight_scale.to(torch.bfloat16).to(torch.float32)
    return ((accumulator * scales[None, :, :]).sum(dim=-1) * input_scale[:, None]).to(
        torch.bfloat16
    )


@pytest.mark.skipif(
    not _ARM_CPU_BACKEND_ACTIVE,
    reason="requires an AArch64 Triton CPU backend",
)
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [(1, 64, 128), (5, 64, 128), (8, 64, 128), (1, 96, 5120), (8, 64, 5120)],
)
def test_w4a8_g128_matches_reference(m, n, k):
    torch.manual_seed(1000 + m + n + k)
    input = (0.4 * torch.randn((m, k), dtype=torch.bfloat16)).contiguous()
    weight = torch.randint(-8, 8, (n, k), dtype=torch.int8)
    scales = 0.001 + 0.02 * torch.rand((n, k // 128))
    packed = pack_rhs_qsi4c128p(weight, scales)

    output = w4a8_g128_linear(input, packed, n, k)

    assert output.shape == (m, n)
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(
        output, _reference(input, weight, scales), rtol=0.02, atol=0.125
    )


def test_w4a8_g128_packer_rejects_invalid_contracts():
    weight = torch.zeros((4, 128), dtype=torch.int8)
    scales = torch.ones((4, 1), dtype=torch.float32)

    with pytest.raises(ValueError, match="INT8"):
        pack_rhs_qsi4c128p(weight.to(torch.int16), scales)
    with pytest.raises(ValueError, match="scale shape"):
        pack_rhs_qsi4c128p(weight, torch.ones((4, 2)))
    invalid_weight = weight.clone()
    invalid_weight[0, 0] = -9
    with pytest.raises(ValueError, match=r"\[-8,7\]"):
        pack_rhs_qsi4c128p(invalid_weight, scales)
    invalid_scale = scales.clone()
    invalid_scale[0, 0] = torch.inf
    with pytest.raises(ValueError, match="finite"):
        pack_rhs_qsi4c128p(weight, invalid_scale)


@pytest.mark.skipif(
    not _ARM_CPU_BACKEND_ACTIVE,
    reason="requires an AArch64 Triton CPU backend",
)
def test_w4a8_g128_decode_is_repeatable():
    torch.manual_seed(43)
    n, k = 64, 128
    input = torch.randn((1, k), dtype=torch.bfloat16)
    weight = torch.randint(-8, 8, (n, k), dtype=torch.int8)
    scales = torch.rand((n, 1), dtype=torch.float32)
    packed = pack_rhs_qsi4c128p(weight, scales)

    first = w4a8_g128_linear(input, packed, n, k)
    second = w4a8_g128_linear(input, packed, n, k)

    torch.testing.assert_close(second, first, rtol=0, atol=0)
