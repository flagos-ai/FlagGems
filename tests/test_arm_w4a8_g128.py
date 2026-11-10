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
    [
        (1, 64, 128),
        (5, 64, 128),
        (8, 64, 128),
        (1, 96, 5120),
        (8, 64, 5120),
        (4, 64, 512),  # prefill boundary: exactly one 4-row panel
        (2, 96, 5120),  # decode path with the wider N/K contract
        (7, 64, 512),  # odd prefill rows: exercises the padded_rows[:rows] slice
        (3, 96, 256),  # multi-row decode partitioning (rows < 4)
    ],
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


@pytest.mark.skipif(
    not _ARM_CPU_BACKEND_ACTIVE,
    reason="requires an AArch64 Triton CPU backend",
)
def test_w4a8_g128_decode_path_matches_prefill_path():
    """Both kernel paths must produce the same result for the same row:
    decode runs the fused SDOT GEMV, prefill runs panel4 pack + I8MM."""
    torch.manual_seed(46)
    n, k = 64, 256
    m = 8
    input = (0.4 * torch.randn((m, k), dtype=torch.bfloat16)).contiguous()
    weight = torch.randint(-8, 8, (n, k), dtype=torch.int8)
    scales = 0.001 + 0.02 * torch.rand((n, k // 128))
    packed = pack_rhs_qsi4c128p(weight, scales)

    prefill = w4a8_g128_linear(input, packed, n, k)
    decode = torch.cat(
        [
            w4a8_g128_linear(input[i : i + 1].contiguous(), packed, n, k)
            for i in range(m)
        ]
    )

    torch.testing.assert_close(decode, prefill, rtol=5e-2, atol=5e-2)


def test_w4a8_g128_pack_layout_round_trips():
    """Byte-level verification of the packed N4/K128 ABI (pure PyTorch).

    The packed buffer per [N//4, groups] tile is:
      bytes [0:8]     four BF16 group scales, one per output row of the tile
      bytes [8:264]  INT4 nibbles: byte ((s*2+h)*4+j)*8+w packs
                     low  = weight[t*4+j, g*128 + s*32 + h*8 + w]      & 15
                     high = weight[t*4+j, g*128 + s*32 + 16 + h*8 + w] & 15
      bytes [264:...] per-row zero-point correction sums (FP32)
    """
    torch.manual_seed(44)
    n, k = 8, 256
    groups = k // 128
    weight = torch.randint(-8, 8, (n, k), dtype=torch.int8)
    scales = 0.001 + 0.02 * torch.rand((n, groups))

    packed = pack_rhs_qsi4c128p(weight, scales)
    assert packed.dtype == torch.uint8
    assert packed.is_contiguous()

    tile_stride = groups * 264 + 16
    assert packed.numel() == (n // 4) * tile_stride
    tiles = packed.reshape(n // 4, tile_stride)
    group_bytes = tiles[:, : groups * 264].reshape(n // 4, groups, 264)

    # Scale block: stored[t, g, j] = bf16(scales[t*4+j, g]).
    stored_scales = group_bytes[:, :, :8].view(torch.bfloat16)  # [t, g, 4]
    recovered_scales = stored_scales.permute(0, 2, 1).reshape(n, groups)
    assert torch.equal(
        recovered_scales.float(), scales.to(torch.bfloat16).float()
    )

    # Nibble block: decode both nibbles back to signed INT4 weights.
    region = group_bytes[:, :, 8:264].reshape(n // 4, groups, 4, 2, 4, 8)
    low = region.to(torch.int16) & 15
    high = (region.to(torch.int16) >> 4) & 15
    sign_fix = lambda nib: torch.where(nib < 8, nib, nib - 16).to(torch.int8)
    low_signed = sign_fix(low)
    high_signed = sign_fix(high)

    t = torch.arange(n // 4).view(-1, 1, 1, 1, 1, 1)
    g = torch.arange(groups).view(1, -1, 1, 1, 1, 1)
    s = torch.arange(4).view(1, 1, -1, 1, 1, 1)
    h = torch.arange(2).view(1, 1, 1, -1, 1, 1)
    j = torch.arange(4).view(1, 1, 1, 1, -1, 1)
    w = torch.arange(8).view(1, 1, 1, 1, 1, -1)
    shape = (n // 4, groups, 4, 2, 4, 8)
    rows = (t * 4 + j).expand(shape)
    cols = (g * 128 + s * 32 + h * 8 + w).expand(shape)

    recovered = torch.zeros((n, k), dtype=torch.int8)
    recovered[rows, cols] = low_signed
    recovered[rows, cols + 16] = high_signed
    assert torch.equal(recovered, weight)

    # Tail block: per-row sum over groups of (group weight sum * 16) * scale.
    tail = tiles[:, groups * 264 :].view(torch.float32)  # [t, 4]
    scaled_sums = (
        weight.reshape(n, groups, 128).to(torch.int16).sum(dim=-1) * 16
    ).to(torch.float32)
    expected_tail = (
        scaled_sums * scales.to(torch.bfloat16).to(torch.float32).reshape(n, groups)
    ).sum(dim=1).reshape(n // 4, 4)
    torch.testing.assert_close(tail, expected_tail, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    platform.machine().lower() not in {"arm64", "aarch64"},
    reason="input validation is only reachable on AArch64 hosts",
)
def test_w4a8_g128_linear_rejects_invalid_contracts():
    n, k = 64, 128
    weight = torch.randint(-8, 8, (n, k), dtype=torch.int8)
    scales = torch.rand((n, 1), dtype=torch.float32)
    packed = pack_rhs_qsi4c128p(weight, scales)
    x = torch.randn((2, k), dtype=torch.bfloat16).contiguous()

    with pytest.raises(ValueError, match="BF16"):
        w4a8_g128_linear(x.float(), packed, n, k)
    non_contiguous = torch.randn((k, 2), dtype=torch.bfloat16).t()
    with pytest.raises(ValueError, match="BF16"):
        w4a8_g128_linear(non_contiguous, packed, n, k)
    with pytest.raises(ValueError, match="UINT8"):
        w4a8_g128_linear(x, packed.to(torch.int8), n, k)
    with pytest.raises(ValueError, match="byte count"):
        w4a8_g128_linear(x, packed[:-1].contiguous(), n, k)
    with pytest.raises(ValueError, match="input shape"):
        w4a8_g128_linear(x, packed, n, k=64)
    with pytest.raises(ValueError, match="N%4=0"):
        w4a8_g128_linear(x, packed, n=65, k=k)
    with pytest.raises(ValueError, match="CPU tensors only"):
        meta_x = torch.empty((2, k), dtype=torch.bfloat16, device="meta")
        w4a8_g128_linear(meta_x, packed, n, k)


@pytest.mark.skipif(
    not _ARM_CPU_BACKEND_ACTIVE,
    reason="requires an AArch64 Triton CPU backend",
)
@pytest.mark.parametrize("prefix", [(2, 3), (3, 1, 1), (5,)])
@pytest.mark.parametrize("k", [128, 512])
def test_w4a8_g128_linear_high_dim_input(prefix, k):
    n = 64
    torch.manual_seed(47)
    input = (0.4 * torch.randn((*prefix, k), dtype=torch.bfloat16)).contiguous()
    weight = torch.randint(-8, 8, (n, k), dtype=torch.int8)
    scales = 0.001 + 0.02 * torch.rand((n, k // 128))
    packed = pack_rhs_qsi4c128p(weight, scales)

    output = w4a8_g128_linear(input, packed, n, k)

    assert output.shape == (*prefix, n)
    assert output.dtype == torch.bfloat16
    rows = input.numel() // k
    ref = _reference(input.reshape(rows, k), weight, scales).reshape(*prefix, n)
    torch.testing.assert_close(output, ref, rtol=0.02, atol=0.125)
