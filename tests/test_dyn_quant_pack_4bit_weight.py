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


def _portable_pack_oracle(weights, scales, bias):
    parts = [weights.flatten().float(), scales.flatten().float()]
    if bias is not None:
        parts.append(bias.flatten().float())
    return torch.cat(parts)


def _make_inputs(
    block_size,
    in_features,
    out_features,
    with_bias,
    scale_dtype,
    scale_layout,
):
    weight_numel = out_features * (in_features // 2)
    weights = (
        torch.arange(weight_numel, dtype=torch.int32).remainder(256).to(torch.uint8)
    )
    weights = weights.reshape(out_features, in_features // 2)

    groups = in_features // block_size
    scales = torch.arange(out_features * groups, dtype=torch.float32).div(8).sub(1)
    if scale_layout == "matrix":
        scales = scales.reshape(out_features, groups)
    scales = scales.to(scale_dtype)

    bias = None
    if with_bias:
        bias = torch.arange(out_features, dtype=torch.float32).div(4).sub(0.5)
        bias = bias.to(scale_dtype)
    return weights, scales, bias


@pytest.mark.dyn_quant_pack_4bit_weight
@pytest.mark.parametrize(
    "block_size,in_features,out_features,with_bias,scale_dtype,scale_layout",
    [
        pytest.param(34, 34, 3, False, torch.float32, "matrix", id="channel-tail"),
        pytest.param(32, 96, 5, True, torch.bfloat16, "matrix", id="grouped-bf16"),
        pytest.param(32, 128, 7, True, torch.float16, "flat", id="grouped-flat"),
    ],
)
def test_dyn_quant_pack_4bit_weight_portable_abi(
    block_size,
    in_features,
    out_features,
    with_bias,
    scale_dtype,
    scale_layout,
):
    weights_cpu, scales_cpu, bias_cpu = _make_inputs(
        block_size,
        in_features,
        out_features,
        with_bias,
        scale_dtype,
        scale_layout,
    )
    expected = _portable_pack_oracle(weights_cpu, scales_cpu, bias_cpu)
    weights = weights_cpu.to(flag_gems.device)

    actual = flag_gems._dyn_quant_pack_4bit_weight(
        weights,
        scales_cpu.to(flag_gems.device),
        None if bias_cpu is None else bias_cpu.to(flag_gems.device),
        block_size,
        in_features,
        out_features,
    )

    assert actual.dtype == torch.float32
    assert actual.ndim == 1
    assert actual.numel() == expected.numel()
    assert actual.is_contiguous()
    assert actual.device == weights.device
    if not utils.TO_CPU:
        expected = expected.to(flag_gems.device)
    utils.gems_assert_equal(actual, expected)


@pytest.mark.dyn_quant_pack_4bit_weight
def test_dyn_quant_pack_4bit_weight_preserves_nibble_bytes_and_segment_order():
    weights_cpu = torch.tensor(
        [[0x80, 0x7F, 0xF0, 0x0F, 0x00, 0xFF, 0x17, 0x71]], dtype=torch.uint8
    )
    scales_cpu = torch.tensor([[0.5]], dtype=torch.float32)
    bias_cpu = torch.tensor([-1.25], dtype=torch.float32)
    expected = torch.tensor(
        [128, 127, 240, 15, 0, 255, 23, 113, 0.5, -1.25],
        dtype=torch.float32,
    )

    actual = flag_gems._dyn_quant_pack_4bit_weight(
        weights_cpu.to(flag_gems.device),
        scales_cpu.to(flag_gems.device),
        bias_cpu.to(flag_gems.device),
        16,
        16,
        1,
    )

    if not utils.TO_CPU:
        expected = expected.to(flag_gems.device)
    utils.gems_assert_equal(actual, expected)


@pytest.mark.dyn_quant_pack_4bit_weight
def test_dyn_quant_pack_4bit_weight_noncontiguous_inputs():
    block_size, in_features, out_features = 32, 96, 5
    groups = in_features // block_size
    weights_base_cpu = (
        torch.arange(out_features * (in_features // 2), dtype=torch.int32)
        .remainder(256)
        .to(torch.uint8)
        .reshape(in_features // 2, out_features)
    )
    scales_base_cpu = torch.arange(out_features * groups, dtype=torch.float32).reshape(
        groups, out_features
    )
    bias_base_cpu = torch.arange(out_features * 2, dtype=torch.float32)
    weights_cpu = weights_base_cpu.T
    scales_cpu = scales_base_cpu.T
    bias_cpu = bias_base_cpu[::2]
    weights = weights_base_cpu.to(flag_gems.device).T
    scales = scales_base_cpu.to(flag_gems.device).T
    bias = bias_base_cpu.to(flag_gems.device)[::2]
    for tensor in (weights_cpu, scales_cpu, bias_cpu, weights, scales, bias):
        assert not tensor.is_contiguous()
    expected = _portable_pack_oracle(weights_cpu, scales_cpu, bias_cpu)

    actual = flag_gems._dyn_quant_pack_4bit_weight(
        weights,
        scales,
        bias,
        block_size,
        in_features,
        out_features,
    )

    if not utils.TO_CPU:
        expected = expected.to(flag_gems.device)
    utils.gems_assert_equal(actual, expected)


@pytest.mark.dyn_quant_pack_4bit_weight
@pytest.mark.parametrize("with_bias", [False, True])
def test_dyn_quant_pack_4bit_weight_zero_out_features(with_bias):
    weights = torch.empty((0, 32), dtype=torch.uint8, device=flag_gems.device)
    scales = torch.empty((0, 1), dtype=torch.float32, device=flag_gems.device)
    bias = torch.empty((0,), dtype=torch.float32, device=flag_gems.device)

    actual = flag_gems._dyn_quant_pack_4bit_weight(
        weights, scales, bias if with_bias else None, 64, 64, 0
    )

    assert actual.dtype == torch.float32
    assert actual.shape == (0,)
    assert actual.is_contiguous()


def _invalid_inputs(case):
    block_size, in_features, out_features = 64, 64, 4
    weights = torch.zeros((out_features, in_features // 2), dtype=torch.uint8)
    scales = torch.ones((out_features, 1), dtype=torch.float32)
    bias = torch.zeros(out_features, dtype=torch.float32)

    if case == "zero-block":
        block_size = 0
    elif case == "negative-block":
        block_size = -32
    elif case == "non-group-block":
        block_size = 16
    elif case == "non-dividing-block":
        block_size, in_features = 64, 96
        weights = torch.zeros((out_features, in_features // 2), dtype=torch.uint8)
    elif case == "zero-in-features":
        block_size, in_features = 0, 0
        weights = torch.empty((out_features, 0), dtype=torch.uint8)
        scales = torch.empty((out_features, 0), dtype=torch.float32)
    elif case == "negative-in-features":
        block_size, in_features = -64, -64
    elif case == "odd-in-features":
        block_size, in_features = 63, 63
        weights = torch.zeros((out_features, in_features // 2), dtype=torch.uint8)
    elif case == "negative-out-features":
        out_features = -1
    elif case == "weight-dtype":
        weights = weights.float()
    elif case == "weight-rank":
        weights = weights.flatten()
    elif case == "weight-rows":
        weights = weights[:-1]
    elif case == "weight-columns":
        weights = weights[:, :-1]
    elif case == "scale-dtype":
        scales = scales.to(torch.int32)
    elif case == "scale-rank":
        scales = scales.unsqueeze(0)
    elif case == "scale-shape":
        scales = scales.reshape(1, out_features)
    elif case == "scale-length":
        scales = scales.flatten()[:-1]
    elif case == "bias-dtype":
        bias = bias.to(torch.int32)
    elif case == "bias-rank":
        bias = bias.unsqueeze(0)
    elif case == "bias-length":
        bias = bias[:-1]
    else:
        raise AssertionError(f"unhandled invalid test case: {case}")

    return weights, scales, bias, block_size, in_features, out_features


@pytest.mark.dyn_quant_pack_4bit_weight
@pytest.mark.parametrize(
    "case",
    [
        "zero-block",
        "negative-block",
        "non-group-block",
        "non-dividing-block",
        "zero-in-features",
        "negative-in-features",
        "odd-in-features",
        "negative-out-features",
        "weight-dtype",
        "weight-rank",
        "weight-rows",
        "weight-columns",
        "scale-dtype",
        "scale-rank",
        "scale-shape",
        "scale-length",
        "bias-dtype",
        "bias-rank",
        "bias-length",
    ],
)
def test_dyn_quant_pack_4bit_weight_rejects_invalid_contract(case):
    weights, scales, bias, block_size, in_features, out_features = _invalid_inputs(case)
    weights = weights.to(flag_gems.device)
    scales = scales.to(flag_gems.device)
    bias = bias.to(flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems._dyn_quant_pack_4bit_weight(
            weights,
            scales,
            bias,
            block_size,
            in_features,
            out_features,
        )


@pytest.mark.dyn_quant_pack_4bit_weight
@pytest.mark.parametrize("mismatched", ["scales", "bias"])
def test_dyn_quant_pack_4bit_weight_rejects_device_mismatch(mismatched):
    weights = torch.zeros((4, 32), dtype=torch.uint8, device=flag_gems.device)
    if weights.device.type == "cpu":
        pytest.skip("requires distinct accelerator and CPU devices")
    parameter_device = weights.device
    scales = torch.ones((4, 1), dtype=torch.float32, device=parameter_device)
    bias = torch.zeros(4, dtype=torch.float32, device=parameter_device)
    if mismatched == "scales":
        scales = scales.cpu()
    else:
        bias = bias.cpu()

    with pytest.raises(RuntimeError, match="same device"):
        flag_gems._dyn_quant_pack_4bit_weight(weights, scales, bias, 64, 64, 4)
