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

from . import conftest as cfg

SHAPES = (
    [((2,), 4, 8)]
    if cfg.QUICK_MODE
    else [
        ((2,), 4, 8),
        ((3,), 33, 65),
        ((2, 5), 128, 256),
        ((64,), 512, 1024),
    ]
)


def _make_qparams(device):
    return (
        torch.tensor(0.05, device=device),
        torch.tensor(127, device=device),
        torch.tensor(0.03125, device=device),
        torch.tensor(-3, device=device),
        torch.tensor(0.08, device=device),
        torch.tensor(121, device=device),
    )


@pytest.mark.wrapped_linear_prepack
@pytest.mark.wrapped_quantized_linear_prepacked
@pytest.mark.parametrize("leading_shape,N,K", SHAPES)
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_wrapped_quantized_linear_prepacked(leading_shape, N, K, noncontiguous):
    if noncontiguous:
        input = torch.randn((*leading_shape, K * 2), device=flag_gems.device)[..., ::2]
        weight = torch.randn((N, K * 2), device=flag_gems.device)[..., ::2]
        bias = torch.randn((N * 2,), device=flag_gems.device)[::2]
    else:
        input = torch.randn((*leading_shape, K), device=flag_gems.device)
        weight = torch.randn((N, K), device=flag_gems.device)
        bias = torch.randn((N,), device=flag_gems.device)

    input_scale, input_zp, weight_scale, weight_zp, output_scale, output_zp = (
        _make_qparams(flag_gems.device)
    )
    # The native packed object is an opaque CPU pointer and its consumer has no
    # usable CUDA path, so this operator always needs a CPU native reference.
    ref_input = input.cpu()
    ref_weight = weight.cpu()
    ref_bias = bias.cpu()
    ref_input_scale = input_scale.cpu()
    ref_input_zp = input_zp.cpu()
    ref_weight_scale = weight_scale.cpu()
    ref_weight_zp = weight_zp.cpu()
    ref_output_scale = output_scale.cpu()
    ref_output_zp = output_zp.cpu()
    ref_packed = torch.ops.aten._wrapped_linear_prepack(
        ref_weight, ref_weight_scale, ref_weight_zp, ref_bias
    )
    reference = torch.ops.aten._wrapped_quantized_linear_prepacked(
        ref_input,
        ref_input_scale,
        ref_input_zp,
        ref_packed,
        ref_output_scale,
        ref_output_zp,
        N,
    )

    with flag_gems.use_gems():
        packed = torch.ops.aten._wrapped_linear_prepack(
            weight, weight_scale, weight_zp, bias
        )
        actual = torch.ops.aten._wrapped_quantized_linear_prepacked(
            input,
            input_scale,
            input_zp,
            packed,
            output_scale,
            output_zp,
            N,
        )

    assert actual.shape == reference.shape
    torch.testing.assert_close(
        actual.cpu(), reference, rtol=0.0, atol=float(ref_output_scale) * 1.01
    )


@pytest.mark.wrapped_linear_prepack
@pytest.mark.wrapped_quantized_linear_prepacked
def test_wrapped_quantized_linear_prepacked_halfway_rounding():
    # float32 division and reciprocal multiplication differ for this half-way
    # value. ATen quantize_per_tensor uses reciprocal multiplication.
    input = torch.tensor([[0.775]], device=flag_gems.device)
    weight = torch.tensor([[3.96875]], device=flag_gems.device)
    bias = torch.zeros(1, device=flag_gems.device)
    input_scale = torch.tensor(0.05, device=flag_gems.device)
    input_zp = torch.tensor(127, device=flag_gems.device)
    weight_scale = torch.tensor(0.03125, device=flag_gems.device)
    weight_zp = torch.tensor(0, device=flag_gems.device)
    output_scale = torch.tensor(0.1, device=flag_gems.device)
    output_zp = torch.tensor(0, device=flag_gems.device)

    ref_packed = torch.ops.aten._wrapped_linear_prepack(
        weight.cpu(), weight_scale.cpu(), weight_zp.cpu(), bias.cpu()
    )
    reference = torch.ops.aten._wrapped_quantized_linear_prepacked(
        input.cpu(),
        input_scale.cpu(),
        input_zp.cpu(),
        ref_packed,
        output_scale.cpu(),
        output_zp.cpu(),
        1,
    )

    with flag_gems.use_gems():
        packed = torch.ops.aten._wrapped_linear_prepack(
            weight, weight_scale, weight_zp, bias
        )
        actual = torch.ops.aten._wrapped_quantized_linear_prepacked(
            input,
            input_scale,
            input_zp,
            packed,
            output_scale,
            output_zp,
            1,
        )

    torch.testing.assert_close(actual.cpu(), reference, rtol=0.0, atol=0.0)


@pytest.mark.wrapped_linear_prepack
@pytest.mark.wrapped_quantized_linear_prepacked
def test_wrapped_quantized_linear_prepacked_empty_batch():
    N, K = 5, 7
    input = torch.empty((2, 0, K), device=flag_gems.device)
    weight = torch.randn((N, K), device=flag_gems.device)
    bias = torch.randn((N,), device=flag_gems.device)
    input_scale, input_zp, weight_scale, weight_zp, output_scale, output_zp = (
        _make_qparams(flag_gems.device)
    )

    with flag_gems.use_gems():
        packed = torch.ops.aten._wrapped_linear_prepack(
            weight, weight_scale, weight_zp, bias
        )
        actual = torch.ops.aten._wrapped_quantized_linear_prepacked(
            input,
            input_scale,
            input_zp,
            packed,
            output_scale,
            output_zp,
            N,
        )

    assert actual.shape == (2, 0, N)
    assert actual.numel() == 0


@pytest.mark.wrapped_linear_prepack
@pytest.mark.wrapped_quantized_linear_prepacked
@pytest.mark.parametrize("N,K", [(0, 7), (5, 0)])
def test_wrapped_quantized_linear_prepacked_empty_weight_dimension(N, K):
    input = torch.randn((2, K), device=flag_gems.device)
    weight = torch.randn((N, K), device=flag_gems.device)
    bias = torch.randn((N,), device=flag_gems.device)
    input_scale, input_zp, weight_scale, weight_zp, output_scale, output_zp = (
        _make_qparams(flag_gems.device)
    )
    ref_packed = torch.ops.aten._wrapped_linear_prepack(
        weight.cpu(), weight_scale.cpu(), weight_zp.cpu(), bias.cpu()
    )
    reference = torch.ops.aten._wrapped_quantized_linear_prepacked(
        input.cpu(),
        input_scale.cpu(),
        input_zp.cpu(),
        ref_packed,
        output_scale.cpu(),
        output_zp.cpu(),
        N,
    )

    with flag_gems.use_gems():
        packed = torch.ops.aten._wrapped_linear_prepack(
            weight, weight_scale, weight_zp, bias
        )
        actual = torch.ops.aten._wrapped_quantized_linear_prepacked(
            input,
            input_scale,
            input_zp,
            packed,
            output_scale,
            output_zp,
            N,
        )

    assert actual.shape == reference.shape
    if K == 0:
        # FBGEMM leaves the zero-inner-dimension output buffer undefined. Gems
        # makes the edge case deterministic by returning dequantized uint8 zero.
        expected = -float(output_zp) * float(output_scale)
        torch.testing.assert_close(
            actual, torch.full_like(actual, expected), rtol=0.0, atol=1e-6
        )
    else:
        torch.testing.assert_close(actual.cpu(), reference, rtol=0.0, atol=0.081)


@pytest.mark.wrapped_quantized_linear_prepacked
def test_wrapped_quantized_linear_prepacked_rejects_vector_input():
    N, K = 5, 7
    input = torch.randn((K,), device=flag_gems.device)
    weight = torch.randn((N, K), device=flag_gems.device)
    bias = torch.randn((N,), device=flag_gems.device)
    input_scale, input_zp, weight_scale, weight_zp, output_scale, output_zp = (
        _make_qparams(flag_gems.device)
    )
    packed = flag_gems._wrapped_linear_prepack(weight, weight_scale, weight_zp, bias)

    with pytest.raises(RuntimeError):
        flag_gems._wrapped_quantized_linear_prepacked(
            input,
            input_scale,
            input_zp,
            packed,
            output_scale,
            output_zp,
            N,
        )
