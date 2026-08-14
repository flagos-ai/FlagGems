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
from .conftest import QUICK_MODE

FEATURE_ALPHA_DROPOUT_INPLACE_SHAPES = (
    [(8, 16, 16, 16)]
    if QUICK_MODE
    else [(2, 3), (4, 8, 16), (8, 16, 16, 16), (2, 32, 4, 4, 4)]
)

_ALPHA = 1.7580993408473766


def _assert_feature_alpha_dropout_result(output, original, p, dtype):
    scale = 1.0 / math.sqrt((_ALPHA * _ALPHA * p + 1.0) * (1.0 - p))
    shift = _ALPHA * scale * p
    dropped_value = _ALPHA * scale * (p - 1.0)
    batch_size, n_channels = original.shape[:2]
    input_by_feature = original.reshape(batch_size, n_channels, -1).float()
    output_by_feature = output.reshape(batch_size, n_channels, -1).float()
    atol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4

    dropped = 0
    for batch in range(batch_size):
        for channel in range(n_channels):
            channel_output = output_by_feature[batch, channel]
            expected_dropped = torch.full_like(channel_output, dropped_value)
            if torch.allclose(channel_output, expected_dropped, rtol=0, atol=atol):
                dropped += 1
            else:
                expected_kept = input_by_feature[batch, channel] * scale + shift
                assert torch.allclose(
                    channel_output, expected_kept, rtol=1e-3, atol=atol
                )

    total = batch_size * n_channels
    tolerance = max(0.2, 3.0 * math.sqrt(p * (1.0 - p) / total))
    assert abs(dropped / total - p) <= tolerance


@pytest.mark.inplace
@pytest.mark.feature_alpha_dropout_
@pytest.mark.parametrize("shape", FEATURE_ALPHA_DROPOUT_INPLACE_SHAPES)
@pytest.mark.parametrize("p", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_feature_alpha_dropout_(shape, p, dtype):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    original = input.clone()
    data_ptr = input.data_ptr()
    with flag_gems.use_gems():
        output = torch.ops.aten.feature_alpha_dropout_(input, p, True)

    assert output is input
    assert output.data_ptr() == data_ptr
    assert output.dtype == original.dtype
    _assert_feature_alpha_dropout_result(output, original, p, dtype)


@pytest.mark.inplace
@pytest.mark.feature_alpha_dropout_
@pytest.mark.parametrize("p, train", [(0.0, True), (0.5, False)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_feature_alpha_dropout__identity(p, train, dtype):
    input = torch.randn((2, 3, 4), dtype=dtype, device=flag_gems.device)
    original = utils.to_reference(input.clone())
    data_ptr = input.data_ptr()
    with flag_gems.use_gems():
        output = torch.ops.aten.feature_alpha_dropout_(input, p, train)

    assert output is input
    assert output.data_ptr() == data_ptr
    utils.gems_assert_equal(output, original)


@pytest.mark.inplace
@pytest.mark.feature_alpha_dropout_
def test_feature_alpha_dropout__p_one_special_values():
    input = torch.tensor(
        [[[float("nan"), float("inf"), -float("inf"), -0.0, 1.0, -1.0]]],
        device=flag_gems.device,
    )
    reference = utils.to_reference(input.clone())
    torch.ops.aten.feature_alpha_dropout_(reference, 1.0, True)
    data_ptr = input.data_ptr()
    with flag_gems.use_gems():
        output = torch.ops.aten.feature_alpha_dropout_(input, 1.0, True)

    assert output is input
    assert output.data_ptr() == data_ptr
    utils.gems_assert_equal(output, reference, equal_nan=True)
    # CPU and CUDA can produce NaNs with different sign bits for inf * 0. NaN
    # sign is not part of the operator contract, but signed zero still is.
    output_not_nan = ~torch.isnan(output)
    reference_not_nan = ~torch.isnan(reference)
    utils.gems_assert_equal(
        torch.signbit(output)[output_not_nan],
        torch.signbit(reference)[reference_not_nan],
    )


@pytest.mark.inplace
@pytest.mark.feature_alpha_dropout_
def test_feature_alpha_dropout__empty_returns_input():
    input = torch.empty((0,), device=flag_gems.device)
    data_ptr = input.data_ptr()
    with flag_gems.use_gems():
        output = torch.ops.aten.feature_alpha_dropout_(input, 0.5, True)

    assert output is input
    assert output.data_ptr() == data_ptr


@pytest.mark.inplace
@pytest.mark.feature_alpha_dropout_
def test_feature_alpha_dropout__noncontiguous():
    input = torch.randn((16, 32, 8, 8), device=flag_gems.device).transpose(1, 2)
    original = input.clone()
    data_ptr = input.data_ptr()
    stride = input.stride()
    with flag_gems.use_gems():
        output = torch.ops.aten.feature_alpha_dropout_(input, 0.5, True)

    assert output is input
    assert output.data_ptr() == data_ptr
    assert output.stride() == stride
    _assert_feature_alpha_dropout_result(output, original, 0.5, torch.float32)


@pytest.mark.inplace
@pytest.mark.feature_alpha_dropout_
@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan")])
@pytest.mark.parametrize("train", [True, False])
def test_feature_alpha_dropout__invalid_probability(p, train):
    input = torch.randn((2, 3), device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        torch.ops.aten.feature_alpha_dropout_(input, p, train)


@pytest.mark.inplace
@pytest.mark.feature_alpha_dropout_
def test_feature_alpha_dropout__requires_two_dimensions():
    input = torch.randn((8,), device=flag_gems.device)
    with (
        flag_gems.use_gems(),
        pytest.raises(
            RuntimeError, match="Feature dropout requires at least 2 dimensions"
        ),
    ):
        torch.ops.aten.feature_alpha_dropout_(input, 0.5, True)
