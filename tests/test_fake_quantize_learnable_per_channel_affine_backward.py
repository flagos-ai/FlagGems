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
from . import conftest as cfg

PER_CHANNEL_SHAPES = [(4, 8), (16, 64)] if cfg.QUICK_MODE else [
    (8,),
    (4, 16),
    (16, 64),
    (20, 32, 15),
    (16, 128, 64, 60),
]
AXIS_LIST = [0, 1] if cfg.QUICK_MODE else [0, 1, 2]
PER_CHANNEL_CASES = [
    (shape, axis)
    for shape in PER_CHANNEL_SHAPES
    for axis in AXIS_LIST
    if axis < len(shape)
]
QUANT_RANGES = [(-128, 127), (0, 255), (-7, 7)]
GRAD_FACTORS = [1.0] if cfg.QUICK_MODE else [1.0, 0.5, 2.0]
_ATEN_OP = torch.ops.aten._fake_quantize_learnable_per_channel_affine_backward


def _seed():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


def _reference(tensor):
    return utils.to_reference(tensor, True).to(torch.float32)


def _make_inputs(shape, axis, quant_max):
    _seed()
    channels = shape[axis]
    scale = torch.rand(channels, device=flag_gems.device) * 0.4 + 0.05
    zero_point = torch.rand(channels, device=flag_gems.device) * (quant_max * 0.5)
    self_t = torch.randn(shape, device=flag_gems.device) * (float(quant_max) * 0.05)
    grad = torch.randn(shape, device=flag_gems.device)
    return grad, self_t, scale, zero_point


@pytest.mark.fake_quantize_learnable_per_channel_affine_backward
@pytest.mark.underscore_fake_quantize_learnable_per_channel_affine_backward
@pytest.mark.parametrize("shape, axis", PER_CHANNEL_CASES)
@pytest.mark.parametrize("quant_min, quant_max", QUANT_RANGES)
@pytest.mark.parametrize("grad_factor", GRAD_FACTORS)
def test_fake_quantize_learnable_per_channel_affine_backward(
    shape, axis, quant_min, quant_max, grad_factor
):
    inputs = _make_inputs(shape, axis, quant_max)
    snapshots = tuple(tensor.clone() for tensor in inputs)
    grad, self_t, scale, zero_point = inputs
    ref_out = _ATEN_OP(
        *(_reference(tensor) for tensor in inputs),
        axis,
        quant_min,
        quant_max,
        grad_factor,
    )
    result = flag_gems._fake_quantize_learnable_per_channel_affine_backward(
        grad,
        self_t,
        scale,
        zero_point,
        axis,
        quant_min,
        quant_max,
        grad_factor,
    )

    reduce_size = self_t.numel() // shape[axis]
    for index, (actual, expected) in enumerate(zip(result, ref_out)):
        utils.gems_assert_close(
            actual, expected, torch.float32, reduce_dim=1 if index == 0 else reduce_size
        )
    for value, snapshot in zip(inputs, snapshots):
        utils.gems_assert_equal(value, snapshot)
    assert result[0].shape == self_t.shape
    assert result[1].shape == scale.shape
    assert result[2].shape == zero_point.shape
    assert all(value.dtype == torch.float32 for value in result)


@pytest.mark.skipif(
    cfg.TO_CPU, reason="CPU reference diverges on infinite saturation gradients"
)
@pytest.mark.fake_quantize_learnable_per_channel_affine_backward
@pytest.mark.underscore_fake_quantize_learnable_per_channel_affine_backward
def test_fake_quantize_learnable_per_channel_affine_backward_extreme():
    shape, axis = (4, 16), 1
    scale = torch.full((16,), 0.1, device=flag_gems.device)
    zero_point = torch.zeros(16, device=flag_gems.device)
    self_t = torch.zeros(shape, device=flag_gems.device)
    self_t[:, 0] = float("inf")
    self_t[:, 1] = float("-inf")
    _seed()
    grad = torch.randn(shape, device=flag_gems.device)
    inputs = grad, self_t, scale, zero_point
    ref_out = _ATEN_OP(
        *(_reference(tensor) for tensor in inputs), axis, -128, 127, 1.0
    )
    result = flag_gems._fake_quantize_learnable_per_channel_affine_backward(
        *inputs, axis, -128, 127, 1.0
    )

    for index, (actual, expected) in enumerate(zip(result, ref_out)):
        utils.gems_assert_close(
            actual,
            expected,
            torch.float32,
            reduce_dim=1 if index == 0 else shape[0],
            equal_nan=True,
        )


@pytest.mark.fake_quantize_learnable_per_channel_affine_backward
@pytest.mark.underscore_fake_quantize_learnable_per_channel_affine_backward
def test_fake_quantize_learnable_per_channel_affine_backward_empty():
    shape, axis = (2, 0, 3), 1
    grad = torch.empty(shape, device=flag_gems.device)
    self_t = torch.empty(shape, device=flag_gems.device)
    scale = torch.empty(0, device=flag_gems.device)
    zero_point = torch.empty(0, device=flag_gems.device)
    result = flag_gems._fake_quantize_learnable_per_channel_affine_backward(
        grad, self_t, scale, zero_point, axis, -128, 127, 1.0
    )

    assert result[0].shape == self_t.shape
    assert result[1].shape == scale.shape
    assert result[2].shape == zero_point.shape
    utils.gems_assert_equal(result[1], torch.zeros_like(scale))
    utils.gems_assert_equal(result[2], torch.zeros_like(zero_point))


@pytest.mark.fake_quantize_learnable_per_channel_affine_backward
@pytest.mark.underscore_fake_quantize_learnable_per_channel_affine_backward
def test_fake_quantize_learnable_per_channel_affine_backward_rejects_dtype():
    shape, axis = (4, 8), 1
    tensors = [
        torch.ones(shape, dtype=torch.float16, device=flag_gems.device),
        torch.ones(shape, device=flag_gems.device),
        torch.ones(8, device=flag_gems.device),
        torch.zeros(8, device=flag_gems.device),
    ]
    with pytest.raises(AssertionError):
        flag_gems._fake_quantize_learnable_per_channel_affine_backward(
            *tensors, axis, -128, 127, 1.0
        )
