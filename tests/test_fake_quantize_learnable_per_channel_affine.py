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

PER_CHANNEL_SHAPES = (
    [((2, 19, 7), 1)]
    if utils.QUICK_MODE
    else [
        ((8,), 0),
        ((4, 8), 0),
        ((4, 8), 1),
        ((2, 19, 7), 1),
        ((2, 19, 7), 2),
        ((16, 128, 64, 60), 1),
        ((16, 128, 64, 60), 3),
    ]
)
QUANT_RANGES = [(0, 255), (-128, 127)]
_ATEN_OP = torch.ops.aten._fake_quantize_learnable_per_channel_affine


def _compare_output(output, reference, inp):
    assert output.device == inp.device
    assert output.shape == inp.shape
    assert output.dtype == inp.dtype
    utils.gems_assert_close(output, reference, inp.dtype)


def _seed():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


def _reference(tensor):
    return utils.to_reference(tensor, True).to(tensor.dtype)


def _make_inputs(shape, axis, dtype):
    _seed()
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    channels = shape[axis]
    scale = torch.rand(channels, device=flag_gems.device) * 0.5 + 0.01
    zero_point = torch.randn(channels, device=flag_gems.device)
    return inp, scale, zero_point


@pytest.mark.underscore_fake_quantize_learnable_per_channel_affine
@pytest.mark.parametrize("shape, axis", PER_CHANNEL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("quant_min, quant_max", QUANT_RANGES)
def test_fake_quantize_learnable_per_channel_affine(
    shape, axis, dtype, quant_min, quant_max
):
    if (
        flag_gems.vendor_name == "kunlunxin"
        and dtype == torch.float16
        and not utils.TO_CPU
    ):
        pytest.skip(
            "Kunlunxin native learnable per-channel Half returns Float instead of Half"
        )
    inputs = _make_inputs(shape, axis, dtype)
    snapshots = tuple(tensor.clone() for tensor in inputs)
    ref_out = _ATEN_OP(
        *(_reference(tensor) for tensor in inputs),
        axis,
        quant_min,
        quant_max,
        1.0,
    )
    result = flag_gems._fake_quantize_learnable_per_channel_affine(
        *inputs, axis, quant_min, quant_max, 1.0
    )

    _compare_output(result, ref_out, inputs[0])
    for value, snapshot in zip(inputs, snapshots):
        flag_gems.testing.assert_equal(value, snapshot)


@pytest.mark.underscore_fake_quantize_learnable_per_channel_affine
@pytest.mark.parametrize("shape, axis", PER_CHANNEL_SHAPES)
def test_fake_quantize_learnable_per_channel_affine_grad_factor(shape, axis):
    inputs = _make_inputs(shape, axis, torch.float32)
    ref_out = _ATEN_OP(
        *(_reference(tensor) for tensor in inputs), axis, 0, 255, 2.5
    )
    result = flag_gems._fake_quantize_learnable_per_channel_affine(
        *inputs, axis, 0, 255, 2.5
    )
    _compare_output(result, ref_out, inputs[0])


@pytest.mark.underscore_fake_quantize_learnable_per_channel_affine
@pytest.mark.parametrize("axis, grad_factor", [(1, 0.5), (-2, 0.5), (-2, 2.5)])
def test_fake_quantize_learnable_per_channel_affine_alternate_grad_factor(axis, grad_factor):
    inputs = _make_inputs((2, 19, 7), axis, torch.float32)
    ref_out = _ATEN_OP(
        *(_reference(tensor) for tensor in inputs),
        axis % inputs[0].ndim,
        0,
        255,
        grad_factor,
    )
    result = flag_gems._fake_quantize_learnable_per_channel_affine(
        *inputs, axis, 0, 255, grad_factor
    )
    _compare_output(result, ref_out, inputs[0])


@pytest.mark.underscore_fake_quantize_learnable_per_channel_affine
def test_fake_quantize_learnable_per_channel_affine_rounds_ties_to_even():
    scale = torch.tensor([0.5, 0.25], device=flag_gems.device, dtype=torch.float32)
    zero_point = torch.zeros_like(scale)
    half_steps = scale.new_tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    inp = half_steps[:, None] * scale
    expected = scale.new_tensor([-2, -2, 0, 0, 2, 2])[:, None] * scale
    result = flag_gems._fake_quantize_learnable_per_channel_affine(
        inp, scale, zero_point, 1, -128, 127, 1.0
    )

    assert result.device == inp.device
    assert result.dtype == inp.dtype
    flag_gems.testing.assert_equal(result, expected)


@pytest.mark.underscore_fake_quantize_learnable_per_channel_affine
@pytest.mark.skipif(flag_gems.vendor_name != "kunlunxin", reason="Kunlunxin regression")
@pytest.mark.parametrize("axis", [0, 1])
def test_fake_quantize_learnable_per_channel_affine_transposed(axis):
    inp = torch.randn((4, 8), device=flag_gems.device).t()
    channels = inp.shape[axis]
    scale = torch.linspace(0.1, 1.0, channels, device=flag_gems.device)
    zero_point = torch.arange(channels, dtype=torch.float32, device=flag_gems.device)
    inputs = inp, scale, zero_point
    reference = _ATEN_OP(
        *(_reference(tensor) for tensor in inputs), axis, -128, 127, 1.0
    )
    result = flag_gems._fake_quantize_learnable_per_channel_affine(*inputs, axis, -128, 127, 1.0)
    _compare_output(result, reference, inp)


@pytest.mark.underscore_fake_quantize_learnable_per_channel_affine
def test_fake_quantize_learnable_per_channel_affine_autograd(request):
    _seed()
    inp = torch.randn((4, 8), device=flag_gems.device, requires_grad=True)
    scale = (torch.rand(8, device=flag_gems.device) * 0.5 + 0.01).requires_grad_()
    zero_point = torch.randn(8, device=flag_gems.device, requires_grad=True)
    upstream = torch.randn_like(inp)
    ref_inp = _reference(inp.detach()).requires_grad_()
    ref_scale = _reference(scale.detach()).requires_grad_()
    ref_zero_point = _reference(zero_point.detach()).requires_grad_()
    ref_upstream = _reference(upstream)
    ref_out = _ATEN_OP(ref_inp, ref_scale, ref_zero_point, 1, -128, 127, 0.5)
    ref_grads = torch.autograd.grad(
        ref_out, (ref_inp, ref_scale, ref_zero_point), ref_upstream
    )

    registry = request.config._override_registry
    if registry.get_override("_fake_quantize_learnable_per_channel_affine") is None:
        with flag_gems.use_gems():
            output = _ATEN_OP(inp, scale, zero_point, 1, -128, 127, 0.5)
            grads = torch.autograd.grad(output, (inp, scale, zero_point), upstream)
    else:
        output = flag_gems._fake_quantize_learnable_per_channel_affine(
            inp, scale, zero_point, 1, -128, 127, 0.5
        )
        grads = torch.autograd.grad(output, (inp, scale, zero_point), upstream)

    assert output.device == inp.device
    assert all(actual.device == inp.device for actual in grads)
    utils.gems_assert_close(output, ref_out, torch.float32)
    utils.gems_assert_close(grads[0], ref_grads[0], torch.float32)
    utils.gems_assert_close(grads[1], ref_grads[1], torch.float32, reduce_dim=4)
    utils.gems_assert_close(grads[2], ref_grads[2], torch.float32, reduce_dim=4)


@pytest.mark.underscore_fake_quantize_learnable_per_channel_affine
def test_fake_quantize_learnable_per_channel_affine_rejects_integer_input():
    inp = torch.ones((4, 8), dtype=torch.int32, device=flag_gems.device)
    scale = torch.ones(8, device=flag_gems.device)
    zero_point = torch.zeros(8, device=flag_gems.device)
    with pytest.raises(TypeError):
        flag_gems._fake_quantize_learnable_per_channel_affine(
            inp, scale, zero_point, 1, -128, 127, 1.0
        )
