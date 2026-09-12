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

DEFAULT_LOWER = 0.125
DEFAULT_UPPER = 1.0 / 3.0


def _bounds(training):
    # Equal bounds remove random-number differences between reference and
    # FlagGems while still exercising the training branch.
    return (0.25, 0.25) if training else (DEFAULT_LOWER, DEFAULT_UPPER)


def _run(op_name, self, noise, lower, upper, training, generator=None):
    op = getattr(flag_gems, op_name)
    return op(self, noise, lower, upper, training, generator)


def _training_sample_mask(inp):
    # torch_npu records a unit slope at signed zero, while the CPU/CUDA ATen
    # implementation samples signed zero. Match the native reference selected
    # by FlagGems on each platform.
    return inp < 0 if flag_gems.vendor_name == "ascend" else inp <= 0


def _assert_training_contract(result, original, noise, lower, upper):
    sampled = _training_sample_mask(original)
    not_sampled = ~sampled
    lower_bound = torch.tensor(lower, dtype=noise.dtype, device=noise.device)
    upper_bound = torch.tensor(upper, dtype=noise.dtype, device=noise.device)

    assert torch.all(noise[sampled] >= lower_bound)
    assert torch.all(noise[sampled] <= upper_bound)
    utils.gems_assert_equal(noise[not_sampled], torch.ones_like(noise[not_sampled]))

    expected = torch.where(sampled, original * noise, original)
    utils.gems_assert_close(result, expected, original.dtype)


@pytest.mark.rrelu_with_noise
@pytest.mark.parametrize("op_name", ["rrelu_with_noise", "rrelu_with_noise_"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_rrelu_with_noise(op_name, training, shape, dtype):
    lower, upper = _bounds(training)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    noise = torch.zeros_like(inp)
    if not training:
        noise.uniform_(lower, upper)

    ref_inp = utils.to_reference(inp.clone())
    ref_noise = utils.to_reference(noise.clone())
    ref_result = _run(op_name, ref_inp, ref_noise, lower, upper, training)

    result = _run(op_name, inp, noise, lower, upper, training)

    # This checks the public alias contract. It cannot by itself distinguish a
    # direct out0 write from a temporary followed by copy_, so the kernel call
    # must also be reviewed/covered by the implementation path below.
    if op_name.endswith("_"):
        assert result.data_ptr() == inp.data_ptr()
    utils.gems_assert_close(result, ref_result, dtype)
    utils.gems_assert_close(noise, ref_noise, dtype)


@pytest.mark.rrelu_with_noise
@pytest.mark.parametrize("op_name", ["rrelu_with_noise", "rrelu_with_noise_"])
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_rrelu_with_noise_training_random_contract(op_name, dtype):
    lower, upper = DEFAULT_LOWER, DEFAULT_UPPER
    original = torch.linspace(-2.0, 2.0, 4097, dtype=dtype, device=flag_gems.device)
    inp = original.clone()
    noise = torch.zeros_like(inp)

    result = _run(op_name, inp, noise, lower, upper, True)

    sampled = _training_sample_mask(original)
    _assert_training_contract(result, original, noise, lower, upper)
    sampled_noise = noise[sampled]
    assert torch.any(sampled_noise != sampled_noise[0])
    if not op_name.endswith("_"):
        utils.gems_assert_equal(inp, original)


@pytest.mark.rrelu_with_noise
@pytest.mark.parametrize("op_name", ["rrelu_with_noise", "rrelu_with_noise_"])
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_rrelu_with_noise_generator_reproducibility(op_name, dtype):
    lower, upper = DEFAULT_LOWER, DEFAULT_UPPER
    original = -torch.ones((4096,), dtype=dtype, device=flag_gems.device)

    generators = []
    for seed in (2026, 2026, 2027):
        generator = torch.Generator(device=flag_gems.device)
        generator.manual_seed(seed)
        generators.append(generator)

    outputs = []
    noises = []
    for generator in generators:
        inp = original.clone()
        noise = torch.zeros_like(inp)
        outputs.append(_run(op_name, inp, noise, lower, upper, True, generator))
        noises.append(noise)

    utils.gems_assert_equal(outputs[0], outputs[1])
    utils.gems_assert_equal(noises[0], noises[1])
    assert not torch.equal(noises[0], noises[2])

    advanced_input = original.clone()
    advanced_noise = torch.zeros_like(advanced_input)
    advanced_output = _run(
        op_name,
        advanced_input,
        advanced_noise,
        lower,
        upper,
        True,
        generators[0],
    )

    assert not torch.equal(noises[0], advanced_noise)
    _assert_training_contract(advanced_output, original, advanced_noise, lower, upper)


@pytest.mark.rrelu_with_noise
@pytest.mark.parametrize("op_name", ["rrelu_with_noise", "rrelu_with_noise_"])
def test_rrelu_with_noise_eval_does_not_advance_generator(op_name):
    generator = torch.Generator(device=flag_gems.device)
    generator.manual_seed(2026)
    state_before = generator.get_state().clone()
    inp = torch.randn((257,), device=flag_gems.device)
    noise = torch.randn_like(inp)

    _run(
        op_name,
        inp,
        noise,
        DEFAULT_LOWER,
        DEFAULT_UPPER,
        False,
        generator,
    )

    assert torch.equal(generator.get_state(), state_before)


@pytest.mark.rrelu_with_noise
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_rrelu_with_noise_inplace_alias(training, dtype):
    lower, upper = _bounds(training)
    inp = torch.randn((37,), dtype=dtype, device=flag_gems.device)
    noise = torch.zeros_like(inp)
    if not training:
        noise.uniform_(lower, upper)
    input_ptr = inp.data_ptr()

    result = _run("rrelu_with_noise_", inp, noise, lower, upper, training)

    assert result.data_ptr() == input_ptr


@pytest.mark.rrelu_with_noise
@pytest.mark.parametrize("op_name", ["rrelu_with_noise", "rrelu_with_noise_"])
def test_rrelu_with_noise_training_mask(op_name):
    # NaN takes the non-sampled path and records one. Signed-zero behavior is
    # checked against the platform's native aten implementation.
    dtype = torch.float32
    lower = upper = 0.25
    values = [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.0, -1.0]
    inp = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    noise = torch.zeros_like(inp)

    ref_inp = utils.to_reference(inp.clone())
    ref_noise = torch.zeros_like(ref_inp)
    ref_result = _run(op_name, ref_inp, ref_noise, lower, upper, True)

    result = _run(op_name, inp, noise, lower, upper, True)

    utils.gems_assert_close(result, ref_result, dtype, equal_nan=True)
    utils.gems_assert_close(noise, ref_noise, dtype)


@pytest.mark.rrelu_with_noise
@pytest.mark.parametrize("op_name", ["rrelu_with_noise", "rrelu_with_noise_"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_rrelu_with_noise_non_contiguous(op_name, training, dtype):
    lower, upper = (DEFAULT_LOWER, DEFAULT_UPPER) if training else _bounds(training)
    input_base = torch.linspace(
        -2.0,
        2.0,
        17 * 22,
        dtype=dtype,
        device=flag_gems.device,
    ).reshape(17, 22)
    noise_base = torch.zeros_like(input_base)
    input_untouched = input_base[:, 1::2].clone()
    noise_untouched = noise_base[:, 1::2].clone()
    inp = input_base[:, ::2]
    noise = noise_base[:, ::2]
    assert not inp.is_contiguous()
    assert not noise.is_contiguous()

    original = inp.clone()
    if not training:
        ref_input_base = utils.to_reference(input_base.clone())
        ref_noise_base = utils.to_reference(noise_base.clone())
        ref_inp = ref_input_base[:, ::2]
        ref_noise = ref_noise_base[:, ::2]
        ref_result = _run(op_name, ref_inp, ref_noise, lower, upper, training)

    result = _run(op_name, inp, noise, lower, upper, training)

    if training:
        _assert_training_contract(result, original, noise, lower, upper)
        sampled_noise = noise[_training_sample_mask(original)]
        assert torch.any(sampled_noise != sampled_noise[0])
    else:
        utils.gems_assert_close(result, ref_result, dtype)
    utils.gems_assert_equal(input_base[:, 1::2], input_untouched)
    utils.gems_assert_equal(noise_base[:, 1::2], noise_untouched)


@pytest.mark.rrelu_with_noise
@pytest.mark.parametrize("op_name", ["rrelu_with_noise", "rrelu_with_noise_"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("shape", [(0,), (0, 7), (2, 0, 3)])
def test_rrelu_with_noise_empty(op_name, training, shape):
    inp = torch.empty(shape, device=flag_gems.device)
    noise = torch.empty_like(inp)
    input_ptr = inp.data_ptr()

    result = _run(op_name, inp, noise, DEFAULT_LOWER, DEFAULT_UPPER, training)

    assert result.shape == inp.shape
    assert result.dtype == inp.dtype
    if op_name.endswith("_"):
        assert result.data_ptr() == input_ptr


@pytest.mark.rrelu_with_noise
@pytest.mark.parametrize("op_name", ["rrelu_with_noise", "rrelu_with_noise_"])
def test_rrelu_with_noise_eval_does_not_modify_noise(op_name):
    inp = torch.randn((257,), device=flag_gems.device)
    noise = torch.randn_like(inp)
    noise_before = noise.clone()
    input_before = inp.clone()

    _run(op_name, inp, noise, DEFAULT_LOWER, DEFAULT_UPPER, False)

    utils.gems_assert_equal(noise, noise_before)
    if not op_name.endswith("_"):
        utils.gems_assert_equal(inp, input_before)


@pytest.mark.rrelu_with_noise
@pytest.mark.parametrize("training", [False, True])
def test_rrelu_with_noise_autograd(training):
    dtype = torch.float32
    lower, upper = _bounds(training)
    inp = torch.randn((257,), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.clone()).requires_grad_()
    ref_noise = torch.zeros_like(ref_inp)
    ref_out = _run("rrelu_with_noise", ref_inp, ref_noise, lower, upper, training)
    ref_out.sum().backward()

    gems_inp = inp.clone().requires_grad_()
    gems_noise = torch.zeros_like(gems_inp)
    gems_out = _run(
        "rrelu_with_noise", gems_inp, gems_noise, lower, upper, training
    )
    gems_out.sum().backward()

    utils.gems_assert_close(gems_inp.grad, ref_inp.grad, dtype)


@pytest.mark.rrelu_with_noise
@pytest.mark.parametrize("training", [False, True])
def test_rrelu_with_noise_inplace_autograd_non_leaf(training):
    # A leaf requiring grad is correctly rejected by PyTorch for any in-place
    # operator. A non-leaf requiring grad is the legal path and exercises the
    # self_is_result=True backward route.
    dtype = torch.float32
    lower, upper = _bounds(training)
    source = torch.randn((257,), dtype=dtype, device=flag_gems.device)

    ref_leaf = utils.to_reference(source.clone()).requires_grad_()
    ref_self = ref_leaf * 1.0
    ref_noise = torch.zeros_like(ref_self)
    ref_out = _run("rrelu_with_noise_", ref_self, ref_noise, lower, upper, training)
    ref_out.sum().backward()

    gems_leaf = source.clone().requires_grad_()
    gems_self = gems_leaf * 1.0
    gems_noise = torch.zeros_like(gems_self)
    gems_out = _run(
        "rrelu_with_noise_", gems_self, gems_noise, lower, upper, training
    )
    gems_out.sum().backward()

    utils.gems_assert_close(gems_leaf.grad, ref_leaf.grad, dtype)


@pytest.mark.rrelu_with_noise
@pytest.mark.parametrize("training", [False, True])
def test_rrelu_with_noise_backward_self_is_result(training):
    # Explicitly cover the backward variant used when the forward op was
    # in-place and the saved self tensor is the result tensor.
    dtype = torch.float32
    lower, upper = _bounds(training)
    grad_output = torch.randn((257,), dtype=dtype, device=flag_gems.device)
    result = torch.randn((257,), dtype=dtype, device=flag_gems.device)
    noise = torch.full_like(result, lower)

    ref_grad = utils.to_reference(grad_output)
    ref_result = utils.to_reference(result)
    ref_noise = utils.to_reference(noise)
    ref_out = torch.ops.aten.rrelu_with_noise_backward(
        ref_grad, ref_result, ref_noise, lower, upper, training, True
    )

    gems_out = torch.ops.aten.rrelu_with_noise_backward(
        grad_output, result, noise, lower, upper, training, True
    )

    utils.gems_assert_close(gems_out, ref_out, dtype)
