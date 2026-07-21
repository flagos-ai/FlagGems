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

CORE_CASES = [
    ((64, 128), (128,)),
    ((256, 512), (512,)),
    ((1024, 512), (512,)),
    ((2048, 512), (512,)),
    ((16, 32, 16, 32), (16, 32)),
    ((1024, 513), (513,)),
]

BOUNDARY_CASES = [
    ((1, 32), (32,)),
    ((2, 3, 4), (3, 4)),
]

AFFINE_MODES = ("both", "weight", "bias", "none")


def _configure_backward_test_env(monkeypatch):
    if flag_gems.vendor_name == "mthreads":
        # Keep compatibility with the older LLVM stack used by this backend.
        monkeypatch.setenv("DISABLE_LLVM_OPT", "1")


def _make_affine(normalized_shape, dtype, mode):
    weight = None
    bias = None
    if mode in ("both", "weight"):
        weight = torch.randn(normalized_shape, dtype=dtype, device=flag_gems.device)
    if mode in ("both", "bias"):
        bias = torch.randn(normalized_shape, dtype=dtype, device=flag_gems.device)
    return weight, bias


def _reference(input, residual, normalized_shape, weight, bias, eps):
    return torch.layer_norm(
        utils.to_reference(input, True),
        normalized_shape,
        utils.to_reference(weight, True),
        utils.to_reference(bias, True),
        eps,
    ) + utils.to_reference(residual, True)


def _make_grad_tensors(shape, normalized_shape, dtype, affine_mode="both"):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    residual = torch.randn_like(input, requires_grad=True)
    weight, bias = _make_affine(normalized_shape, dtype, affine_mode)
    if weight is not None:
        weight.requires_grad_(True)
    if bias is not None:
        bias.requires_grad_(True)
    grad_output = torch.randn_like(input)

    ref_input = utils.to_reference(input.detach().clone(), True).requires_grad_(True)
    ref_residual = utils.to_reference(residual.detach().clone(), True).requires_grad_(
        True
    )
    ref_weight = (
        utils.to_reference(weight.detach().clone(), True).requires_grad_(True)
        if weight is not None
        else None
    )
    ref_bias = (
        utils.to_reference(bias.detach().clone(), True).requires_grad_(True)
        if bias is not None
        else None
    )
    ref_grad_output = utils.to_reference(grad_output, True)
    return (
        input,
        residual,
        weight,
        bias,
        grad_output,
        ref_input,
        ref_residual,
        ref_weight,
        ref_bias,
        ref_grad_output,
    )


def _assert_backward_close(
    shape,
    normalized_shape,
    dtype,
    affine_mode="both",
):
    (
        input,
        residual,
        weight,
        bias,
        grad_output,
        ref_input,
        ref_residual,
        ref_weight,
        ref_bias,
        ref_grad_output,
    ) = _make_grad_tensors(shape, normalized_shape, dtype, affine_mode)

    expected = (
        torch.layer_norm(ref_input, normalized_shape, ref_weight, ref_bias, 1e-5)
        + ref_residual
    )
    actual = flag_gems.post_layer_norm_residual(
        input, residual, normalized_shape, weight, bias, 1e-5
    )

    expected.backward(ref_grad_output)
    actual.backward(grad_output)

    M = math.prod(shape) // math.prod(normalized_shape)
    utils.gems_assert_close(input.grad, ref_input.grad, dtype)
    utils.gems_assert_close(residual.grad, ref_residual.grad, dtype)
    if weight is not None:
        utils.gems_assert_close(weight.grad, ref_weight.grad, dtype, reduce_dim=M)
    if bias is not None:
        utils.gems_assert_close(bias.grad, ref_bias.grad, dtype, reduce_dim=M)


@pytest.mark.post_layer_norm_residual
@pytest.mark.parametrize("shape,normalized_shape", CORE_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_post_layer_norm_residual_forward(shape, normalized_shape, dtype):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    residual = torch.randn_like(input)
    weight, bias = _make_affine(normalized_shape, dtype, "both")

    expected = _reference(input, residual, normalized_shape, weight, bias, 1e-5)
    actual = flag_gems.post_layer_norm_residual(
        input, residual, normalized_shape, weight, bias, 1e-5
    )

    utils.gems_assert_close(actual, expected, dtype)


@pytest.mark.post_layer_norm_residual
def test_post_layer_norm_residual_no_grad_with_grad_inputs():
    shape = (64, 128)
    normalized_shape = (128,)
    dtype = torch.float32
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    residual = torch.randn_like(input, requires_grad=True)
    weight, bias = _make_affine(normalized_shape, dtype, "both")
    weight.requires_grad_(True)
    bias.requires_grad_(True)

    expected = _reference(input, residual, normalized_shape, weight, bias, 1e-5)
    with torch.no_grad():
        actual = flag_gems.post_layer_norm_residual(
            input, residual, normalized_shape, weight, bias, 1e-5
        )

    assert not actual.requires_grad
    utils.gems_assert_close(actual, expected, dtype)


@pytest.mark.post_layer_norm_residual
@pytest.mark.parametrize("shape,normalized_shape", BOUNDARY_CASES)
@pytest.mark.parametrize("affine_mode", AFFINE_MODES)
@pytest.mark.parametrize("eps", (1e-5, 1e-6))
def test_post_layer_norm_residual_optional_affine(
    shape, normalized_shape, affine_mode, eps
):
    dtype = torch.float32
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    residual = torch.randn_like(input)
    weight, bias = _make_affine(normalized_shape, dtype, affine_mode)

    expected = _reference(input, residual, normalized_shape, weight, bias, eps)
    actual = flag_gems.post_layer_norm_residual(
        input, residual, normalized_shape, weight, bias, eps
    )

    utils.gems_assert_close(actual, expected, dtype)


@pytest.mark.post_layer_norm_residual
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_post_layer_norm_residual_large_normalized_shape(dtype):
    shape = (2, 4097)
    normalized_shape = (4097,)
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    residual = torch.randn_like(input)
    weight, bias = _make_affine(normalized_shape, dtype, "both")

    expected = _reference(input, residual, normalized_shape, weight, bias, 1e-5)
    actual = flag_gems.post_layer_norm_residual(
        input, residual, normalized_shape, weight, bias, 1e-5
    )

    utils.gems_assert_close(actual, expected, dtype)


@pytest.mark.post_layer_norm_residual
def test_post_layer_norm_residual_noncontiguous_fallback():
    dtype = torch.float32
    input = torch.randn((8, 4), dtype=dtype, device=flag_gems.device).T
    residual = torch.randn((8, 4), dtype=dtype, device=flag_gems.device).T
    normalized_shape = (8,)

    expected = _reference(input, residual, normalized_shape, None, None, 1e-5)
    actual = flag_gems.post_layer_norm_residual(
        input, residual, normalized_shape, None, None, 1e-5
    )

    utils.gems_assert_close(actual, expected, dtype)


@pytest.mark.post_layer_norm_residual
@pytest.mark.parametrize(
    "input_shape,residual_shape,error",
    [
        ((4, 8), (4, 7), "same shape"),
        ((4, 8), (4, 8), "same dtype"),
    ],
)
def test_post_layer_norm_residual_rejects_mismatched_metadata(
    input_shape, residual_shape, error
):
    input = torch.randn(input_shape, dtype=torch.float32, device=flag_gems.device)
    residual_dtype = torch.float16 if error == "same dtype" else torch.float32
    residual = torch.randn(
        residual_shape, dtype=residual_dtype, device=flag_gems.device
    )

    with pytest.raises(ValueError, match=error):
        flag_gems.post_layer_norm_residual(input, residual, (8,))


@pytest.mark.post_layer_norm_residual
def test_post_layer_norm_residual_rejects_mismatched_device():
    input = torch.randn((4, 8), dtype=torch.float32, device=flag_gems.device)
    residual = torch.randn((4, 8), dtype=torch.float32, device="cpu")

    with pytest.raises(ValueError, match="same device"):
        flag_gems.post_layer_norm_residual(input, residual, (8,))


@pytest.mark.post_layer_norm_residual
def test_post_layer_norm_residual_validates_normalized_shape():
    input = torch.randn((4, 8), dtype=torch.float32, device=flag_gems.device)
    residual = torch.randn_like(input)

    with pytest.raises(ValueError, match="normalized_shape"):
        flag_gems.post_layer_norm_residual(input, residual, (4,))


@pytest.mark.post_layer_norm_residual
@pytest.mark.parametrize("shape,normalized_shape", CORE_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_post_layer_norm_residual_backward(monkeypatch, shape, normalized_shape, dtype):
    _configure_backward_test_env(monkeypatch)
    _assert_backward_close(shape, normalized_shape, dtype)


@pytest.mark.post_layer_norm_residual
@pytest.mark.parametrize("shape,normalized_shape", BOUNDARY_CASES)
@pytest.mark.parametrize("affine_mode", AFFINE_MODES)
def test_post_layer_norm_residual_backward_optional_affine(
    monkeypatch, shape, normalized_shape, affine_mode
):
    _configure_backward_test_env(monkeypatch)
    _assert_backward_close(
        shape, normalized_shape, torch.float32, affine_mode=affine_mode
    )


@pytest.mark.post_layer_norm_residual
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_post_layer_norm_residual_backward_large_normalized_shape(monkeypatch, dtype):
    _configure_backward_test_env(monkeypatch)
    _assert_backward_close((2, 4097), (4097,), dtype)


@pytest.mark.post_layer_norm_residual
def test_post_layer_norm_residual_backward_residual_only(monkeypatch):
    _configure_backward_test_env(monkeypatch)
    shape = (16, 64)
    input = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    residual = torch.randn_like(input, requires_grad=True)
    weight, bias = _make_affine((64,), torch.float32, "both")
    grad_output = torch.randn_like(input)

    output = flag_gems.post_layer_norm_residual(
        input, residual, (64,), weight, bias, 1e-5
    )
    (grad_residual,) = torch.autograd.grad(output, residual, grad_output)

    utils.gems_assert_close(grad_residual, grad_output, torch.float32)
