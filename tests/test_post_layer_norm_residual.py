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
def test_post_layer_norm_residual_large_normalized_shape():
    shape = (2, 4097)
    normalized_shape = (4097,)
    dtype = torch.float32
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
