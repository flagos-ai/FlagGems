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


@pytest.mark.fake_quantize_per_tensor_affine
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("scale", [0.03, 0.125])
@pytest.mark.parametrize("quant_min, quant_max", [(0, 255), (-128, 127)])
def test_accuracy_fake_quantize_per_tensor_affine(
    shape, dtype, scale, quant_min, quant_max
):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 4
    zero_point = 3 if quant_min == 0 else -7
    ref = torch.fake_quantize_per_tensor_affine(
        utils.to_reference(input), scale, zero_point, quant_min, quant_max
    )

    with flag_gems.use_gems():
        result = torch.fake_quantize_per_tensor_affine(
            input, scale, zero_point, quant_min, quant_max
        )

    utils.gems_assert_equal(result, ref)


@pytest.mark.fake_quantize_per_tensor_affine
def test_accuracy_fake_quantize_per_tensor_affine_tensor_qparams():
    input = torch.randn((16, 32), dtype=torch.float32, device=flag_gems.device)
    scale = torch.tensor(0.05, dtype=torch.float32, device=flag_gems.device)
    zero_point = torch.tensor(5, dtype=torch.int32, device=flag_gems.device)
    ref = torch.ops.aten.fake_quantize_per_tensor_affine.tensor_qparams(
        utils.to_reference(input),
        utils.to_reference(scale),
        utils.to_reference(zero_point),
        0,
        255,
    )

    with flag_gems.use_gems():
        result = torch.ops.aten.fake_quantize_per_tensor_affine.tensor_qparams(
            input, scale, zero_point, 0, 255
        )

    utils.gems_assert_equal(result, ref)


@pytest.mark.fake_quantize_per_tensor_affine
def test_accuracy_fake_quantize_per_tensor_affine_half_to_even():
    input = torch.tensor(
        [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    ref = torch.fake_quantize_per_tensor_affine(
        utils.to_reference(input), 1.0, 0, -128, 127
    )

    with flag_gems.use_gems():
        result = torch.fake_quantize_per_tensor_affine(input, 1.0, 0, -128, 127)

    utils.gems_assert_equal(result, ref)


@pytest.mark.fake_quantize_per_tensor_affine
def test_accuracy_fake_quantize_per_tensor_affine_noncontiguous():
    input = torch.randn((8, 16), device=flag_gems.device).T
    ref = torch.fake_quantize_per_tensor_affine(
        utils.to_reference(input), 0.1, 0, 0, 255
    )

    with flag_gems.use_gems():
        result = torch.fake_quantize_per_tensor_affine(input, 0.1, 0, 0, 255)

    utils.gems_assert_equal(result, ref)


@pytest.mark.fake_quantize_per_tensor_affine
def test_accuracy_fake_quantize_per_tensor_affine_empty():
    input = torch.empty((2, 0, 3), device=flag_gems.device)

    with flag_gems.use_gems():
        result = torch.fake_quantize_per_tensor_affine(input, 0.1, 0, 0, 255)

    assert result.shape == input.shape
    assert result.dtype == input.dtype


@pytest.mark.fake_quantize_per_tensor_affine
@pytest.mark.parametrize("zero_point, quant_min, quant_max", [(256, 0, 255), (0, 2, 1)])
def test_fake_quantize_per_tensor_affine_invalid_qparams(
    zero_point, quant_min, quant_max
):
    input = torch.ones(2, device=flag_gems.device)

    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        torch.fake_quantize_per_tensor_affine(
            input, 0.1, zero_point, quant_min, quant_max
        )
