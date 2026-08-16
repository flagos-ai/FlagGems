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


@pytest.mark.fake_quantize_per_tensor_affine_cachemask
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("scale", [0.03, 0.125])
@pytest.mark.parametrize("quant_min, quant_max", [(0, 255), (-128, 127)])
def test_accuracy_fake_quantize_per_tensor_affine_cachemask(
    shape, dtype, scale, quant_min, quant_max
):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 4
    zero_point = 3 if quant_min == 0 else -7
    ref_output, ref_mask = torch.ops.aten.fake_quantize_per_tensor_affine_cachemask(
        utils.to_reference(input), scale, zero_point, quant_min, quant_max
    )

    with flag_gems.use_gems():
        output, cachemask = torch.ops.aten.fake_quantize_per_tensor_affine_cachemask(
            input, scale, zero_point, quant_min, quant_max
        )

    utils.gems_assert_equal(output, ref_output)
    utils.gems_assert_equal(cachemask, ref_mask)


@pytest.mark.fake_quantize_per_tensor_affine_cachemask
@pytest.mark.parametrize("scale", [0.0, 1.0])
def test_accuracy_fake_quantize_per_tensor_affine_cachemask_boundaries(scale):
    input = torch.tensor(
        [float("-inf"), -3.5, -0.5, 0.5, 3.5, float("inf"), float("nan")],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    ref_output, ref_mask = torch.ops.aten.fake_quantize_per_tensor_affine_cachemask(
        utils.to_reference(input), scale, 0, -2, 2
    )

    with flag_gems.use_gems():
        output, cachemask = torch.ops.aten.fake_quantize_per_tensor_affine_cachemask(
            input, scale, 0, -2, 2
        )

    utils.gems_assert_equal(output, ref_output)
    utils.gems_assert_equal(cachemask, ref_mask)


@pytest.mark.fake_quantize_per_tensor_affine_cachemask_out
def test_accuracy_fake_quantize_per_tensor_affine_cachemask_out():
    input = torch.randn((8, 16), device=flag_gems.device)
    ref_output, ref_mask = torch.ops.aten.fake_quantize_per_tensor_affine_cachemask(
        utils.to_reference(input), 0.03, 3, 0, 255
    )
    out0 = torch.empty((16, 8), device=flag_gems.device).T
    out1 = torch.empty((16, 8), dtype=torch.bool, device=flag_gems.device).T

    with flag_gems.use_gems():
        result = torch.ops.aten.fake_quantize_per_tensor_affine_cachemask.out(
            input, 0.03, 3, 0, 255, out0=out0, out1=out1
        )

    assert result[0] is out0
    assert result[1] is out1
    utils.gems_assert_equal(out0, ref_output)
    utils.gems_assert_equal(out1, ref_mask)


@pytest.mark.fake_quantize_per_tensor_affine_cachemask
def test_accuracy_fake_quantize_per_tensor_affine_cachemask_noncontiguous():
    input = torch.randn((8, 16), device=flag_gems.device).T
    ref_output, ref_mask = torch.ops.aten.fake_quantize_per_tensor_affine_cachemask(
        utils.to_reference(input), 0.1, 0, 0, 255
    )

    with flag_gems.use_gems():
        output, cachemask = torch.ops.aten.fake_quantize_per_tensor_affine_cachemask(
            input, 0.1, 0, 0, 255
        )

    utils.gems_assert_equal(output, ref_output)
    utils.gems_assert_equal(cachemask, ref_mask)


@pytest.mark.fake_quantize_per_tensor_affine_cachemask
def test_accuracy_fake_quantize_per_tensor_affine_cachemask_empty():
    input = torch.empty((2, 0, 3), device=flag_gems.device)

    with flag_gems.use_gems():
        output, cachemask = torch.ops.aten.fake_quantize_per_tensor_affine_cachemask(
            input, 0.1, 0, 0, 255
        )

    assert output.shape == input.shape
    assert output.dtype == input.dtype
    assert cachemask.shape == input.shape
    assert cachemask.dtype == torch.bool


@pytest.mark.fake_quantize_per_tensor_affine_cachemask
@pytest.mark.parametrize("zero_point, quant_min, quant_max", [(256, 0, 255), (0, 2, 1)])
def test_fake_quantize_per_tensor_affine_cachemask_invalid_qparams(
    zero_point, quant_min, quant_max
):
    input = torch.ones(2, device=flag_gems.device)

    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        torch.ops.aten.fake_quantize_per_tensor_affine_cachemask(
            input, 0.1, zero_point, quant_min, quant_max
        )
