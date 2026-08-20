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


@pytest.mark.fake_quantize_per_channel_affine_cachemask_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fake_quantize_per_channel_affine_cachemask_backward(shape, dtype):
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_gems.device)
    ref_grad = utils.to_reference(grad)
    ref_mask = utils.to_reference(mask)

    ref_out = torch.ops.aten.fake_quantize_per_channel_affine_cachemask_backward(
        ref_grad, ref_mask
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.fake_quantize_per_channel_affine_cachemask_backward(
            grad, mask
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.fake_quantize_per_channel_affine_cachemask_backward
def test_fake_quantize_per_channel_affine_cachemask_backward_mask_semantics():
    grad = torch.tensor(
        [-3.0, -0.0, 1.5, float("inf"), float("nan")],
        device=flag_gems.device,
    )
    mask = torch.tensor([True, False, True, False, True], device=flag_gems.device)
    ref_out = torch.ops.aten.fake_quantize_per_channel_affine_cachemask_backward(
        utils.to_reference(grad), utils.to_reference(mask)
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.fake_quantize_per_channel_affine_cachemask_backward(
            grad, mask
        )

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.fake_quantize_per_channel_affine_cachemask_backward
def test_fake_quantize_per_channel_affine_cachemask_backward_noncontiguous():
    grad = torch.randn((2, 3, 5), device=flag_gems.device).transpose(0, 2)
    mask = torch.randint(
        0, 2, (2, 3, 5), dtype=torch.bool, device=flag_gems.device
    ).transpose(0, 2)
    ref_out = torch.ops.aten.fake_quantize_per_channel_affine_cachemask_backward(
        utils.to_reference(grad), utils.to_reference(mask)
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.fake_quantize_per_channel_affine_cachemask_backward(
            grad, mask
        )

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.fake_quantize_per_channel_affine_cachemask_backward
def test_fake_quantize_per_channel_affine_cachemask_backward_empty():
    grad = torch.empty((2, 0, 3), device=flag_gems.device)
    mask = torch.empty((2, 0, 3), dtype=torch.bool, device=flag_gems.device)

    with flag_gems.use_gems():
        output = torch.ops.aten.fake_quantize_per_channel_affine_cachemask_backward(
            grad, mask
        )

    assert output.shape == grad.shape
    assert output.dtype == grad.dtype
